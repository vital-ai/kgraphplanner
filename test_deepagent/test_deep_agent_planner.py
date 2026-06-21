"""Test: Deep Agent with KGraphPlanner as a SubAgent.

The top-level Deep Agent handles conversation and simple tool queries.
When a complex multi-step research/analysis task is requested, it delegates
to the KGraphPlanner sub-agent which:
  1. Plans a ProgramSpec (LLM-generated workflow)
  2. Expands it into a GraphSpec
  3. Executes it via parallel workers (tool + chat)
  4. Returns the aggregated result

This demonstrates the CompiledSubAgent pattern where the sub-agent's
runnable wraps the full KGraphPlannerAgent pipeline.

Prerequisites:
    - OPENAI_API_KEY in .env
    - Tool server running at configured endpoint (default http://localhost:8008)
    - Keycloak for JWT auth

Usage:
    python test_deepagent/test_deep_agent_planner.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", message=".*NotRequired.*", category=UserWarning)

import httpx
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer

from deepagents import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent

from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.sample.auto_tools import get_kgraph_tools
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker


# ============================================================
# Helpers
# ============================================================

class TeeWriter:
    """Duplicate writes to both a file and the original stream."""

    def __init__(self, log_path: str, original):
        self._file = open(log_path, "w", encoding="utf-8")
        self._original = original

    def write(self, data):
        self._original.write(data)
        self._original.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def print_header(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


# ============================================================
# Auth
# ============================================================

async def get_keycloak_token():
    """Get JWT token from Keycloak using env credentials."""
    username = os.getenv('KEYCLOAK_USER')
    password = os.getenv('KEYCLOAK_PASSWORD')
    realm = os.getenv('KEYCLOAK_REALM')
    client_id = os.getenv('KEYCLOAK_CLIENT_ID')
    client_secret = os.getenv('KEYCLOAK_CLIENT_SECRET')

    if not username or not password:
        return None, "KEYCLOAK_USER/KEYCLOAK_PASSWORD not set"

    token_url = f"http://localhost:8085/realms/{realm}/protocol/openid-connect/token"
    data = {
        'grant_type': 'password',
        'client_id': client_id,
        'username': username,
        'password': password,
        'scope': 'openid profile email',
    }
    if client_secret:
        data['client_secret'] = client_secret

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(token_url, data=data, timeout=5)
        resp.raise_for_status()
        token = resp.json().get('access_token')
        if not token:
            return None, "No access_token in Keycloak response"
        return token, None
    except Exception as e:
        return None, f"Keycloak token request failed: {e}"


async def create_authenticated_tool_manager() -> ToolManager:
    """Create ToolManager with JWT auth from Keycloak."""
    config = AgentConfig.from_env()
    tm = ToolManager(config=config)
    tm.load_tools_from_config()

    token, err = await get_keycloak_token()
    if err:
        print(f"  JWT auth: {err}")
    else:
        tm.set_jwt_token(token)
        print(f"  JWT token set")

    tools = tm.list_available_tools()
    print(f"  Available tools: {tools}")
    return tm


# ============================================================
# Worker Registry (same as planner test cases)
# ============================================================

def make_worker_registry(
    exec_llm: ChatOpenAI,
    tool_manager: ToolManager,
) -> Dict[str, Any]:
    """Build a domain-agnostic worker registry for the planner."""
    available_tools = tool_manager.list_available_tools()
    web_tool_ids = [t for t in available_tools if "web_search" in t]
    tool_ids = web_tool_ids if web_tool_ids else available_tools[:1]

    return {
        "research_worker": KGraphToolWorker(
            name="research_worker",
            llm=exec_llm,
            system_directive=(
                "Use available tools to research the given topic. "
                "Your output MUST include specific named entities "
                "(restaurants, places, businesses, etc.) — not just "
                "general reference links or city overviews. "
                "Search for concrete recommendations, reviews, and "
                "ranked lists. Include names, addresses/neighborhoods, "
                "and source URLs."
            ),
            tool_manager=tool_manager,
            available_tool_ids=tool_ids,
        ),
        "analyst_a": KGraphChatWorker(
            name="analyst_a",
            llm=exec_llm,
            system_directive=(
                "Analyze the provided input from perspective A (as described in "
                "your task instructions). Produce concise, structured notes."
            ),
        ),
        "analyst_b": KGraphChatWorker(
            name="analyst_b",
            llm=exec_llm,
            system_directive=(
                "Analyze the provided input from perspective B (as described in "
                "your task instructions). Produce concise, structured notes."
            ),
        ),
        "aggregator": KGraphChatWorker(
            name="aggregator",
            llm=exec_llm,
            system_directive=(
                "Combine all provided analyses into a single cohesive "
                "summary report, organized clearly."
            ),
        ),
    }


# ============================================================
# PlannerRunnable — wraps KGraphPlannerAgent as a Runnable
# ============================================================

class PlannerRunnable(RunnableSerializable):
    """Wraps KGraphPlannerAgent as a LangChain Runnable for CompiledSubAgent.

    The Deep Agent framework invokes sub-agents by passing state with a
    'messages' key. This runnable:
      1. Extracts the task description from the last HumanMessage
      2. Runs plan → expand → execute via KGraphPlannerAgent.arun()
      3. Returns state with the aggregated result as an AIMessage
    """

    planner_agent: KGraphPlannerAgent

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, input: Any, config=None, **kwargs) -> dict:
        """Sync invoke — delegates to async."""
        return asyncio.run(self.ainvoke(input, config=config, **kwargs))

    async def ainvoke(self, input: Any, config=None, **kwargs) -> dict:
        """Run the full planner pipeline and return results as messages."""
        # Extract the task prompt from the sub-agent invocation
        messages = input.get("messages", [])
        prompt = ""
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if content:
                prompt = content
                break

        if not prompt:
            return {"messages": [AIMessage(content="No task description provided.")]}

        print(f"\n  [PlannerRunnable] Received task:")
        print(f"  {prompt}")

        # Run plan + expand manually so we can log intermediate artifacts
        planner_config = config or {"configurable": {"thread_id": "planner-sub"}}

        # Step 1: Plan
        print(f"\n  === Sub-Agent Step 1: Plan ===")
        program = await self.planner_agent.plan([HumanMessage(content=prompt)])
        program_spec = program.model_dump()
        print(f"  program_id: {program_spec.get('program_id')}")
        print(f"  templates: {len(program_spec.get('templates', []))}")
        print(f"  static_nodes: {len(program_spec.get('static_nodes', []))}")
        print(f"\n  --- ProgramSpec (full JSON) ---")
        print(json.dumps(program_spec, indent=2, default=str))

        # Step 2: Expand
        print(f"\n  === Sub-Agent Step 2: Expand ===")
        graph_spec = self.planner_agent.expand(program)
        graph_spec_dict = graph_spec.model_dump()
        node_ids = [n.get("id") for n in graph_spec_dict.get("nodes", [])]
        print(f"  GraphSpec: {len(graph_spec_dict.get('nodes', []))} nodes, "
              f"{len(graph_spec_dict.get('edges', []))} edges")
        print(f"  Node IDs: {node_ids}")
        print(f"\n  --- GraphSpec (full JSON) ---")
        print(json.dumps(graph_spec_dict, indent=2, default=str))

        # Step 3: Execute
        print(f"\n  === Sub-Agent Step 3: Execute ===")
        exec_graph = self.planner_agent._build_exec_graph(graph_spec)
        compiled = exec_graph.compile(checkpointer=self.planner_agent.checkpointer)

        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "agent_data": {},
            "work": {}
        }

        result = await compiled.ainvoke(initial_state, config=planner_config)

        # Attach specs
        result["program_spec"] = program_spec
        result["graph_spec"] = graph_spec_dict

        # Extract results
        agent_data = result.get("agent_data", {})
        results = agent_data.get("results", {})
        errors = agent_data.get("errors", {})

        print(f"\n  === Sub-Agent Execution Complete ===")
        print(f"  Results: {len(results)}, Errors: {len(errors)}")

        # Print full results
        print(f"\n  --- Worker Results (full) ---")
        for node_id, res in sorted(results.items()):
            if isinstance(res, dict) and "result_text" in res:
                text = res["result_text"]
                print(f"\n  [{node_id}] (len={len(text)}):")
                print(f"  {text}")
            else:
                print(f"\n  [{node_id}]: {res}")

        if errors:
            print(f"\n  --- Errors ---")
            for node_id, err in errors.items():
                print(f"  [{node_id}]: {err}")

        # Find the aggregator result (fan-in node that feeds into 'end')
        end_edges = [e for e in graph_spec_dict.get("edges", []) if e.get("destination") == "end"]
        aggregator_ids = [e["source"] for e in end_edges]

        # Prefer aggregator result, fall back to concatenating all results
        combined_text = ""
        for agg_id in aggregator_ids:
            if agg_id in results:
                res = results[agg_id]
                if isinstance(res, dict) and "result_text" in res:
                    combined_text = res["result_text"]
                else:
                    combined_text = str(res)
                break

        if not combined_text:
            # Concatenate all worker results
            parts = []
            for node_id, res in sorted(results.items()):
                if node_id == "start":
                    continue
                if isinstance(res, dict) and "result_text" in res:
                    parts.append(f"## {node_id}\n{res['result_text']}")
                elif isinstance(res, str):
                    parts.append(f"## {node_id}\n{res}")
            combined_text = "\n\n".join(parts) if parts else "No results produced."

        # Include metadata
        program_id = program_spec.get("program_id", "")

        summary_prefix = (
            f"[Planner executed: program_id={program_id}, "
            f"{len(results)} results, {len(errors)} errors]\n\n"
        )

        return {"messages": [AIMessage(content=summary_prefix + combined_text)]}


# ============================================================
# Top-level system prompt
# ============================================================

TOP_LEVEL_SYSTEM_PROMPT = """\
You are a research and conversational assistant.

For **simple queries** (weather in one city, looking up a single place,
quick web search), use your tools directly.

For **complex research tasks** that involve:
- Multi-entity comparisons (multiple companies, cities, topics)
- Parallel research with dual-track analysis
- Tasks that benefit from structured fan-out/fan-in workflows
- Restaurant recommendations across multiple cities

...delegate to the `research_planner` sub-agent by using the task tool.
Pass the user's full request as the task description.

After receiving results from the planner, present them clearly to the user.
You may summarize, reformat, or highlight key findings.
"""


# ============================================================
# Test cases
# ============================================================

TEST_CASES = [
    {
        "name": "simple_weather",
        "query": "What's the current weather in San Francisco?",
        "expect_subagent": False,
        "expect_keywords": ["san francisco", "temperature"],
        "description": "Simple query — should be handled directly by tools, not sub-agent",
        "timeout": 30,
    },
    {
        "name": "restaurant_planner",
        "query": (
            "Recommend restaurants in 2 cities: Tokyo and Paris. "
            "For each city, research top-rated restaurants, then analyze "
            "from two tracks: (A) cuisine quality and signature dishes, "
            "(B) ambiance and value. Produce a combined dining guide. "
            "Keep it concise — 5 restaurants per city maximum."
        ),
        "expect_subagent": True,
        "expect_keywords": ["tokyo", "paris"],
        "description": "Complex task — should delegate to planner sub-agent",
        "timeout": 300,
    },
]


# ============================================================
# Result extraction
# ============================================================

def extract_response_text(result: dict) -> str:
    """Extract the final AI response text from a Deep Agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if role == "ai" and content:
            if isinstance(content, list):
                text = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            else:
                text = str(content)
            if text.strip():
                return text.strip()
    return ""


def count_tool_calls(result: dict) -> int:
    """Count how many tool calls were made during the agent run."""
    count = 0
    messages = result.get("messages", [])
    for msg in messages:
        if getattr(msg, "type", "") == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                count += len(tool_calls)
    return count


def used_subagent(result: dict) -> bool:
    """Check if the 'task' tool was called (indicates sub-agent delegation)."""
    messages = result.get("messages", [])
    for msg in messages:
        if getattr(msg, "type", "") == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    if tc.get("name") == "task":
                        return True
    return False


# ============================================================
# Main
# ============================================================

RECURSION_LIMIT = 50


async def main():
    # Set up log file
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_deep_agent_planner_{ts}.log")

    tee = TeeWriter(log_file, sys.stdout)
    sys.stdout = tee

    print(f"  Logging to {os.path.abspath(log_file)}")

    # Enable real-time logging for tool calls and worker activity
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    # Reduce noise from httpx but keep tool/worker logs
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    print_header("DEEP AGENT + KGRAPHPLANNER SUB-AGENT TEST")

    # --- Check prerequisites ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  ERROR: OPENAI_API_KEY not set. Exiting.")
        return False

    # --- Auth & tools ---
    print_header("Setting up ToolManager")
    tool_manager = await create_authenticated_tool_manager()

    available_tools = tool_manager.list_available_tools()
    if not available_tools:
        print("  ERROR: No tools available. Exiting.")
        return False

    # --- Build the planner sub-agent ---
    print_header("Building KGraphPlanner SubAgent")

    exec_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    registry = make_worker_registry(exec_llm, tool_manager)
    print(f"  Worker registry: {list(registry.keys())}")

    planner_agent = KGraphPlannerAgent(
        name="research_planner",
        planner_llm=planner_llm,
        worker_registry=registry,
        execution_llm=exec_llm,
        checkpointer=KGraphMemoryCheckpointer(),
    )

    planner_runnable = PlannerRunnable(planner_agent=planner_agent)
    print(f"  PlannerRunnable built")

    # --- Build the top-level Deep Agent ---
    print_header("Building Top-Level Deep Agent")

    # Direct tools for simple queries
    kgraph_tools = get_kgraph_tools(tool_manager)
    print(f"  Direct tools: {[t.name for t in kgraph_tools]}")

    agent = create_deep_agent(
        model="openai:gpt-4o-mini",
        tools=kgraph_tools,
        subagents=[
            CompiledSubAgent(
                name="research_planner",
                description=(
                    "A research and analysis planner that executes complex "
                    "multi-step workflows. Use this for: multi-entity comparisons, "
                    "restaurant recommendations across multiple cities, parallel "
                    "research with dual-track analysis, or any task requiring "
                    "structured fan-out/fan-in processing. "
                    "Pass the user's full request as the task description."
                ),
                runnable=planner_runnable,
            ),
        ],
        system_prompt=TOP_LEVEL_SYSTEM_PROMPT,
        checkpointer=KGraphMemoryCheckpointer(),
    )
    print(f"  Top-level agent built")

    # --- Run test cases ---
    results = []
    total_start = time.time()

    for i, tc in enumerate(TEST_CASES, 1):
        print_header(f"TEST {i}/{len(TEST_CASES)}: {tc['name']} — {tc['description']}")
        print(f"  Query: {tc['query']}\n")

        t_start = time.time()
        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [{"role": "user", "content": tc["query"]}]},
                    config={
                        "configurable": {"thread_id": f"test-{tc['name']}"},
                        "recursion_limit": RECURSION_LIMIT,
                    },
                ),
                timeout=tc["timeout"],
            )
            elapsed = time.time() - t_start

            response_text = extract_response_text(result)
            tool_call_count = count_tool_calls(result)
            did_use_subagent = used_subagent(result)

            # Check keywords
            lower_text = response_text.lower()
            kw_missing = [kw for kw in tc["expect_keywords"] if kw.lower() not in lower_text]
            kw_pass = len(kw_missing) == 0

            print(f"  Time: {elapsed:.1f}s")
            print(f"  Tool calls: {tool_call_count}")
            print(f"  Used sub-agent: {did_use_subagent} (expected: {tc['expect_subagent']})")
            print(f"  Keywords pass: {kw_pass}")
            if kw_missing:
                print(f"  Missing keywords: {kw_missing}")
            print(f"  Response length: {len(response_text)}")
            print(f"\n  [Response]\n")
            for line in response_text.split("\n"):
                print(f"    {line}")

            passed = kw_pass and len(response_text) > 50
            results.append({
                "name": tc["name"],
                "passed": passed,
                "elapsed": elapsed,
                "tool_calls": tool_call_count,
                "used_subagent": did_use_subagent,
                "kw_pass": kw_pass,
                "kw_missing": kw_missing,
                "response_len": len(response_text),
                "error": None,
            })

        except asyncio.TimeoutError:
            elapsed = time.time() - t_start
            print(f"  TIMEOUT after {elapsed:.1f}s (limit={tc['timeout']}s)")
            results.append({
                "name": tc["name"],
                "passed": False,
                "elapsed": elapsed,
                "tool_calls": 0,
                "used_subagent": False,
                "kw_pass": False,
                "kw_missing": tc["expect_keywords"],
                "response_len": 0,
                "error": f"Timeout after {tc['timeout']}s",
            })

        except Exception as e:
            elapsed = time.time() - t_start
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "name": tc["name"],
                "passed": False,
                "elapsed": elapsed,
                "tool_calls": 0,
                "used_subagent": False,
                "kw_pass": False,
                "kw_missing": tc["expect_keywords"],
                "response_len": 0,
                "error": str(e),
            })

    total_elapsed = time.time() - total_start

    # --- Summary ---
    print_header("TEST SUMMARY")
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        sub = " (sub-agent)" if r["used_subagent"] else " (direct)"
        error_info = f"  error={r['error']}" if r["error"] else ""
        print(
            f"  [{status}]  {r['name']:<30s}  "
            f"{r['elapsed']:5.1f}s  "
            f"tools={r['tool_calls']}{sub}{error_info}"
        )

    print(f"\n  Total: {passed_count} passed, {failed_count} failed, {total_elapsed:.1f}s elapsed")
    print(f"  Log:   {os.path.abspath(log_file)}")

    tee.close()
    sys.stdout = tee._original
    return failed_count == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
