"""
Case: General Conversational Agent — chat worker ↔ tool worker loop.

Uses the agent implementation from ``kgraphplanner.sample.general``.
See that package for the full architecture description, prompts, and
graph spec.
"""

from __future__ import annotations

import io
import os
import sys
import asyncio

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from kgraphplanner.sample.general import build_agent
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import (
    create_tool_manager, check_tools_available, refresh_jwt_token,
    log, write_log, save_png, execute_with_logging, OUTPUT_DIR,
)


# ============================================================
# Test requests
# ============================================================

TEST_REQUESTS = [
    {
        "input": "What's the weather like in New York City right now?",
        "description": "Weather lookup (tool call expected)",
    },
    {
        "input": "Search the web for the latest news about artificial intelligence.",
        "description": "Web search (tool call expected)",
    },
    {
        "input": "Where is the Statue of Liberty located?",
        "description": "Place search (tool call expected)",
    },
    {
        "input": "Hello! How are you doing today?",
        "description": "General greeting (direct response expected)",
    },
    {
        "input": "What's the current temperature in San Francisco?",
        "description": "Weather lookup #2 (tool call expected)",
    },
    {
        "input": "Find the top 5 rated restaurants in the world and then find the current weather in those places.",
        "description": "Multi-step planning (search + multiple weather lookups)",
    },
    {
        "input": "What are the top 5 largest music festivals in Europe and what is the weather like in those cities right now?",
        "description": "Multi-step planning #2 (web search + weather — non-restaurant domain)",
    },
]


# ============================================================
# Main entry point
# ============================================================

async def run(request_filter: list[int] | None = None) -> TestResult:
    """Run the general conversational agent test.
    
    Args:
        request_filter: Optional list of 1-based request numbers to run.
                        If None, all requests are run.
    """
    load_dotenv()
    buf = io.StringIO()

    # --- Setup ---
    log(buf, "  === Step 1: Setup ===")
    tm = create_tool_manager()
    if not check_tools_available(tm):
        return TestResult(
            name="General Agent (chat ↔ tool)",
            passed=True,
            details={"skipped": True, "reason": "No tools available (tool server not running?)"},
        )

    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.3)
    llm_low = ChatOpenAI(model="gpt-5-mini", temperature=0.3, reasoning_effort="low")
    log(buf, f"  LLM: gpt-5-mini  (summarization: reasoning_effort=low)")

    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)
    agent = build_agent(llm, tm, checkpointer, summarization_llm=llm_low)
    compiled = agent.get_compiled_graph()

    # --- Graph PNG ---
    log(buf, "\n  === Step 2: Graph Diagram ===")
    await save_png(compiled, "general_agent_graph.png", buf)

    # --- Agent info ---
    agent_info = agent.get_agent_info()
    log(buf, f"\n  Agent: {agent_info['name']}  type={agent_info['agent_type']}  "
            f"nodes={agent_info['node_count']}  edges={agent_info['edge_count']}")
    log(buf, f"  Worker nodes: {agent_info['worker_nodes']}")

    # --- Execute test requests ---
    if request_filter:
        requests_to_run = [(i, TEST_REQUESTS[i - 1]) for i in request_filter if 1 <= i <= len(TEST_REQUESTS)]
    else:
        requests_to_run = list(enumerate(TEST_REQUESTS, 1))

    log(buf, f"\n  === Step 3: Execute {len(requests_to_run)} Requests ===")
    sub_results = []

    for i, tc in requests_to_run:
        import time as _time
        user_input = tc["input"]
        desc = tc["description"]
        log(buf, f"\n  --- Request {i}: {desc} ---")
        log(buf, f"  Input: {user_input}")

        try:
            # Refresh JWT before each request (60s TTL)
            refresh_jwt_token(tm)

            _req_t0 = _time.time()
            config = {"configurable": {"thread_id": f"general-agent-test-{i}"}}
            messages = [HumanMessage(content=user_input)]

            result = await execute_with_logging(
                agent.arun(messages=messages, config=config),
                buf,
            )

            agent_data = result.get("agent_data", {})
            results_map = agent_data.get("results", {})

            # Extract results from each worker
            orch_result = results_map.get("orchestrator", {})
            tool_result = results_map.get("tool_executor", {})
            final_result = results_map.get("final_responder", {})
            final_text = final_result.get("result_text", "")

            orch_action = orch_result.get("action", "?")
            log(buf, f"  Orchestrator action: {orch_action}")
            if tool_result:
                tool_text = str(tool_result.get("result_text", ""))
                log(buf, f"  Tool result: {tool_text}")
            log(buf, f"  Final response: {final_text}")

            _req_elapsed = _time.time() - _req_t0
            log(buf, f"  ⏱️  Request {i} took {_req_elapsed:.1f}s")

            ok = bool(final_text and len(final_text) > 20)
            sub_results.append({
                "test": i, "desc": desc, "ok": ok, "elapsed": _req_elapsed,
            })

        except Exception as e:
            _req_elapsed = _time.time() - _req_t0
            log(buf, f"  ❌ Error after {_req_elapsed:.1f}s: {e}")
            import traceback
            log(buf, traceback.format_exc())
            sub_results.append({
                "test": i, "desc": desc, "ok": False, "error": str(e),
                "elapsed": _req_elapsed,
            })

        await asyncio.sleep(0.5)

    # --- Summary ---
    passed_count = sum(1 for r in sub_results if r["ok"])
    total_count = len(sub_results)

    log(buf, f"\n  === Results: {passed_count}/{total_count} requests succeeded ===")
    for r in sub_results:
        status = "✅" if r["ok"] else "❌"
        elapsed = r.get("elapsed", 0)
        log(buf, f"    {status} Test {r['test']}: {r['desc']} ({elapsed:.1f}s)")

    # --- Write log ---
    write_log(buf, "general_agent_run.log")

    return TestResult(
        name="General Agent (chat ↔ tool)",
        passed=passed_count == total_count,
        details={
            "requests_total": total_count,
            "requests_passed": passed_count,
            "graph_id": "general_agent",
            "worker_nodes": agent_info.get("worker_nodes", []),
            "output_dir": OUTPUT_DIR,
        },
    )
