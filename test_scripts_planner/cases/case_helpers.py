"""
Shared helpers for planner test cases.

Provides common infrastructure: Keycloak auth, ToolManager setup,
generic worker registry, logging, PNG generation, execution with
timing capture, and plan-derived assertions.
"""
from __future__ import annotations

import os
import sys
import io
import json
import logging as _logging
from datetime import datetime
from typing import Dict, List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import MermaidDrawMethod

from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager
from test_scripts_planner.cases.test_result import TestResult

OUTPUT_DIR = os.path.join(project_root, "test_output")


# --- Auth & tool setup ---

def get_keycloak_token():
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
        'scope': 'openid profile email'
    }
    if client_secret:
        data['client_secret'] = client_secret

    try:
        resp = requests.post(token_url, data=data, timeout=5)
        resp.raise_for_status()
        token = resp.json().get('access_token')
        if not token:
            return None, "No access_token in Keycloak response"
        return token, None
    except Exception as e:
        return None, f"Keycloak token request failed: {e}"


def create_tool_manager():
    """Create ToolManager with config from KGPLAN__ env vars, loaded tools, and JWT auth."""
    config = AgentConfig.from_env()
    tm = ToolManager(config=config)
    tm.load_tools_from_config()

    token, err = get_keycloak_token()
    if err:
        print(f"  JWT auth: {err}")
    else:
        tm.set_jwt_token(token)
        print(f"  JWT token set")

    return tm


# --- Generic worker registry ---

def make_generic_registry(
    exec_llm: ChatOpenAI,
    tool_manager: ToolManager,
    tool_ids: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Build a domain-agnostic worker registry.

    Workers have generic directives — the planner-generated ProgramSpec
    provides the domain-specific prompts and args at runtime.

      - research_worker: tool worker (web search, etc.)
      - analyst_a: chat worker (track A analysis)
      - analyst_b: chat worker (track B analysis)
      - aggregator: chat worker (combine analyses)
    """
    if tool_ids is None:
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


# --- Logging ---

def log(buf: io.StringIO, msg: str):
    """Print and write to log buffer."""
    print(msg)
    buf.write(msg + "\n")


def write_log(buf: io.StringIO, filename: str):
    """Flush log buffer to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(buf.getvalue())
    print(f"  Log written to {path}")


# --- PNG diagrams ---

async def save_png(compiled_graph, filename: str, buf: io.StringIO):
    """Generate and save a Mermaid PNG diagram."""
    import asyncio
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    try:
        graph_drawable = compiled_graph.get_graph()
        # Use API method to avoid pyppeteer Chrome atexit cleanup errors
        image_bytes = await asyncio.to_thread(
            graph_drawable.draw_mermaid_png,
            draw_method=MermaidDrawMethod.API
        )
        with open(path, "wb") as f:
            f.write(image_bytes)
        log(buf, f"  Diagram saved: {path} ({len(image_bytes)} bytes)")
    except Exception as e:
        log(buf, f"  Diagram generation failed ({filename}): {e}")


# --- Execution with timing capture ---

class _Tee:
    """Tee stdout to also write into a StringIO buffer."""
    def __init__(self, original, buf):
        self._orig = original
        self._buf = buf
    def write(self, data):
        self._orig.write(data)
        self._buf.write(data)
    def flush(self):
        self._orig.flush()


async def execute_with_timing(compiled, initial_state, thread_id: str, buf: io.StringIO):
    """
    Run the compiled graph, capturing both stdout and logger output
    into the log buffer for offline inspection.
    """
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, buf)

    log_handler = _logging.StreamHandler(buf)
    log_handler.setLevel(_logging.INFO)
    log_handler.setFormatter(_logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    _logging.getLogger().addHandler(log_handler)

    try:
        result = await compiled.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
    finally:
        sys.stdout = old_stdout
        _logging.getLogger().removeHandler(log_handler)

    return result


# --- Plan-derived assertions ---

def assert_plan_results(
    results: dict,
    errors: dict,
    graph_spec_dict: dict,
    buf: io.StringIO,
) -> dict:
    """
    Validate execution results against the plan.
    Returns a details dict for TestResult.
    """
    program_spec = graph_spec_dict  # caller passes the right dict

    # Extract expected worker IDs from graph spec
    worker_node_ids = [
        n["id"] for n in graph_spec_dict.get("nodes", [])
        if n.get("node_type") == "worker"
    ]
    end_edges = [e for e in graph_spec_dict.get("edges", []) if e["destination"] == "end"]
    aggregator_ids = [e["source"] for e in end_edges]

    # Check every worker node produced a result
    missing = [nid for nid in worker_node_ids if nid not in results]
    assert not missing, \
        f"Plan expected results for {worker_node_ids}, missing: {missing} (got: {list(results.keys())})"

    # Check aggregator produced a result
    agg_missing = [nid for nid in aggregator_ids if nid not in results]
    assert not agg_missing, \
        f"Aggregator node(s) {aggregator_ids} missing from results (got: {list(results.keys())})"

    log(buf, f"  Worker results: {len([n for n in worker_node_ids if n in results])}/{len(worker_node_ids)}")

    return {
        "expected_worker_nodes": len(worker_node_ids),
        "result_count": len(results),
        "worker_result_count": len([n for n in worker_node_ids if n in results]),
        "error_count": len(errors),
    }
