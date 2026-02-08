"""
Case: KGraphToolWorker in a planner-style exec graph pipeline.

Uses the real ToolManager with web_search tool loaded from config.
Builds a GraphSpec manually: start → tool_worker → summarizer → end
The tool worker researches a topic, then the chat summarizer condenses it.

Requires: KGPLAN__ env vars, Keycloak credentials in env, tool server running.
If tools or auth are unavailable, the test is skipped (not failed).
"""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, Binding,
    WorkerNodeSpec, StartNodeSpec, EndNodeSpec,
)
from test_scripts_planner.cases.test_result import TestResult


def _get_keycloak_token():
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


def _create_tool_manager():
    """Create ToolManager with config from KGPLAN__ env vars, loaded tools, and JWT auth."""
    config = AgentConfig.from_env()
    tm = ToolManager(config=config)
    tm.load_tools_from_config()

    token, err = _get_keycloak_token()
    if err:
        print(f"  JWT auth: {err}")
    else:
        tm.set_jwt_token(token)
        print(f"  JWT token set")

    return tm


def _build_tool_pipeline_graph() -> GraphSpec:
    """
    Build a simple GraphSpec:
      start → researcher (tool worker) → summarizer (chat worker) → end
    """
    return GraphSpec(
        graph_id="tool_pipeline",
        nodes=[
            StartNodeSpec(id="start", initial_data={"args": {"topic": "quantum computing"}}),
            WorkerNodeSpec(id="researcher", worker_name="researcher",
                          defaults={"prompt": "Research quantum computing using web search."}),
            WorkerNodeSpec(id="summarizer", worker_name="summarizer",
                          defaults={"prompt": "Summarize the research findings concisely."}),
            EndNodeSpec(id="end"),
        ],
        edges=[
            EdgeSpec(source="start", destination="researcher",
                     bindings={"request": [Binding(from_node="start", path="$.topic")]}),
            EdgeSpec(source="researcher", destination="summarizer",
                     bindings={"input": [Binding(from_node="researcher", path="$.result_text")]}),
            EdgeSpec(source="summarizer", destination="end"),
        ],
        exit_points=["end"],
    )


async def run() -> TestResult:
    """Run the tool worker pipeline test."""
    load_dotenv()

    exec_llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=2000)

    # Set up ToolManager with config, tools, and JWT auth
    try:
        tm = _create_tool_manager()
        available = tm.list_available_tools()
        print(f"  Available tools: {available}")
    except Exception as e:
        return TestResult(
            name="Tool Worker Pipeline",
            passed=True,
            details={"skipped": True, "reason": f"ToolManager init failed: {e}"}
        )

    if not available:
        return TestResult(
            name="Tool Worker Pipeline",
            passed=True,
            details={"skipped": True, "reason": "No tools available (tool server not running?)"}
        )

    # Pick the first available tool
    tool_id = available[0]
    print(f"  Using tool: {tool_id}")

    # Build workers
    registry = {
        "researcher": KGraphToolWorker(
            name="researcher",
            llm=exec_llm,
            system_directive="You are a research assistant. Use available tools to find information.",
            tool_manager=tm,
            available_tool_ids=[tool_id],
        ),
        "summarizer": KGraphChatWorker(
            name="summarizer",
            llm=exec_llm,
            system_directive="Summarize the provided research concisely.",
        ),
    }

    graph_spec = _build_tool_pipeline_graph()

    agent = KGraphExecGraphAgent(
        name="test_tool_pipeline",
        graph_spec=graph_spec,
        worker_registry=registry,
        checkpointer=MemorySaver(),
    )

    result = await agent.arun(
        messages=[],
        config={"configurable": {"thread_id": "case-tool-worker-1"}}
    )

    agent_data = result.get("agent_data", {})
    results = agent_data.get("results", {})
    errors = agent_data.get("errors", {})

    print(f"  Results ({len(results)} total):")
    for node_id, res in sorted(results.items()):
        if isinstance(res, dict) and "result_text" in res:
            text = res["result_text"]
            print(f"    [{node_id}] (len={len(text)}): {text[:120]}")
        else:
            print(f"    [{node_id}]: {str(res)[:120]}")

    if errors:
        print(f"  Errors:")
        for node_id, err in errors.items():
            print(f"    [{node_id}]: {err}")

    # Validate
    assert "researcher" in results, f"Expected 'researcher' in results, got {list(results.keys())}"
    assert "summarizer" in results, f"Expected 'summarizer' in results, got {list(results.keys())}"

    researcher_rt = results["researcher"].get("result_text", "") if isinstance(results["researcher"], dict) else ""
    summarizer_rt = results["summarizer"].get("result_text", "") if isinstance(results["summarizer"], dict) else ""

    assert len(researcher_rt) > 0, "Expected non-empty researcher result_text"
    assert len(summarizer_rt) > 0, "Expected non-empty summarizer result_text"

    details = {
        "tool_used": tool_id,
        "researcher_len": len(researcher_rt),
        "summarizer_len": len(summarizer_rt),
        "result_count": len(results),
        "error_count": len(errors),
    }

    return TestResult(name="Tool Worker Pipeline", passed=True, details=details)
