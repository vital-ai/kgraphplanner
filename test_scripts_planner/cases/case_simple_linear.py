"""
Case: Simple linear planner workflow.
start → researcher → summarizer → end

Tests the basic KGraphPlannerAgent pipeline:
  User Request → LLM ProgramSpec → Expand → Execute → Results
"""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from test_scripts_planner.cases.test_result import TestResult


async def run() -> TestResult:
    """Run the simple linear planner test."""
    load_dotenv()

    planner_llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    exec_llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=2000)

    registry = {
        "researcher": KGraphChatWorker(
            name="researcher", llm=exec_llm,
            system_directive="You are a research assistant. Provide concise factual information."
        ),
        "summarizer": KGraphChatWorker(
            name="summarizer", llm=exec_llm,
            system_directive="You are a summarizer. Condense information into brief summaries."
        ),
    }

    agent = KGraphPlannerAgent(
        name="test_planner_simple",
        planner_llm=planner_llm,
        worker_registry=registry,
        execution_llm=exec_llm,
        checkpointer=MemorySaver()
    )

    result = await agent.arun(
        messages=[HumanMessage(content="Research what OpenAI does, then summarize the findings.")],
        config={"configurable": {"thread_id": "case-simple-1"}}
    )

    program_spec = result.get("program_spec")
    graph_spec = result.get("graph_spec")
    agent_data = result.get("agent_data", {})
    results = agent_data.get("results", {})
    errors = agent_data.get("errors", {})

    details = {
        "program_id": program_spec.get("program_id") if program_spec else None,
        "node_count": len(graph_spec.get("nodes", [])) if graph_spec else 0,
        "edge_count": len(graph_spec.get("edges", [])) if graph_spec else 0,
        "result_count": len(results),
        "error_count": len(errors),
        "result_keys": sorted(results.keys()),
    }

    # Print details
    print(f"  program_id: {details['program_id']}")
    print(f"  nodes: {details['node_count']}, edges: {details['edge_count']}")
    print(f"  results: {details['result_keys']}")
    for node_id, res in sorted(results.items()):
        if isinstance(res, dict) and "result_text" in res:
            print(f"  [{node_id}] (len={len(res['result_text'])}): {res['result_text'][:120]}")
        else:
            print(f"  [{node_id}]: {str(res)[:120]}")

    # Assertions
    assert program_spec is not None, "Expected program_spec"
    assert graph_spec is not None, "Expected graph_spec"
    assert len(results) > 1, f"Expected at least 2 results (start + workers), got {len(results)}"
    assert len(errors) == 0, f"Expected no errors, got {errors}"

    return TestResult(name="Simple Linear Planner", passed=True, details=details)
