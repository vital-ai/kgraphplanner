"""
Case: LLM-driven multi-company analysis with full feature set.
Asks the planner to generate a ProgramSpec with templates for multiple
companies, fan-out to parallel analysis tracks, and fan-in aggregation.
Mirrors the original test_planner.py demo scenario.

Tests: LLM ProgramSpec generation with templates, fan-out, fan-in,
       end-to-end execution of LLM-planned graph.
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


def make_full_registry(exec_llm):
    """Create a worker registry matching the original test_planner.py demo."""
    return {
        "research_worker": KGraphChatWorker(
            name="research_worker", llm=exec_llm,
            system_directive="Research the company using available knowledge; produce factual notes."
        ),
        "analyst_a": KGraphChatWorker(
            name="analyst_a", llm=exec_llm,
            system_directive="Analyze input A; produce concise structured notes on strengths and products."
        ),
        "analyst_b": KGraphChatWorker(
            name="analyst_b", llm=exec_llm,
            system_directive="Analyze input B; produce concise structured notes on risks and competition."
        ),
        "aggregator": KGraphChatWorker(
            name="aggregator", llm=exec_llm,
            system_directive="Aggregate a list of analyses into a single combined summary report."
        )
    }


async def run() -> TestResult:
    """Run the LLM-driven multi-company analysis test."""
    load_dotenv()

    planner_llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    exec_llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=2000)

    registry = make_full_registry(exec_llm)

    agent = KGraphPlannerAgent(
        name="test_planner_full",
        planner_llm=planner_llm,
        worker_registry=registry,
        execution_llm=exec_llm,
        checkpointer=MemorySaver()
    )

    result = await agent.arun(
        messages=[HumanMessage(
            content="Research 3 AI companies (OpenAI, Anthropic, Cohere), "
                    "analyze each via two parallel tracks (analyst_a for strengths/products, "
                    "analyst_b for risks/competition), then aggregate all analyses into a combined report."
        )],
        config={"configurable": {"thread_id": "case-planner-full-1"}}
    )

    program_spec = result.get("program_spec")
    graph_spec = result.get("graph_spec")
    agent_data = result.get("agent_data", {})
    results = agent_data.get("results", {})
    errors = agent_data.get("errors", {})

    # Print details
    if program_spec:
        templates = program_spec.get("templates", [])
        print(f"  program_id: {program_spec.get('program_id')}")
        print(f"  static_nodes: {len(program_spec.get('static_nodes', []))}")
        print(f"  templates: {len(templates)}")
        for t in templates:
            print(f"    template '{t.get('name')}': "
                  f"{len(t.get('loop_nodes', []))} loop_nodes, "
                  f"{len(t.get('fan_out', []))} fan_out, "
                  f"{len(t.get('fan_in', []))} fan_in")
    else:
        print("  (planning failed)")

    if graph_spec:
        node_ids = [n.get("id") for n in graph_spec.get("nodes", [])]
        print(f"  GraphSpec: {len(graph_spec.get('nodes', []))} nodes, {len(graph_spec.get('edges', []))} edges")
        print(f"  Node IDs: {node_ids}")

    print(f"  Results ({len(results)} total):")
    for node_id, res in sorted(results.items()):
        if isinstance(res, dict) and "result_text" in res:
            text = res["result_text"]
            print(f"    [{node_id}] (len={len(text)}): {text[:100]}")
        else:
            print(f"    [{node_id}]: {str(res)[:100]}")

    if errors:
        print(f"  Errors:")
        for node_id, err in errors.items():
            print(f"    [{node_id}]: {err}")

    # Assertions
    assert program_spec is not None, "Expected program_spec"
    assert graph_spec is not None, "Expected graph_spec"

    templates = program_spec.get("templates", [])
    assert len(templates) >= 1, f"Expected at least 1 template, got {len(templates)}"

    has_fan_out = any(len(t.get("fan_out", [])) > 0 for t in templates)
    has_fan_in = any(len(t.get("fan_in", [])) > 0 for t in templates)

    worker_results = {k: v for k, v in results.items() if k != "start"}
    assert len(worker_results) >= 4, \
        f"Expected at least 4 worker results (3 research + aggregator), got {len(worker_results)}: {list(worker_results.keys())}"
    assert len(errors) == 0, f"Expected no errors, got {errors}"

    details = {
        "program_id": program_spec.get("program_id"),
        "template_count": len(templates),
        "has_fan_out": has_fan_out,
        "has_fan_in": has_fan_in,
        "node_count": len(graph_spec.get("nodes", [])),
        "edge_count": len(graph_spec.get("edges", [])),
        "result_count": len(results),
        "worker_result_count": len(worker_results),
        "error_count": len(errors),
    }

    return TestResult(
        name="LLM Planner Full Features (multi-company analysis)",
        passed=True, details=details
    )
