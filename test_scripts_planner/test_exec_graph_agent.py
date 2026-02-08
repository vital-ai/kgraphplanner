#!/usr/bin/env python3
"""
Integration test for KGraphExecGraphAgent.

Tests:
1. Simple linear graph: start → worker → end
2. Verifies entry node seeds start data, hop sets activation, worker produces result
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, Binding,
    WorkerNodeSpec, StartNodeSpec, EndNodeSpec
)


def create_simple_linear_graph() -> GraphSpec:
    """start → summarizer → end"""
    return GraphSpec(
        graph_id="simple_linear",
        name="Simple Linear Test",
        nodes=[
            StartNodeSpec(
                id="start",
                initial_data={"args": {"topic": "artificial intelligence"}}
            ),
            WorkerNodeSpec(
                id="summarizer",
                worker_name="chat_worker",
                defaults={"prompt": "Write a one-sentence summary about the given topic."}
            ),
            EndNodeSpec(id="end")
        ],
        edges=[
            EdgeSpec(
                source="start",
                destination="summarizer",
                prompt="Summarize this topic",
                bindings={
                    "topic": [Binding(from_node="start", path="$.topic")]
                }
            ),
            EdgeSpec(source="summarizer", destination="end")
        ],
        exit_points=["end"]
    )


def create_two_step_graph() -> GraphSpec:
    """start → researcher → analyst → end"""
    return GraphSpec(
        graph_id="two_step",
        name="Two Step Pipeline",
        nodes=[
            StartNodeSpec(
                id="start",
                initial_data={"args": {"company": "OpenAI"}}
            ),
            WorkerNodeSpec(
                id="researcher",
                worker_name="chat_worker",
                defaults={"prompt": "Briefly describe what this company does in 1-2 sentences."}
            ),
            WorkerNodeSpec(
                id="analyst",
                worker_name="chat_worker",
                defaults={"prompt": "Based on the research provided, give a one-sentence analysis."}
            ),
            EndNodeSpec(id="end")
        ],
        edges=[
            EdgeSpec(
                source="start",
                destination="researcher",
                prompt="Research this company",
                bindings={
                    "company": [Binding(from_node="start", path="$.company")]
                }
            ),
            EdgeSpec(
                source="researcher",
                destination="analyst",
                prompt="Analyze the research",
                bindings={
                    "research": [Binding(from_node="researcher", path="$.result_text")]
                }
            ),
            EdgeSpec(source="analyst", destination="end")
        ],
        exit_points=["end"]
    )


async def test_simple_linear():
    """Test a simple start → worker → end graph."""
    print("=" * 60)
    print("TEST 1: Simple Linear Graph (start → summarizer → end)")
    print("=" * 60)

    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=200)

    chat_worker = KGraphChatWorker(
        name="chat_worker",
        llm=llm,
        system_directive="You are a concise assistant. Keep responses brief."
    )

    graph_spec = create_simple_linear_graph()
    registry = {"chat_worker": chat_worker}

    agent = KGraphExecGraphAgent(
        name="test_exec_agent",
        graph_spec=graph_spec,
        worker_registry=registry,
        checkpointer=MemorySaver()
    )

    print(f"\nAgent info: {agent.get_agent_info()}")

    result = await agent.arun(
        messages=[],
        config={"configurable": {"thread_id": "test-linear-1"}}
    )

    agent_data = result.get("agent_data", {})
    results = agent_data.get("results", {})
    errors = agent_data.get("errors", {})

    print(f"\n--- Results ---")
    for node_id, res in results.items():
        if isinstance(res, dict) and "result_text" in res:
            print(f"  [{node_id}]: {res['result_text'][:200]}")
        else:
            print(f"  [{node_id}]: {res}")

    if errors:
        print(f"\n--- Errors ---")
        for node_id, err in errors.items():
            print(f"  [{node_id}]: {err}")

    assert "summarizer" in results, "Expected 'summarizer' in results"
    assert "result_text" in results["summarizer"], "Expected 'result_text' in summarizer result"
    print(f"\n✅ TEST 1 PASSED")
    return True


async def test_two_step_pipeline():
    """Test a two-step pipeline: start → researcher → analyst → end."""
    print("\n" + "=" * 60)
    print("TEST 2: Two Step Pipeline (start → researcher → analyst → end)")
    print("=" * 60)

    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=200)

    chat_worker = KGraphChatWorker(
        name="chat_worker",
        llm=llm,
        system_directive="You are a concise assistant. Keep responses brief."
    )

    graph_spec = create_two_step_graph()
    registry = {"chat_worker": chat_worker}

    agent = KGraphExecGraphAgent(
        name="test_exec_agent_2",
        graph_spec=graph_spec,
        worker_registry=registry,
        checkpointer=MemorySaver()
    )

    result = await agent.arun(
        messages=[],
        config={"configurable": {"thread_id": "test-pipeline-1"}}
    )

    agent_data = result.get("agent_data", {})
    results = agent_data.get("results", {})
    errors = agent_data.get("errors", {})

    print(f"\n--- Results ---")
    for node_id, res in results.items():
        if isinstance(res, dict) and "result_text" in res:
            print(f"  [{node_id}]: {res['result_text'][:200]}")
        else:
            print(f"  [{node_id}]: {res}")

    if errors:
        print(f"\n--- Errors ---")
        for node_id, err in errors.items():
            print(f"  [{node_id}]: {err}")

    assert "researcher" in results, "Expected 'researcher' in results"
    assert "analyst" in results, "Expected 'analyst' in results"
    print(f"\n✅ TEST 2 PASSED")
    return True


async def main():
    passed = 0
    failed = 0

    try:
        if await test_simple_linear():
            passed += 1
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    try:
        if await test_two_step_pipeline():
            passed += 1
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
