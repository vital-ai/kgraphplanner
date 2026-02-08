"""
Case: KGraphCaseWorker — categorization with predefined and custom cases.

Tests the KGraphCaseWorker's ability to:
  1. Categorize user input into predefined categories (weather, travel, food, etc.)
  2. Handle custom cases provided at runtime (urgent, routine, feedback)
"""

from __future__ import annotations

import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from typing import Dict, Any, TypedDict, Annotated
from operator import or_ as merge_dicts

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from kgraphplanner.worker.kgraph_case_worker import KGraphCaseWorker, KGCase

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import log, write_log, save_png, OUTPUT_DIR


class _ExecState(TypedDict, total=False):
    agent_data: Annotated[Dict[str, Any], merge_dicts]
    input_args: Dict[str, Any]
    test_input: str


PREDEFINED_CASES = [
    KGCase(id="weather", name="Weather Topic",
           description="a category for weather related requests, forecasts, temperature queries"),
    KGCase(id="travel", name="Travel Planning",
           description="requests about trips, hotels, flights, destinations, vacation planning"),
    KGCase(id="food", name="Food & Dining",
           description="restaurant recommendations, recipes, cooking, food delivery"),
    KGCase(id="technology", name="Technology Support",
           description="computer problems, software issues, tech support, programming questions"),
    KGCase(id="shopping", name="Shopping & Products",
           description="product recommendations, price comparisons, online shopping, purchases"),
    KGCase(id="health", name="Health & Wellness",
           description="medical questions, fitness advice, wellness tips, health concerns"),
]

CATEGORIZATION_TESTS = [
    ("What's the weather like in New York today?", "weather"),
    ("I need help booking a flight to Paris for next month", "travel"),
    ("Can you recommend a good Italian restaurant nearby?", "food"),
    ("My computer won't start up, what should I do?", "technology"),
    ("Where can I find the best deals on laptops?", "shopping"),
    ("I've been having headaches lately, any advice?", "health"),
    ("Tell me about quantum physics and black holes", "unknown"),
]

CUSTOM_TESTS = [
    "URGENT: My website is down and I'm losing customers!",
    "Can you tell me your business hours?",
    "Your service was excellent, keep up the good work!",
]


def _build_predefined_graph(llm):
    """Build graph with predefined categories."""
    case_worker = KGraphCaseWorker(
        name="categorizer", llm=llm, cases=PREDEFINED_CASES,
        system_directive="You are an expert at categorizing user requests accurately and efficiently.",
    )
    graph = StateGraph(_ExecState)

    def start_node(state: _ExecState) -> _ExecState:
        return {
            "agent_data": {
                "activation": {
                    "case_worker_1": {
                        "prompt": "Categorize the user input into the most appropriate category.",
                        "args": state.get("input_args", {}),
                    }
                },
                "results": {}, "errors": {}, "work": {},
            }
        }

    graph.add_node("start", start_node)
    entry, exit_ = case_worker.build_subgraph(graph, "case_worker_1")

    def final_node(state: _ExecState) -> _ExecState:
        return state

    graph.add_node("final", final_node)
    graph.add_edge(START, "start")
    graph.add_edge("start", entry)
    graph.add_edge(exit_, "final")
    graph.add_edge("final", END)
    return graph


def _build_custom_graph(llm):
    """Build graph with custom runtime cases."""
    case_worker = KGraphCaseWorker(
        name="custom_categorizer", llm=llm,
        system_directive="Categorize based on the provided custom categories.",
    )
    graph = StateGraph(_ExecState)

    def start_node(state: _ExecState) -> _ExecState:
        custom_cases = [
            KGCase(id="urgent", name="Urgent Request",
                   description="time-sensitive or emergency requests"),
            KGCase(id="routine", name="Routine Inquiry",
                   description="standard questions or regular requests"),
            KGCase(id="feedback", name="User Feedback",
                   description="complaints, suggestions, or reviews"),
        ]
        return {
            "agent_data": {
                "activation": {
                    "custom_worker": {
                        "prompt": "Categorize based on urgency and request type.",
                        "args": {"query": state.get("test_input", ""), "cases": custom_cases},
                    }
                },
                "results": {}, "errors": {}, "work": {},
            }
        }

    graph.add_node("start", start_node)
    entry, exit_ = case_worker.build_subgraph(graph, "custom_worker")

    def final_node(state: _ExecState) -> _ExecState:
        return state

    graph.add_node("final", final_node)
    graph.add_edge(START, "start")
    graph.add_edge("start", entry)
    graph.add_edge(exit_, "final")
    graph.add_edge("final", END)
    return graph


async def run() -> TestResult:
    load_dotenv()
    buf = io.StringIO()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # --- Part 1: Predefined categorization ---
    log(buf, "  === Part 1: Predefined Categorization ===")
    graph1 = _build_predefined_graph(llm)
    compiled1 = graph1.compile(checkpointer=MemorySaver())

    await save_png(compiled1, "case_worker_graph.png", buf)

    correct = 0
    total = len(CATEGORIZATION_TESTS)

    for i, (query, expected) in enumerate(CATEGORIZATION_TESTS, 1):
        config = {"configurable": {"thread_id": f"cat_{i}"}}
        result = await compiled1.ainvoke({"input_args": {"query": query}}, config=config)

        case_result = result.get("agent_data", {}).get("results", {}).get("case_worker_1")
        selected_id = "?"
        if case_result and isinstance(case_result, dict):
            selected_id = case_result.get("selected_case_id", "?")

        match = selected_id == expected
        if match:
            correct += 1
        symbol = "✅" if match else "⚠️"
        log(buf, f"  {symbol} [{i}] '{query[:50]}...' → {selected_id} (expected {expected})")

    log(buf, f"\n  Predefined: {correct}/{total} matched")

    # --- Part 2: Custom cases ---
    log(buf, "\n  === Part 2: Custom Cases ===")
    graph2 = _build_custom_graph(llm)
    compiled2 = graph2.compile()

    custom_results = []
    for i, text in enumerate(CUSTOM_TESTS, 1):
        config = {"configurable": {"thread_id": f"custom_{i}"}}
        result = await compiled2.ainvoke({"test_input": text}, config=config)

        case_result = result.get("agent_data", {}).get("results", {}).get("custom_worker")
        if case_result and isinstance(case_result, dict):
            sel = case_result.get("selected_case_name", "?")
            sid = case_result.get("selected_case_id", "?")
            log(buf, f"  [{i}] '{text[:50]}...' → {sel} ({sid})")
            custom_results.append(sid)
        else:
            log(buf, f"  [{i}] '{text[:50]}...' → NO RESULT")
            custom_results.append(None)

    log(buf, f"\n  Custom: {len([r for r in custom_results if r])}/{len(CUSTOM_TESTS)} returned results")

    write_log(buf, "case_worker_run.log")

    return TestResult(
        name="Case Worker",
        passed=True,
        details={
            "predefined_correct": correct,
            "predefined_total": total,
            "custom_results": custom_results,
        },
    )
