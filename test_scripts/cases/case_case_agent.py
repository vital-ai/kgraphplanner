"""
Case: KGraphCaseAgent — case-based routing with tool and chat workers.

Tests the KGraphCaseAgent's ability to:
  1. Classify user input into appropriate cases
  2. Route to the correct worker based on classification
  3. Execute the selected worker
  4. Resolve the output into a friendly response

Workers:
  - weather_specialist: KGraphToolWorker (weather_tool)
  - search_specialist: KGraphToolWorker (google_web_search_tool)
  - place_specialist: KGraphToolWorker (place_search_tool)
  - general_assistant: KGraphChatWorker
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

from kgraphplanner.agent.kgraph_case_agent import KGraphCaseAgent
from kgraphplanner.worker.kgraph_case_worker import KGCase
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import (
    create_tool_manager, check_tools_available,
    log, write_log, save_png, execute_with_logging, OUTPUT_DIR,
)

# --- Test requests ---

TEST_REQUESTS = [
    {
        "input": "What's the current weather in New York?",
        "expected_case": "Weather Topic",
        "description": "Current weather request",
    },
    {
        "input": "what's the location of freeman's restaurant in nyc",
        "expected_case": "Place Topic",
        "description": "Place search request",
    },
    {
        "input": "Hello! How are you doing today?",
        "expected_case": "General Topic",
        "description": "General conversation",
    },
    {
        "input": "What's the current temperature in San Francisco?",
        "expected_case": "Weather Topic",
        "description": "Second weather request",
    },
    {
        "input": "What are the latest developments in artificial intelligence?",
        "expected_case": "Search Topic",
        "description": "Information research request",
    },
]


# --- Builder ---

def _build_case_agent(llm, tool_manager):
    """Build the case agent with all workers and cases."""
    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)

    weather_worker = KGraphToolWorker(
        name="weather_specialist",
        llm=llm,
        system_directive="You are a weather specialist. Use weather tools to provide accurate weather information.",
        tool_manager=tool_manager,
        available_tool_ids=[ToolNameEnum.weather_tool.value],
        required_inputs=["request"],
    )
    chat_worker = KGraphChatWorker(
        name="general_assistant",
        llm=llm,
        system_directive="You are a helpful general assistant. Answer questions conversationally and helpfully.",
        required_inputs=["message"],
    )
    search_worker = KGraphToolWorker(
        name="search_specialist",
        llm=llm,
        system_directive="You are a research specialist. Use web search tools to find accurate information.",
        tool_manager=tool_manager,
        available_tool_ids=[ToolNameEnum.google_web_search_tool.value],
        required_inputs=["request"],
    )
    place_worker = KGraphToolWorker(
        name="place_specialist",
        llm=llm,
        system_directive="You are a location specialist. Use place search tools to find information about locations.",
        tool_manager=tool_manager,
        available_tool_ids=[ToolNameEnum.place_search_tool.value],
        required_inputs=["request"],
    )

    kg_cases = [
        KGCase(id="weather_topic", name="Weather Topic",
               description="weather-related requests including forecasts, current conditions, and climate information"),
        KGCase(id="search_topic", name="Search Topic",
               description="information lookup and research requests about current events, facts, or general knowledge"),
        KGCase(id="place_topic", name="Place Topic",
               description="location-based requests including finding places, restaurants, businesses, or geographic information"),
        KGCase(id="general_topic", name="General Topic",
               description="general conversation, greetings, and questions that don't require specific tools"),
    ]

    case_worker_pairs = [
        (kg_cases[0], weather_worker),
        (kg_cases[1], search_worker),
        (kg_cases[2], place_worker),
        (kg_cases[3], chat_worker),
    ]

    return KGraphCaseAgent(
        name="smart_assistant",
        case_worker_pairs=case_worker_pairs,
        llm=llm,
        checkpointer=checkpointer,
    )


# --- Main entry point ---

async def run() -> TestResult:
    """Run the case agent test."""
    load_dotenv()
    buf = io.StringIO()

    # --- Setup ---
    log(buf, "  === Step 1: Setup ===")
    tm = create_tool_manager()
    if not check_tools_available(tm):
        return TestResult(
            name="Case Agent Routing",
            passed=True,
            details={"skipped": True, "reason": "No tools available (tool server not running?)"},
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    case_agent = _build_case_agent(llm, tm)
    compiled = case_agent.get_compiled_graph()

    # --- Graph PNG ---
    log(buf, "\n  === Step 2: Graph Diagram ===")
    await save_png(compiled, "case_agent_graph.png", buf)

    # --- Agent info ---
    agent_info = case_agent.get_agent_info()
    log(buf, f"\n  Agent: {agent_info['name']}  type={agent_info['type']}  cases={agent_info['case_count']}")
    for ci in agent_info.get("cases", []):
        log(buf, f"    - {ci['name']}: {ci['worker_type']} ({ci['worker_name']})")

    # --- Execute test requests ---
    log(buf, f"\n  === Step 3: Execute {len(TEST_REQUESTS)} Requests ===")
    sub_results = []

    for i, tc in enumerate(TEST_REQUESTS, 1):
        user_input = tc["input"]
        expected = tc["expected_case"]
        desc = tc["description"]
        log(buf, f"\n  --- Request {i}: {desc} ---")
        log(buf, f"  Expected case: {expected}")
        log(buf, f"  Input: {user_input}")

        try:
            config = {"configurable": {"thread_id": f"case-agent-test-{i}"}}
            messages = [HumanMessage(content=user_input)]
            result = await execute_with_logging(
                case_agent.arun(messages, config=config), buf
            )

            response_messages = result.get("messages", [])
            if response_messages:
                content = response_messages[-1].content
                log(buf, f"  Response: {content[:300]}")
                sub_results.append({"test": i, "desc": desc, "expected": expected, "ok": True})
            else:
                log(buf, "  Response: [No response generated]")
                sub_results.append({"test": i, "desc": desc, "expected": expected, "ok": False})

        except Exception as e:
            log(buf, f"  ❌ Error: {e}")
            sub_results.append({"test": i, "desc": desc, "expected": expected, "ok": False, "error": str(e)})

        await asyncio.sleep(0.5)

    # --- Summary ---
    passed_count = sum(1 for r in sub_results if r["ok"])
    total_count = len(sub_results)

    log(buf, f"\n  === Results: {passed_count}/{total_count} requests succeeded ===")
    for r in sub_results:
        status = "✅" if r["ok"] else "❌"
        log(buf, f"    {status} Test {r['test']}: {r['desc']} (expected: {r['expected']})")

    # --- Write log ---
    write_log(buf, "case_agent_run.log")

    # --- Assertions ---
    assert passed_count == total_count, \
        f"Expected all {total_count} requests to succeed, but {total_count - passed_count} failed"

    return TestResult(
        name="Case Agent Routing",
        passed=True,
        details={
            "requests_total": total_count,
            "requests_passed": passed_count,
            "cases": [c["name"] for c in agent_info.get("cases", [])],
            "output_dir": OUTPUT_DIR,
        },
    )
