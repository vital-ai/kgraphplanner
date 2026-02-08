"""
Case: Multi-location weather — parallel tool calls via KGraphToolAgent.

Tests the KGraphToolAgent's ability to:
  1. Accept a multi-city weather query
  2. Make parallel weather tool calls
  3. Return a combined response

Uses KGraphToolWorker with weather_tool and Keycloak JWT auth.
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

from kgraphplanner.agent.kgraph_tool_agent import KGraphToolAgent
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import (
    create_tool_manager, check_tools_available,
    log, write_log, save_png, execute_with_logging, OUTPUT_DIR,
)


async def run() -> TestResult:
    """Run the multi-location weather test."""
    load_dotenv()
    buf = io.StringIO()

    # --- Setup ---
    log(buf, "  === Step 1: Setup ===")
    tm = create_tool_manager()
    if not check_tools_available(tm):
        return TestResult(
            name="Multi-Weather",
            passed=True,
            details={"skipped": True, "reason": "No tools available (tool server not running?)"},
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)

    tool_worker = KGraphToolWorker(
        name="multi_weather_worker",
        llm=llm,
        system_directive=(
            "You are a weather assistant that can check weather conditions in multiple "
            "locations simultaneously. When asked about weather in multiple cities, make "
            "separate weather tool calls for each location using their coordinates."
        ),
        required_inputs=["request"],
        tool_manager=tm,
        available_tool_ids=[ToolNameEnum.weather_tool.value],
    )

    agent = KGraphToolAgent(
        name="multi_weather_agent",
        checkpointer=checkpointer,
        tool_worker=tool_worker,
        tool_manager=tm,
        tool_names=[ToolNameEnum.weather_tool.value],
    )

    # --- Graph PNG ---
    log(buf, "\n  === Step 2: Graph Diagram ===")
    compiled = agent.get_compiled_graph()
    await save_png(compiled, "multi_weather_graph.png", buf)

    # --- Execute ---
    query = "What's the weather like in New York City, London, and Tokyo right now?"
    log(buf, f"\n  === Step 3: Execute ===")
    log(buf, f"  Query: {query}")

    try:
        config = {"configurable": {"thread_id": "multi-weather-test"}, "recursion_limit": 15}
        result = await execute_with_logging(
            agent.arun(messages=[HumanMessage(content=query)], config=config), buf
        )

        response_messages = result.get("messages", [])
        if response_messages:
            content = response_messages[-1].content
            log(buf, f"\n  Response (len={len(content)}):")
            log(buf, f"  {content[:500]}")

            # Basic check: should mention at least 2 of the 3 cities
            content_lower = content.lower()
            cities_mentioned = sum(1 for city in ["new york", "london", "tokyo"] if city in content_lower)
            log(buf, f"\n  Cities mentioned in response: {cities_mentioned}/3")
        else:
            content = ""
            cities_mentioned = 0
            log(buf, "  [No response generated]")

    except Exception as e:
        log(buf, f"  ❌ Error: {e}")
        write_log(buf, "multi_weather_run.log")
        raise

    # --- Agent info ---
    agent_info = agent.get_agent_info()
    log(buf, f"\n  Agent info: {agent_info}")

    # --- Write log ---
    write_log(buf, "multi_weather_run.log")

    # --- Assertions ---
    assert content, "Expected non-empty response"
    assert cities_mentioned >= 2, f"Expected at least 2 cities mentioned, got {cities_mentioned}"

    return TestResult(
        name="Multi-Weather",
        passed=True,
        details={
            "query": query,
            "response_len": len(content),
            "cities_mentioned": cities_mentioned,
        },
    )
