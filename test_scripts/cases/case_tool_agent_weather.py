"""
Case: Tool Agent — Weather.

Tests KGraphToolAgent with weather_tool.
"""

from __future__ import annotations

import io
import os
import sys

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
    create_tool_manager, log, write_log, save_png, OUTPUT_DIR,
)


async def run() -> TestResult:
    load_dotenv()
    buf = io.StringIO()

    log(buf, "  === Setup ===")
    tm = create_tool_manager()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)

    tool_worker = KGraphToolWorker(
        name="weather_worker",
        llm=llm,
        system_directive="You are a weather assistant. Use the weather tool to get current weather information for locations.",
        required_inputs=["request"],
        tool_manager=tm,
        available_tool_ids=[ToolNameEnum.weather_tool.value],
    )

    agent = KGraphToolAgent(
        name="weather_agent",
        checkpointer=checkpointer,
        tool_worker=tool_worker,
        tool_manager=tm,
        tool_names=[ToolNameEnum.weather_tool.value],
    )

    compiled = agent.get_compiled_graph()
    await save_png(compiled, "tool_agent_weather_graph.png", buf)

    location = "San Francisco, CA"
    log(buf, f"\n  === Execute ===")
    log(buf, f"  Location: {location}")

    config = {"configurable": {"thread_id": "ta-weather"}, "recursion_limit": 10}
    result = await agent.arun(
        messages=[HumanMessage(content=f"What's the current weather in {location}?")],
        config=config,
    )

    content = result["messages"][-1].content
    log(buf, f"  Response (len={len(content)}):")
    log(buf, f"  {content[:300]}")

    write_log(buf, "tool_agent_weather_run.log")

    assert content, "Expected non-empty response"

    return TestResult(
        name="Tool Agent: Weather",
        passed=True,
        details={"location": location, "response_len": len(content)},
    )
