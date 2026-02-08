"""
Case: Tool Agent — Multi-Tool.

Tests KGraphToolAgent with all tools (web search, address validation,
weather, place search) handling a complex request.
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

ALL_TOOLS = [
    ToolNameEnum.google_web_search_tool.value,
    ToolNameEnum.google_address_validation_tool.value,
    ToolNameEnum.weather_tool.value,
    ToolNameEnum.place_search_tool.value,
]


async def run() -> TestResult:
    load_dotenv()
    buf = io.StringIO()

    log(buf, "  === Setup ===")
    tm = create_tool_manager()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)

    tool_worker = KGraphToolWorker(
        name="multi_tool_worker",
        llm=llm,
        system_directive=(
            "You are a versatile assistant with access to web search, address validation, "
            "weather, and place search tools. Choose the appropriate tool based on the user's request."
        ),
        required_inputs=["request"],
        tool_manager=tm,
        available_tool_ids=ALL_TOOLS,
    )

    agent = KGraphToolAgent(
        name="multi_tool_agent",
        checkpointer=checkpointer,
        tool_worker=tool_worker,
        tool_manager=tm,
        tool_names=ALL_TOOLS,
    )

    compiled = agent.get_compiled_graph()
    await save_png(compiled, "tool_agent_multi_graph.png", buf)

    query = (
        "I'm planning a trip to New York City. Can you help me find the weather forecast "
        "and some good restaurants near Times Square?"
    )
    log(buf, f"\n  === Execute ===")
    log(buf, f"  Query: {query}")

    config = {"configurable": {"thread_id": "ta-multi-tool"}, "recursion_limit": 10}
    result = await agent.arun(messages=[HumanMessage(content=query)], config=config)

    content = result["messages"][-1].content
    log(buf, f"  Response (len={len(content)}):")
    log(buf, f"  {content[:500]}")

    agent_info = agent.get_agent_info()
    log(buf, f"\n  Agent info: {agent_info}")

    write_log(buf, "tool_agent_multi_run.log")

    assert content, "Expected non-empty response"

    return TestResult(
        name="Tool Agent: Multi-Tool",
        passed=True,
        details={"query": query[:60] + "...", "response_len": len(content)},
    )
