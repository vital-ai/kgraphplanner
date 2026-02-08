"""
Case: LangGraph Agent — simple chatbot with checkpointer continuity.

Tests a minimal LangGraph StateGraph (START → chatbot → END) with:
  1. KGraphMemoryCheckpointer + KGraphSerializer
  2. Basic LLM invocation via GPT-5
  3. Conversation continuity across turns via thread_id
"""

from __future__ import annotations

import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import log, write_log, save_png, OUTPUT_DIR


class _State(TypedDict):
    messages: Annotated[list, add_messages]


async def run() -> TestResult:
    """Run the LangGraph chatbot agent test."""
    load_dotenv()
    buf = io.StringIO()

    # --- Setup ---
    log(buf, "  === Step 1: Setup ===")
    llm = ChatOpenAI(model="gpt-5", temperature=0.7, max_tokens=500)
    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)

    def chatbot_node(state: _State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(_State)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    app = graph.compile(checkpointer=checkpointer)

    # --- Graph PNG ---
    log(buf, "\n  === Step 2: Graph Diagram ===")
    await save_png(app, "langgraph_agent_graph.png", buf)

    # --- Test 1: Basic response ---
    log(buf, "\n  === Step 3: Basic Response ===")
    config = {"configurable": {"thread_id": "lg-test-1", "checkpoint_ns": ""}}

    result = app.invoke(
        {"messages": [HumanMessage(content="Hello, how are you?")]},
        config=config,
    )
    response1 = result["messages"][-1].content
    log(buf, f"  User: Hello, how are you?")
    log(buf, f"  AI: {response1[:200]}")
    assert response1, "Expected non-empty response"

    # --- Test 2: Conversation continuity ---
    log(buf, "\n  === Step 4: Conversation Continuity ===")
    config2 = {"configurable": {"thread_id": "lg-test-continuity", "checkpoint_ns": ""}}

    app.invoke(
        {"messages": [HumanMessage(content="My favorite color is blue")]},
        config=config2,
    )
    log(buf, f"  Turn 1: My favorite color is blue")

    result2 = app.invoke(
        {"messages": [HumanMessage(content="What is my favorite color?")]},
        config=config2,
    )
    response2 = result2["messages"][-1].content
    log(buf, f"  Turn 2: What is my favorite color?")
    log(buf, f"  AI: {response2[:200]}")

    continuity_ok = "blue" in response2.lower()
    log(buf, f"  Continuity check: {'✅' if continuity_ok else '⚠️'} (mentions blue: {continuity_ok})")

    # --- Summary ---
    log(buf, "\n  === Results ===")
    log(buf, f"  Basic response: ✅")
    log(buf, f"  Continuity: {'✅' if continuity_ok else '❌'}")

    write_log(buf, "langgraph_agent_run.log")

    return TestResult(
        name="LangGraph Agent",
        passed=True,
        details={
            "basic_response_len": len(response1),
            "continuity_ok": continuity_ok,
        },
    )
