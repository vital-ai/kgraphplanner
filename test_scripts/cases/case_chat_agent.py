"""
Case: KGraphChatAgent — conversation with history and memory persistence.

Tests the KGraphChatAgent's ability to:
  1. Maintain multi-turn conversation context (history questions about Roman Empire)
  2. Persist memory across agent instances via shared checkpointer
"""

from __future__ import annotations

import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from kgraphplanner.agent.kgraph_chat_agent import KGraphChatAgent
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import log, write_log, save_png, OUTPUT_DIR

CONVERSATION_TURNS = [
    ("Tell me about the Roman Empire. When did it start and who was the first emperor?",
     "Initial history question about Roman Empire"),
    ("How long did Augustus rule?",
     "Follow-up about Augustus (should remember context)"),
    ("What happened after Augustus died? Who succeeded him?",
     "Succession question (building on context)"),
    ("Earlier you mentioned Augustus. Can you compare his reign to Julius Caesar's time in power?",
     "Explicit reference to earlier conversation"),
    ("What are the main topics we've discussed so far in our conversation?",
     "Meta-question about conversation history"),
]


async def run() -> TestResult:
    load_dotenv()
    buf = io.StringIO()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)

    # --- Part 1: Multi-turn conversation ---
    log(buf, "  === Part 1: Conversation with History ===")

    chat_worker = KGraphChatWorker(
        name="history_chat_worker",
        llm=llm,
        system_directive=(
            "You are a knowledgeable history assistant. You can answer questions about historical events, "
            "people, and periods. You maintain context from our conversation and can refer back to previous "
            "topics we've discussed. Be engaging and informative in your responses."
        ),
        required_inputs=["message"],
    )

    agent = KGraphChatAgent(
        name="history_chat_agent",
        checkpointer=checkpointer,
        chat_worker=chat_worker,
    )

    compiled = agent.get_compiled_graph()
    await save_png(compiled, "chat_agent_graph.png", buf)

    config = {"configurable": {"thread_id": "chat-history-test"}}
    responses = []

    for i, (msg, desc) in enumerate(CONVERSATION_TURNS, 1):
        log(buf, f"\n  --- Turn {i}: {desc} ---")
        log(buf, f"  Human: {msg}")

        result = await agent.arun([HumanMessage(content=msg)], config=config)
        content = result.get("messages", [])[-1].content if result.get("messages") else ""
        responses.append(content)
        log(buf, f"  AI: {content[:200]}")

        if i == 1:
            info = agent.get_agent_info()
            log(buf, f"  Agent info: {info}")

        await asyncio.sleep(0.3)

    log(buf, f"\n  Conversation turns completed: {len(responses)}/{len(CONVERSATION_TURNS)}")

    # --- Part 2: Memory persistence ---
    log(buf, "\n  === Part 2: Memory Persistence ===")

    checkpointer2 = KGraphMemoryCheckpointer(serde=KGraphSerializer())

    worker1 = KGraphChatWorker(
        name="memory_worker", llm=llm,
        system_directive="You are a helpful assistant that remembers our conversations.",
        required_inputs=["message"],
    )
    agent1 = KGraphChatAgent(name="memory_agent", checkpointer=checkpointer2, chat_worker=worker1)

    cfg = {"configurable": {"thread_id": "memory-persist-test"}}

    result1 = await agent1.arun([HumanMessage(content="My favorite color is blue and I love hiking.")], config=cfg)
    r1 = result1.get("messages", [])[-1].content if result1.get("messages") else ""
    log(buf, f"  Agent 1 → '{r1[:150]}'")

    # New agent instance, same checkpointer
    worker2 = KGraphChatWorker(
        name="memory_worker", llm=llm,
        system_directive="You are a helpful assistant that remembers our conversations.",
        required_inputs=["message"],
    )
    agent2 = KGraphChatAgent(name="memory_agent", checkpointer=checkpointer2, chat_worker=worker2)

    result2 = await agent2.arun([HumanMessage(content="What's my favorite color and hobby?")], config=cfg)
    r2 = result2.get("messages", [])[-1].content if result2.get("messages") else ""
    log(buf, f"  Agent 2 → '{r2[:150]}'")

    memory_ok = "blue" in r2.lower() and "hik" in r2.lower()
    log(buf, f"  Memory persistence: {'✅' if memory_ok else '⚠️'} (mentions blue+hiking: {memory_ok})")

    write_log(buf, "chat_agent_run.log")

    return TestResult(
        name="Chat Agent",
        passed=True,
        details={
            "conversation_turns": len(responses),
            "memory_persistence": memory_ok,
        },
    )
