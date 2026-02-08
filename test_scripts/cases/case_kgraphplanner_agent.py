"""
Case: KGraphPlannerAgent — classification-based routing with async event tracking.

Tests the KGraphPlannerAgent's ability to:
  1. Classify user input into categories (greeting, help_request, question, etc.)
  2. Route and generate appropriate responses per category
  3. Emit detailed async events (node_start/end, llm_call, classification, etc.)
  4. Maintain conversation continuity via checkpointer

Uses gpt-5.2 for response generation and gpt-4o-mini for classification.
"""

from __future__ import annotations

import io
import os
import sys
import asyncio
import logging
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from kgraphplanner.agent.kgraphplanner_agent import KGraphPlannerAgent
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import (
    log, write_log, save_png, execute_with_logging, OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

# --- Test requests ---

TEST_CASES = [
    {
        "message": "Hello! Good morning!",
        "expected_category": "greeting",
        "description": "Simple greeting",
    },
    {
        "message": "I need help with organizing my tasks",
        "expected_category": "help_request",
        "description": "Help request",
    },
    {
        "message": "What is the difference between Python and JavaScript?",
        "expected_category": "question",
        "description": "Information question",
    },
    {
        "message": "Can you create a weekly meal plan for me?",
        "expected_category": "creation_request",
        "description": "Creation request",
    },
    {
        "message": "Help me plan a project timeline for launching a mobile app",
        "expected_category": "planning_request",
        "description": "Planning request",
    },
    {
        "message": "I'm feeling overwhelmed with work lately",
        "expected_category": "general",
        "description": "General conversation",
    },
]


# --- Event consumer ---

async def _event_consumer(event_queue: asyncio.Queue, buf: io.StringIO, label: str) -> int:
    """Consume async events from the agent, logging details to buf. Returns event count."""
    event_count = 0
    node_stack = []

    while True:
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=2.0)
            event_count += 1

            timestamp = datetime.fromtimestamp(event['timestamp']).strftime('%H:%M:%S.%f')[:-3]
            event_type = event['type']
            data = event['data']

            if event_type == "node_start":
                node_name = data.get('node', 'unknown')
                node_stack.append(node_name)
                log(buf, f"    🚀 [{timestamp}] NODE_START: {node_name} (depth: {len(node_stack)})")
            elif event_type == "node_end":
                node_name = data.get('node', 'unknown')
                if node_stack and node_stack[-1] == node_name:
                    node_stack.pop()
                log(buf, f"    ✅ [{timestamp}] NODE_END: {node_name} (depth: {len(node_stack)})")
            elif event_type == "llm_call_start":
                purpose = data.get('purpose', data.get('category', 'unknown'))
                log(buf, f"    🧠 [{timestamp}] LLM_CALL_START: {purpose}")
            elif event_type == "llm_call_end":
                category = data.get('category')
                response_len = data.get('response_length', '?')
                log(buf, f"    🧠 [{timestamp}] LLM_CALL_END: category={category}, response_len={response_len}")
            elif event_type == "classification_complete":
                category = data.get('category')
                msg_preview = (data.get('message', '')[:50] + '...') if len(data.get('message', '')) > 50 else data.get('message', '')
                log(buf, f"    🏷️  [{timestamp}] CLASSIFICATION: '{msg_preview}' → {category}")
            elif event_type == "response_generated":
                category = data.get('category')
                response_len = data.get('response_length')
                log(buf, f"    💬 [{timestamp}] RESPONSE_GENERATED: category={category}, length={response_len}")
            elif event_type == "agent_start":
                msg_count = data.get('message_count')
                log(buf, f"    🤖 [{timestamp}] AGENT_START: processing {msg_count} messages")
            elif event_type == "agent_complete":
                final_step = data.get('final_step')
                log(buf, f"    🎉 [{timestamp}] AGENT_COMPLETE: final_step={final_step}")
            elif event_type == "agent_error":
                error = data.get('error')
                log(buf, f"    ❌ [{timestamp}] AGENT_ERROR: {error}")
            else:
                log(buf, f"    📋 [{timestamp}] {event_type.upper()}: {data}")

            event_queue.task_done()

        except asyncio.TimeoutError:
            break
        except Exception as e:
            log(buf, f"    ⚠️ Event consumer error: {e}")
            break

    log(buf, f"    Events for {label}: {event_count}")
    return event_count


# --- Main entry point ---

async def run() -> TestResult:
    """Run the KGraphPlannerAgent classification + continuity test."""
    load_dotenv()
    buf = io.StringIO()

    # --- Setup ---
    log(buf, "  === Step 1: Setup ===")
    model = ChatOpenAI(model="gpt-5.2", max_tokens=500)
    classification_model = ChatOpenAI(model="gpt-4o-mini", max_tokens=50)
    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)
    event_queue = asyncio.Queue()

    agent = KGraphPlannerAgent(
        name="kgraphplanner_agent",
        model=model,
        event_queue=event_queue,
        checkpointer=checkpointer,
        classification_model=classification_model,
    )

    # --- Graph PNG ---
    log(buf, "\n  === Step 2: Graph Diagram ===")
    compiled = agent.get_compiled_graph()
    await save_png(compiled, "kgraphplanner_agent_graph.png", buf)

    # --- Classification tests ---
    log(buf, f"\n  === Step 3: Execute {len(TEST_CASES)} Classification Tests ===")
    sub_results = []
    total_events = 0

    for i, tc in enumerate(TEST_CASES, 1):
        message_content = tc["message"]
        expected = tc["expected_category"]
        desc = tc["description"]
        log(buf, f"\n  --- Test {i}: {desc} ---")
        log(buf, f"  Message: '{message_content}'")
        log(buf, f"  Expected category: {expected}")

        consumer_task = asyncio.create_task(
            _event_consumer(event_queue, buf, f"Test {i}")
        )

        try:
            config = {"configurable": {"thread_id": f"kgp-test-{i}"}}
            messages = [HumanMessage(content=message_content)]
            result = await agent.arun(messages, config=config)

            actual_category = result.get("agent_data", {}).get("request_category", "unknown")
            final_messages = result.get("messages", [])
            response_preview = ""
            if final_messages:
                content = final_messages[-1].content
                response_preview = content[:200]
                log(buf, f"  Response: {response_preview}")

            match = actual_category == expected
            log(buf, f"  Classification: {'✅' if match else '⚠️'} expected={expected}, actual={actual_category}")
            log(buf, f"  Processing step: {result.get('agent_data', {}).get('processing_step')}")

            sub_results.append({
                "test": i, "desc": desc,
                "expected": expected, "actual": actual_category,
                "match": match, "ok": True,
            })

        except Exception as e:
            log(buf, f"  ❌ Error: {e}")
            sub_results.append({
                "test": i, "desc": desc,
                "expected": expected, "actual": None,
                "match": False, "ok": False, "error": str(e),
            })

        event_count = await consumer_task
        total_events += event_count
        await asyncio.sleep(0.3)

    # --- Conversation continuity test ---
    log(buf, "\n  === Step 4: Conversation Continuity ===")
    continuity_ok = False

    consumer_task = asyncio.create_task(
        _event_consumer(event_queue, buf, "Continuity")
    )

    config = {"configurable": {"thread_id": "kgp-test-conversation"}}
    try:
        result1 = await agent.arun(
            [HumanMessage(content="My name is Alice and I like hiking")], config=config
        )
        r1_content = result1['messages'][-1].content
        log(buf, f"  Response 1: {r1_content[:200]}")

        result2 = await agent.arun(
            [HumanMessage(content="What's my name and what do I like?")], config=config
        )
        r2_content = result2['messages'][-1].content
        log(buf, f"  Response 2: {r2_content[:200]}")

        # Basic check: response 2 should mention "Alice" or "hiking"
        r2_lower = r2_content.lower()
        continuity_ok = "alice" in r2_lower or "hiking" in r2_lower
        log(buf, f"  Continuity check: {'✅' if continuity_ok else '⚠️'} (mentions Alice/hiking: {continuity_ok})")

    except Exception as e:
        log(buf, f"  ❌ Continuity test error: {e}")

    event_count = await consumer_task
    total_events += event_count

    # --- Summary ---
    passed_count = sum(1 for r in sub_results if r["ok"])
    match_count = sum(1 for r in sub_results if r["match"])
    total_count = len(sub_results)

    log(buf, f"\n  === Results ===")
    log(buf, f"  Classification tests: {passed_count}/{total_count} succeeded, {match_count}/{total_count} matched expected category")
    for r in sub_results:
        status = "✅" if r["ok"] else "❌"
        cat_match = "=" if r["match"] else "≠"
        log(buf, f"    {status} Test {r['test']}: {r['desc']} (expected={r['expected']} {cat_match} actual={r['actual']})")
    log(buf, f"  Continuity: {'✅' if continuity_ok else '❌'}")
    log(buf, f"  Total events: {total_events}")

    # --- Write log ---
    write_log(buf, "kgraphplanner_agent_run.log")

    # --- Assertions ---
    assert passed_count == total_count, \
        f"Expected all {total_count} tests to succeed, but {total_count - passed_count} failed"

    return TestResult(
        name="KGraphPlanner Agent",
        passed=True,
        details={
            "tests_total": total_count,
            "tests_passed": passed_count,
            "category_matches": match_count,
            "continuity_ok": continuity_ok,
            "total_events": total_events,
            "output_dir": OUTPUT_DIR,
        },
    )
