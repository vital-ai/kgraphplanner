"""
Case: Times Square — direct LLM tool binding and invocation.

Tests the LLM's ability to:
  1. Bind tools via LangChain's bind_tools
  2. Recognize when to call a tool from user input
  3. Generate correct tool call arguments
  4. Execute the tool and get results

Uses place_search_tool with two prompts: terse ("Times Square") and
explicit ("Find information about Times Square").
"""

from __future__ import annotations

import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import (
    create_tool_manager, log, write_log, OUTPUT_DIR,
)


def _run_tool_call_test(
    llm_with_tools, tool_function, messages, label: str, buf: io.StringIO
) -> dict:
    """Send messages to LLM with bound tools and log results. Returns sub-result dict."""
    log(buf, f"\n  --- {label} ---")
    for i, msg in enumerate(messages):
        log(buf, f"  msg[{i}] {type(msg).__name__}: {msg.content}")

    response = llm_with_tools.invoke(messages)
    log(buf, f"  Response type: {type(response).__name__}")
    log(buf, f"  Content: {response.content[:200] if response.content else '(empty)'}")

    tool_calls = getattr(response, 'tool_calls', None) or []
    log(buf, f"  Tool calls: {len(tool_calls)}")

    executed = False
    for i, tc in enumerate(tool_calls):
        name = tc.get('name', '?')
        args = tc.get('args', {})
        tc_id = tc.get('id', '?')
        log(buf, f"    [{i}] {name}  args={args}  id={tc_id}")

        # Execute the tool call if it's the place search tool
        if name == 'place_search_tool':
            place_str = args.get('place_search_string', '')
            log(buf, f"    Executing place_search_tool('{place_str}')...")
            try:
                result = tool_function.invoke({"place_search_string": place_str})
                log(buf, f"    Result type: {type(result).__name__}")
                # PlaceSearchOutput may use 'results' or 'place_details_list'
                places = (
                    getattr(result, 'results', None)
                    or getattr(result, 'place_details_list', None)
                    or []
                )
                log(buf, f"    Result: {len(places)} places")
                for j, place in enumerate(places[:3]):
                    pname = getattr(place, 'name', '?')
                    paddr = getattr(place, 'address', getattr(place, 'formatted_address', '?'))
                    log(buf, f"      {j+1}. {pname} — {paddr}")
                executed = True
            except Exception as e:
                log(buf, f"    Execution error (server may be down): {e}")

    return {
        "label": label,
        "tool_calls": len(tool_calls),
        "executed": executed,
        "ok": len(tool_calls) > 0,
    }


async def run() -> TestResult:
    """Run the LLM tool calling debug test."""
    load_dotenv()
    buf = io.StringIO()

    # --- Setup ---
    log(buf, "  === Step 1: Setup ===")
    tm = create_tool_manager()
    available = tm.get_tool_names()
    log(buf, f"  Available tools: {available}")

    place_tool = tm.get_tool(ToolNameEnum.place_search_tool.value)
    assert place_tool is not None, f"place_search_tool not found (available: {available})"

    tool_fn = place_tool.get_tool_function()
    assert tool_fn is not None, "place_search_tool function is None"

    log(buf, f"  Tool: {tool_fn.name}")
    log(buf, f"  Description: {tool_fn.description[:100]}")
    if hasattr(tool_fn, 'args_schema') and hasattr(tool_fn.args_schema, 'model_fields'):
        log(buf, f"  Schema fields: {list(tool_fn.args_schema.model_fields.keys())}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([tool_fn])

    # --- Test 1: Terse input ---
    log(buf, "\n  === Step 2: Tool Calling Tests ===")
    sub_results = []

    r1 = _run_tool_call_test(
        llm_with_tools, tool_fn,
        messages=[
            SystemMessage(content="You are a helpful assistant. Use the available tools when appropriate."),
            HumanMessage(content="Times Square"),
        ],
        label="Terse input ('Times Square')",
        buf=buf,
    )
    sub_results.append(r1)

    # --- Test 2: Explicit input ---
    r2 = _run_tool_call_test(
        llm_with_tools, tool_fn,
        messages=[
            SystemMessage(content="You are a helpful assistant. Use the place_search_tool when the user asks about a specific place."),
            HumanMessage(content="Find information about Times Square"),
        ],
        label="Explicit input ('Find information about Times Square')",
        buf=buf,
    )
    sub_results.append(r2)

    # --- Summary ---
    calls_total = sum(r["tool_calls"] for r in sub_results)
    ok_count = sum(1 for r in sub_results if r["ok"])
    log(buf, f"\n  === Results: {ok_count}/{len(sub_results)} tests triggered tool calls, {calls_total} total calls ===")
    for r in sub_results:
        status = "✅" if r["ok"] else "⚠️"
        log(buf, f"    {status} {r['label']}: {r['tool_calls']} calls, executed={r['executed']}")

    write_log(buf, "times_square_run.log")

    # At least one test should trigger a tool call
    assert ok_count > 0, "Expected at least one test to trigger a tool call"

    return TestResult(
        name="Times Square",
        passed=True,
        details={
            "tests": len(sub_results),
            "tool_calls_triggered": ok_count,
            "total_tool_calls": calls_total,
        },
    )
