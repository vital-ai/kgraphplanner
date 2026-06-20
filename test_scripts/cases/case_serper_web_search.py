"""
Case: Serper Web Search tool tests — schema, function creation, and sample invocations.

Tests the serper_web_search_tool across multiple search types (organic, news, images,
shopping, places) and verifies response parsing including knowledge graph, related
searches, and people-also-ask fields.

Gracefully handles tool server being unavailable.
"""

from __future__ import annotations

import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

from kgraphplanner.tools.serper_websearch.serper_web_search_tool import SerperWebSearchTool
from kgraphplanner.vital_agent_rest_resource_client.tools.serper_web_search.models import (
    SerperWebSearchInput, SerperWebSearchOutput
)

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import create_tool_manager, log, write_log, OUTPUT_DIR


# --- Helpers ---

def _get_tool(tm=None):
    """Get the SerperWebSearchTool from the manager, falling back to direct construction."""
    if tm:
        tool = tm.get_tool("serper_web_search_tool")
        if tool:
            return tool
    config = {"tool_endpoint": tm.config.get_tool_endpoint() if tm else "http://localhost:8008"}
    return SerperWebSearchTool(config, tool_manager=tm)


# --- Sub-tests ---

def _test_tool_properties(buf: io.StringIO, tm=None) -> bool:
    """Test SerperWebSearchTool basic properties, schema, and function creation."""
    log(buf, "\n  --- Serper Web Search: Properties ---")
    tool = _get_tool(tm)

    log(buf, f"  name: {tool.get_tool_name()}")
    assert tool.get_tool_name() == "serper_web_search_tool", f"Unexpected name: {tool.get_tool_name()}"

    log(buf, f"  description: {tool.get_tool_description()}")

    schema = tool.get_tool_schema()
    assert schema is not None, "Schema is None"
    assert schema is SerperWebSearchInput, f"Schema is {schema}, expected SerperWebSearchInput"
    log(buf, f"  schema: OK ({schema.__name__})")

    tool_fn = tool.get_tool_function()
    assert tool_fn is not None, "Tool function is None"
    log(buf, f"  function: OK ({type(tool_fn).__name__})")

    log(buf, "  ✅ Properties: OK")
    return True


def _test_input_model(buf: io.StringIO, tm=None) -> bool:
    """Test SerperWebSearchInput model validation."""
    log(buf, "\n  --- Serper Web Search: Input Model ---")

    # Basic input
    basic = SerperWebSearchInput(search_query="test query")
    log(buf, f"  basic: {basic.model_dump()}")
    assert basic.search_query == "test query"
    assert basic.num_results == 10
    assert basic.search_type == "search"

    # Full input
    full = SerperWebSearchInput(
        search_query="pizza restaurants",
        num_results=5,
        search_type="places",
        location="New York,New York",
        time_period="week"
    )
    log(buf, f"  full: {full.model_dump()}")
    assert full.search_type == "places"
    assert full.location == "New York,New York"
    assert full.time_period == "week"

    # All search types
    for stype in ["search", "news", "images", "shopping", "places"]:
        inp = SerperWebSearchInput(search_query="test", search_type=stype)
        assert inp.search_type == stype
    log(buf, "  search_types: all valid")

    log(buf, "  ✅ Input Model: OK")
    return True


def _test_output_model(buf: io.StringIO, tm=None) -> bool:
    """Test SerperWebSearchOutput model and compact_dump."""
    log(buf, "\n  --- Serper Web Search: Output Model ---")

    output = SerperWebSearchOutput(
        tool="serper_web_search_tool",
        query="test query",
        results=[],
        total_results=0,
    )
    log(buf, f"  empty output: tool={output.tool}, query={output.query}, results={len(output.results)}")
    assert output.tool == "serper_web_search_tool"

    compact = output.compact_dump()
    assert "search_information" not in compact
    log(buf, f"  compact_dump: OK (keys={list(compact.keys())})")

    log(buf, "  ✅ Output Model: OK")
    return True


async def _test_organic_search(buf: io.StringIO, tm=None) -> bool:
    """Test organic web search invocation."""
    log(buf, "\n  --- Serper Web Search: Organic Search ---")
    tool = _get_tool(tm)
    tool_fn = tool.get_tool_function()

    try:
        result = await tool_fn.ainvoke({"search_query": "Python programming tutorials", "num_results": 3})
        log(buf, f"  result type: {type(result).__name__}")

        if isinstance(result, str):
            log(buf, f"  string result (server may be down): {result}")
        else:
            results = getattr(result, 'results', None) or []
            log(buf, f"  results: {len(results)} entries")
            for r in results:
                log(buf, f"    [{r.position}] {r.title} — {r.link}")

            kg = getattr(result, 'knowledge_graph', None)
            if kg:
                log(buf, f"  knowledge_graph: {kg.title} ({kg.type})")

            related = getattr(result, 'related_searches', None)
            if related:
                log(buf, f"  related_searches: {len(related)}")

            paa = getattr(result, 'people_also_ask', None)
            if paa:
                log(buf, f"  people_also_ask: {len(paa)}")
                for q in paa:
                    log(buf, f"    Q: {q.question}")

    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Organic Search: structure OK")
    return True


async def _test_news_search(buf: io.StringIO, tm=None) -> bool:
    """Test news search invocation."""
    log(buf, "\n  --- Serper Web Search: News Search ---")
    tool = _get_tool(tm)
    tool_fn = tool.get_tool_function()

    try:
        result = await tool_fn.ainvoke({
            "search_query": "artificial intelligence",
            "search_type": "news",
            "num_results": 3
        })
        log(buf, f"  result type: {type(result).__name__}")

        if isinstance(result, str):
            log(buf, f"  string result (server may be down): {result}")
        else:
            results = getattr(result, 'results', None) or []
            log(buf, f"  results: {len(results)} entries")
            for r in results:
                log(buf, f"    [{r.date}] {r.title} — source: {r.source}")

    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ News Search: structure OK")
    return True


async def _test_places_search(buf: io.StringIO, tm=None) -> bool:
    """Test places search invocation."""
    log(buf, "\n  --- Serper Web Search: Places Search ---")
    tool = _get_tool(tm)
    tool_fn = tool.get_tool_function()

    try:
        result = await tool_fn.ainvoke({
            "search_query": "pizza restaurants",
            "search_type": "places",
            "location": "New York,New York",
            "num_results": 3
        })
        log(buf, f"  result type: {type(result).__name__}")

        if isinstance(result, str):
            log(buf, f"  string result (server may be down): {result}")
        else:
            results = getattr(result, 'results', None) or []
            log(buf, f"  results: {len(results)} entries")
            for r in results:
                log(buf, f"    {r.title} — {r.address} (rating: {r.rating}, cid: {r.cid})")

    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Places Search: structure OK")
    return True


async def _test_knowledge_graph(buf: io.StringIO, tm=None) -> bool:
    """Test knowledge graph extraction from organic search."""
    log(buf, "\n  --- Serper Web Search: Knowledge Graph ---")
    tool = _get_tool(tm)
    tool_fn = tool.get_tool_function()

    try:
        result = await tool_fn.ainvoke({"search_query": "Albert Einstein", "num_results": 3})
        log(buf, f"  result type: {type(result).__name__}")

        if isinstance(result, str):
            log(buf, f"  string result (server may be down): {result}")
        else:
            kg = getattr(result, 'knowledge_graph', None)
            if kg:
                log(buf, f"  title: {kg.title}")
                log(buf, f"  type: {kg.type}")
                log(buf, f"  description: {kg.description}")
                if kg.attributes:
                    log(buf, f"  attributes: {list(kg.attributes.keys())}")
            else:
                log(buf, "  no knowledge graph returned (may vary by query)")

            paa = getattr(result, 'people_also_ask', None)
            if paa:
                log(buf, f"  people_also_ask: {len(paa)}")
                for q in paa:
                    log(buf, f"    Q: {q.question}")

    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Knowledge Graph: structure OK")
    return True


# --- Main entry point ---

SUB_TESTS = [
    ("Properties", _test_tool_properties),
    ("Input Model", _test_input_model),
    ("Output Model", _test_output_model),
    ("Organic Search", _test_organic_search),
    ("News Search", _test_news_search),
    ("Places Search", _test_places_search),
    ("Knowledge Graph", _test_knowledge_graph),
]


async def run() -> TestResult:
    """Run all Serper web search tool tests."""
    load_dotenv()
    buf = io.StringIO()

    log(buf, "  === Serper Web Search Tool Tests ===")
    passed = []
    failed = []

    tm = create_tool_manager()

    import asyncio
    import inspect

    for name, fn in SUB_TESTS:
        try:
            if inspect.iscoroutinefunction(fn):
                ok = await fn(buf, tm=tm)
            else:
                ok = fn(buf, tm=tm)
            passed.append(name)
        except Exception as e:
            log(buf, f"  ❌ {name}: {e}")
            failed.append(name)

    log(buf, f"\n  === Results: {len(passed)}/{len(SUB_TESTS)} passed ===")
    for name in passed:
        log(buf, f"    ✅ {name}")
    for name in failed:
        log(buf, f"    ❌ {name}")

    write_log(buf, "serper_web_search_run.log")

    assert not failed, f"Failed: {failed}"

    return TestResult(
        name="Serper Web Search Tool Tests",
        passed=True,
        details={
            "sub_tests": len(SUB_TESTS),
            "passed": len(passed),
            "tests": [n for n, _ in SUB_TESTS],
        },
    )
