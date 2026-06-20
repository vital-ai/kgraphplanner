"""
Case: Direct tool tests using LOCAL models — address validation, place search, weather, web search.

Copy of case_tool_direct.py but imports models from kgraphplanner.vital_agent_rest_resource_client
(the local copy used by tool implementations) instead of vital_agent_kg_utils.

Tests each tool's properties, schema, function creation, and sample invocation.
Gracefully handles tool server being unavailable.
"""

from __future__ import annotations

import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

from kgraphplanner.tools.address_validation.address_validation_tool import AddressValidationTool
from kgraphplanner.tools.place_search.place_search_tool import PlaceSearchTool
from kgraphplanner.tools.weather.weather_tool import WeatherTool
from kgraphplanner.tools.websearch.web_search_tool import WebSearchTool
from kgraphplanner.vital_agent_rest_resource_client.tools.google_address_validation.models import AddressValidationInput
from kgraphplanner.vital_agent_rest_resource_client.tools.place_search.models import PlaceSearchInput
from kgraphplanner.vital_agent_rest_resource_client.tools.weather.models import WeatherInput
from kgraphplanner.vital_agent_rest_resource_client.tools.web_search.models import WebSearchInput

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import create_tool_manager, log, write_log, OUTPUT_DIR


# --- Sub-tests ---

async def _test_address_validation(buf: io.StringIO, tm=None) -> bool:
    """Test AddressValidationTool."""
    log(buf, "\n  --- Address Validation Tool ---")
    tool = tm.get_tool("google_address_validation_tool") if tm else AddressValidationTool({"tool_endpoint": "http://localhost:8008"})

    log(buf, f"  name: {tool.get_tool_name()}")
    log(buf, f"  description: {tool.get_tool_description()}")

    schema = tool.get_tool_schema()
    assert schema is not None, "Schema is None"
    log(buf, f"  schema: OK")

    tool_fn = tool.get_tool_function()
    assert tool_fn is not None, "Tool function is None"
    log(buf, f"  function: OK ({type(tool_fn).__name__})")

    sample = AddressValidationInput(address="1600 Amphitheatre Parkway, Mountain View, CA 94043")
    log(buf, f"  sample input: {sample.model_dump()}")

    try:
        result = await tool_fn.ainvoke({"address": "1600 Amphitheatre Parkway, Mountain View, CA 94043"})
        log(buf, f"  result type: {type(result).__name__}")
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Address Validation: structure OK")
    return True


async def _test_place_search(buf: io.StringIO, tm=None) -> bool:
    """Test PlaceSearchTool."""
    log(buf, "\n  --- Place Search Tool ---")
    tool = tm.get_tool("place_search_tool") if tm else PlaceSearchTool({"tool_endpoint": "http://localhost:8008"})

    log(buf, f"  name: {tool.get_tool_name()}")
    log(buf, f"  description: {tool.get_tool_description()}")

    schema = tool.get_tool_schema()
    assert schema is not None, "Schema is None"
    log(buf, f"  schema: OK")

    tool_fn = tool.get_tool_function()
    assert tool_fn is not None, "Tool function is None"
    log(buf, f"  function: OK ({type(tool_fn).__name__})")

    queries = ["New York City", "Times Square", "Times Square New York", "Central Park"]
    for query in queries:
        try:
            result = await tool_fn.ainvoke({"place_search_string": query})
            places = getattr(result, 'results', None) or getattr(result, 'place_details_list', None) or []
            log(buf, f"  query '{query}': {len(places)} places")
        except Exception as e:
            log(buf, f"  query '{query}': error (server may be down): {e}")

    log(buf, "  ✅ Place Search: structure OK")
    return True


async def _test_weather(buf: io.StringIO, tm=None) -> bool:
    """Test WeatherTool."""
    log(buf, "\n  --- Weather Tool ---")
    tool = tm.get_tool("weather_tool") if tm else WeatherTool({"tool_endpoint": "http://localhost:8008"})

    log(buf, f"  name: {tool.get_tool_name()}")
    log(buf, f"  description: {tool.get_tool_description()}")

    schema = tool.get_tool_schema()
    assert schema is not None, "Schema is None"
    log(buf, f"  schema: OK")

    tool_fn = tool.get_tool_function()
    assert tool_fn is not None, "Tool function is None"
    log(buf, f"  function: OK ({type(tool_fn).__name__})")

    sample = WeatherInput(
        place_label="New York City",
        latitude=40.7128, longitude=-74.0060,
        include_previous=False, use_archive=False, archive_date=""
    )
    log(buf, f"  sample input: NYC ({sample.latitude}, {sample.longitude})")

    try:
        result = await tool_fn.ainvoke({
            "place_label": "New York City",
            "latitude": 40.7128, "longitude": -74.0060,
            "include_previous": False, "use_archive": False, "archive_date": ""
        })
        log(buf, f"  result type: {type(result).__name__}")
        weather_data = getattr(result, 'weather_data', None)
        if weather_data:
            current = getattr(weather_data, 'current', None) or {}
            if isinstance(current, dict):
                temp = current.get('temperature_2m', '?')
            else:
                temp = getattr(current, 'temperature_2m', '?')
            log(buf, f"  weather_data: temp={temp}")
        else:
            log(buf, f"  weather_data: None")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Weather: structure OK")
    return True


async def _test_websearch(buf: io.StringIO, tm=None) -> bool:
    """Test WebSearchTool."""
    log(buf, "\n  --- Web Search Tool ---")
    tool = tm.get_tool("google_web_search_tool") if tm else WebSearchTool({"tool_endpoint": "http://localhost:8008"})

    log(buf, f"  name: {tool.get_tool_name()}")
    log(buf, f"  description: {tool.get_tool_description()}")

    schema = tool.get_tool_schema()
    assert schema is not None, "Schema is None"
    log(buf, f"  schema: OK")

    tool_fn = tool.get_tool_function()
    assert tool_fn is not None, "Tool function is None"
    log(buf, f"  function: OK ({type(tool_fn).__name__})")

    sample = WebSearchInput(search_query="Albert Einstein", num_results=5)
    log(buf, f"  sample input: {sample.model_dump()}")

    try:
        result = await tool_fn.ainvoke({"search_query": "Albert Einstein", "num_results": 3})
        log(buf, f"  result type: {type(result).__name__}")
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Web Search: structure OK")
    return True


async def _test_websearch_local_business(buf: io.StringIO, tm=None) -> bool:
    """Test web search with search_type=local and location for business discovery."""
    log(buf, "\n  --- Web Search: Local Business Discovery ---")
    tool = tm.get_tool("google_web_search_tool") if tm else WebSearchTool({"tool_endpoint": "http://localhost:8008"})
    tool_fn = tool.get_tool_function()

    # Pattern #1: Find a business via local search
    log(buf, "  [1] Local search: 'ALM RV LLC' in Norco,Louisiana")
    try:
        result = await tool_fn.ainvoke({
            "search_query": "ALM RV LLC",
            "search_type": "local",
            "location": "Norco,Louisiana",
            "num_results": 5
        })
        log(buf, f"  result type: {type(result).__name__}")
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
        for r in results:
            log(buf, f"    title: {r.title}")
            log(buf, f"    result_type: {r.result_type}")
            log(buf, f"    rating: {r.rating}")
            log(buf, f"    reviews: {r.reviews}")
            log(buf, f"    address: {r.address}")
            log(buf, f"    phone: {r.phone}")
            log(buf, f"    place_id: {r.place_id}")
            log(buf, f"    hours: {r.hours}")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    # Pattern #3: Find competitors in an area
    log(buf, "  [2] Local search: 'mobile RV repair' in Norco,Louisiana")
    try:
        result = await tool_fn.ainvoke({
            "search_query": "mobile RV repair",
            "search_type": "local",
            "location": "Norco,Louisiana",
            "num_results": 10
        })
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
        for r in results:
            log(buf, f"    {r.title} | rating={r.rating} reviews={r.reviews} place_id={r.place_id}")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Web Search Local Business: structure OK")
    return True


async def _test_websearch_ludocid_deepdive(buf: io.StringIO, tm=None) -> bool:
    """Test web search with ludocid parameter for business deep-dive."""
    log(buf, "\n  --- Web Search: ludocid Deep-Dive ---")
    tool = tm.get_tool("google_web_search_tool") if tm else WebSearchTool({"tool_endpoint": "http://localhost:8008"})
    tool_fn = tool.get_tool_function()

    # Step 1: Local search to get a place_id
    log(buf, "  [1] Local search to obtain place_id...")
    place_id = None
    try:
        result = await tool_fn.ainvoke({
            "search_query": "ALM RV LLC",
            "search_type": "local",
            "location": "Norco,Louisiana",
            "num_results": 5
        })
        results = getattr(result, 'results', None) or []
        for r in results:
            if getattr(r, 'place_id', None):
                place_id = r.place_id
                log(buf, f"  found place_id: {place_id} for '{r.title}'")
                break
        if not place_id:
            log(buf, "  no place_id found in local results, using fallback")
            place_id = "4103654625110011635"
    except Exception as e:
        log(buf, f"  local search error: {e}")
        place_id = "4103654625110011635"

    # Step 2: Deep-dive with ludocid (Pattern #2)
    log(buf, f"  [2] ludocid deep-dive with place_id={place_id}")
    try:
        result = await tool_fn.ainvoke({
            "search_query": "ALM RV LLC",
            "ludocid": place_id,
            "num_results": 10
        })
        log(buf, f"  result type: {type(result).__name__}")
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
        for r in results:
            log(buf, f"    {r.title} | {r.link}")
        kg = getattr(result, 'knowledge_graph', None)
        if kg:
            log(buf, f"  knowledge_graph.title: {kg.title}")
            log(buf, f"  knowledge_graph.type: {kg.type}")
            log(buf, f"  knowledge_graph.description: {kg.description}")
        else:
            log(buf, "  knowledge_graph: None")
        rq = getattr(result, 'related_questions', None) or []
        log(buf, f"  related_questions: {len(rq)} entries")
        for q in rq:
            log(buf, f"    Q: {q.question}")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Web Search ludocid Deep-Dive: structure OK")
    return True


async def _test_websearch_kgmid(buf: io.StringIO, tm=None) -> bool:
    """Test web search with kgmid parameter for Knowledge Graph entity lookup."""
    log(buf, "\n  --- Web Search: kgmid Entity Lookup ---")
    tool = tm.get_tool("google_web_search_tool") if tm else WebSearchTool({"tool_endpoint": "http://localhost:8008"})
    tool_fn = tool.get_tool_function()

    # Pattern #13: Knowledge Graph entity lookup
    # Using a well-known entity that should have a KGMID
    log(buf, "  [1] kgmid lookup for a known entity")
    try:
        result = await tool_fn.ainvoke({
            "search_query": "Tesla Inc",
            "kgmid": "/m/0dr90d",
            "num_results": 5
        })
        log(buf, f"  result type: {type(result).__name__}")
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
        for r in results:
            log(buf, f"    {r.title} | {r.link}")
        kg = getattr(result, 'knowledge_graph', None)
        if kg:
            log(buf, f"  knowledge_graph.title: {kg.title}")
            log(buf, f"  knowledge_graph.type: {kg.type}")
            log(buf, f"  knowledge_graph.description: {kg.description}")
        else:
            log(buf, "  knowledge_graph: None")
        api_error = getattr(result, 'api_error', None)
        api_status = getattr(result, 'api_status_code', None)
        if api_error:
            log(buf, f"  api_error: {api_error}")
        if api_status:
            log(buf, f"  api_status_code: {api_status}")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Web Search kgmid: structure OK")
    return True


async def _test_websearch_news_and_site(buf: io.StringIO, tm=None) -> bool:
    """Test web search with search_type=news, time_period, and site: operator patterns."""
    log(buf, "\n  --- Web Search: News & Site-Specific ---")
    tool = tm.get_tool("google_web_search_tool") if tm else WebSearchTool({"tool_endpoint": "http://localhost:8008"})
    tool_fn = tool.get_tool_function()

    # Pattern #8: Recent news about a business
    log(buf, "  [1] News search with time_period")
    try:
        result = await tool_fn.ainvoke({
            "search_query": "Tesla Inc",
            "search_type": "news",
            "time_period": "week",
            "num_results": 5
        })
        log(buf, f"  result type: {type(result).__name__}")
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
        for r in results:
            log(buf, f"    {r.title} | date={r.date} source={r.source} type={r.result_type}")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    # Pattern #4: Site-specific Yelp search
    log(buf, "  [2] Site-specific: Yelp")
    try:
        result = await tool_fn.ainvoke({
            "search_query": 'site:yelp.com "ALM RV" Norco',
            "num_results": 5
        })
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
        for r in results:
            log(buf, f"    {r.title} | {r.link}")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    # Pattern #5: Site-specific BBB search
    log(buf, "  [3] Site-specific: BBB")
    try:
        result = await tool_fn.ainvoke({
            "search_query": 'site:bbb.org "ALM RV LLC"',
            "num_results": 5
        })
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries (0 is normal for small businesses)")
        for r in results:
            log(buf, f"    {r.title} | {r.link}")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    # Pattern #9: Reputation check
    log(buf, "  [4] Reputation search")
    try:
        result = await tool_fn.ainvoke({
            "search_query": '"ALM RV LLC" reviews OR complaints OR scam',
            "num_results": 5
        })
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
        for r in results:
            log(buf, f"    {r.title} | {r.link}")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Web Search News & Site-Specific: structure OK")
    return True


# --- Main entry point ---

SUB_TESTS = [
    ("Address Validation", _test_address_validation),
    ("Place Search", _test_place_search),
    ("Weather", _test_weather),
    ("Web Search", _test_websearch),
    ("Web Search: Local Business", _test_websearch_local_business),
    ("Web Search: ludocid Deep-Dive", _test_websearch_ludocid_deepdive),
    ("Web Search: kgmid Entity", _test_websearch_kgmid),
    ("Web Search: News & Site-Specific", _test_websearch_news_and_site),
]


async def run() -> TestResult:
    """Run all direct tool tests (local models)."""
    load_dotenv()
    buf = io.StringIO()

    log(buf, "  === Direct Tool Tests (Local Models) ===")
    passed = []
    failed = []

    tm = create_tool_manager()

    for name, fn in SUB_TESTS:
        try:
            ok = await fn(buf, tm=tm)
            passed.append(name)
        except Exception as e:
            log(buf, f"  ❌ {name}: {e}")
            failed.append(name)

    log(buf, f"\n  === Results: {len(passed)}/{len(SUB_TESTS)} passed ===")
    for name in passed:
        log(buf, f"    ✅ {name}")
    for name in failed:
        log(buf, f"    ❌ {name}")

    write_log(buf, "tool_direct_local_run.log")

    assert not failed, f"Failed: {failed}"

    return TestResult(
        name="Direct Tool Tests (Local Models)",
        passed=True,
        details={
            "sub_tests": len(SUB_TESTS),
            "passed": len(passed),
            "tools": [n for n, _ in SUB_TESTS],
        },
    )
