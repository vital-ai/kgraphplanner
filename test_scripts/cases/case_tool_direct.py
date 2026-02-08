"""
Case: Direct tool tests — address validation, place search, weather, web search.

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
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.google_address_validation.models import AddressValidationInput
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.place_search.models import PlaceSearchInput
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.weather.models import WeatherInput
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.web_search.models import WebSearchInput

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import create_tool_manager, log, write_log, OUTPUT_DIR


# --- Sub-tests ---

def _test_address_validation(buf: io.StringIO, tm=None) -> bool:
    """Test AddressValidationTool."""
    log(buf, "\n  --- Address Validation Tool ---")
    tool = tm.get_tool("google_address_validation_tool") if tm else AddressValidationTool({"tool_endpoint": "http://localhost:8008"})

    log(buf, f"  name: {tool.get_tool_name()}")
    log(buf, f"  description: {tool.get_tool_description()[:80]}")

    schema = tool.get_tool_schema()
    assert schema is not None, "Schema is None"
    log(buf, f"  schema: OK")

    tool_fn = tool.get_tool_function()
    assert tool_fn is not None, "Tool function is None"
    log(buf, f"  function: OK ({type(tool_fn).__name__})")

    sample = AddressValidationInput(address="1600 Amphitheatre Parkway, Mountain View, CA 94043")
    log(buf, f"  sample input: {sample.model_dump()}")

    try:
        result = tool_fn.invoke({"address": "1600 Amphitheatre Parkway, Mountain View, CA 94043"})
        log(buf, f"  result type: {type(result).__name__}")
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Address Validation: structure OK")
    return True


def _test_place_search(buf: io.StringIO, tm=None) -> bool:
    """Test PlaceSearchTool."""
    log(buf, "\n  --- Place Search Tool ---")
    tool = tm.get_tool("place_search_tool") if tm else PlaceSearchTool({"tool_endpoint": "http://localhost:8008"})

    log(buf, f"  name: {tool.get_tool_name()}")
    log(buf, f"  description: {tool.get_tool_description()[:80]}")

    schema = tool.get_tool_schema()
    assert schema is not None, "Schema is None"
    log(buf, f"  schema: OK")

    tool_fn = tool.get_tool_function()
    assert tool_fn is not None, "Tool function is None"
    log(buf, f"  function: OK ({type(tool_fn).__name__})")

    queries = ["New York City", "Times Square", "Times Square New York", "Central Park"]
    for query in queries:
        try:
            result = tool_fn.invoke({"place_search_string": query})
            places = getattr(result, 'results', None) or getattr(result, 'place_details_list', None) or []
            log(buf, f"  query '{query}': {len(places)} places")
        except Exception as e:
            log(buf, f"  query '{query}': error (server may be down): {e}")

    log(buf, "  ✅ Place Search: structure OK")
    return True


def _test_weather(buf: io.StringIO, tm=None) -> bool:
    """Test WeatherTool."""
    log(buf, "\n  --- Weather Tool ---")
    tool = tm.get_tool("weather_tool") if tm else WeatherTool({"tool_endpoint": "http://localhost:8008"})

    log(buf, f"  name: {tool.get_tool_name()}")
    log(buf, f"  description: {tool.get_tool_description()[:80]}")

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
        result = tool_fn.invoke({
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


def _test_websearch(buf: io.StringIO, tm=None) -> bool:
    """Test WebSearchTool."""
    log(buf, "\n  --- Web Search Tool ---")
    tool = tm.get_tool("google_web_search_tool") if tm else WebSearchTool({"tool_endpoint": "http://localhost:8008"})

    log(buf, f"  name: {tool.get_tool_name()}")
    log(buf, f"  description: {tool.get_tool_description()[:80]}")

    schema = tool.get_tool_schema()
    assert schema is not None, "Schema is None"
    log(buf, f"  schema: OK")

    tool_fn = tool.get_tool_function()
    assert tool_fn is not None, "Tool function is None"
    log(buf, f"  function: OK ({type(tool_fn).__name__})")

    sample = WebSearchInput(search_query="Albert Einstein", num_results=5)
    log(buf, f"  sample input: {sample.model_dump()}")

    try:
        result = tool_fn.invoke({"search_query": "Albert Einstein", "num_results": 3})
        log(buf, f"  result type: {type(result).__name__}")
        results = getattr(result, 'results', None) or []
        log(buf, f"  results: {len(results)} entries")
    except Exception as e:
        log(buf, f"  invoke error (server may be down): {e}")

    log(buf, "  ✅ Web Search: structure OK")
    return True


# --- Main entry point ---

SUB_TESTS = [
    ("Address Validation", _test_address_validation),
    ("Place Search", _test_place_search),
    ("Weather", _test_weather),
    ("Web Search", _test_websearch),
]


async def run() -> TestResult:
    """Run all direct tool tests."""
    load_dotenv()
    buf = io.StringIO()

    log(buf, "  === Direct Tool Tests ===")
    passed = []
    failed = []

    tm = create_tool_manager()

    for name, fn in SUB_TESTS:
        try:
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

    write_log(buf, "tool_direct_run.log")

    assert not failed, f"Failed: {failed}"

    return TestResult(
        name="Direct Tool Tests",
        passed=True,
        details={
            "sub_tests": len(SUB_TESTS),
            "passed": len(passed),
            "tools": [n for n, _ in SUB_TESTS],
        },
    )
