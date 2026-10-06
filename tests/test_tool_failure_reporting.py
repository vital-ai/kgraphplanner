"""
Regression tests for the tool failure-reporting contract.

The bug: every agent-facing tool caught its exceptions and returned a plausible
empty result -- `"No search results available for query: X"`,
`PlaceSearchOutput(results=[])`, a fabricated `WeatherData` with empty readings.
LangChain stamped the `ToolMessage` `status="success"` either way, so a stopped
tool server was indistinguishable from a genuine negative answer and got reported
to merchants as fact.

What must hold now:
  - a tool that could not complete produces `ToolMessage(status="error")`;
  - the model still sees the same message text, so agent behaviour is unchanged;
  - a genuine zero-result stays a successful, empty result. An unknown business
    really does have no results, and reporting that as a failure would be a
    different bug of the same family.
"""
import json

import pytest
from langchain_core.tools import ToolException

from kgraphplanner.tools.tool_errors import (
    check_api_error,
    fail,
    fail_if_error_payload,
    reports_failures,
)


class _Out:
    """Stands in for a service output model carrying the api_error contract."""

    def __init__(self, api_error=None, api_status_code=None, results=None):
        self.api_error = api_error
        self.api_status_code = api_status_code
        self.results = results or []


async def _invoke(tool, **args):
    """Invoke as the agent does, so we get a ToolMessage rather than a raw return."""
    return await tool.ainvoke(
        {"type": "tool_call", "name": tool.name, "args": args, "id": "call-1"}
    )


# ----------------------------------------------------------------------
# The contract primitives
# ----------------------------------------------------------------------


def test_fail_raises_tool_exception_not_bare_exception():
    """A bare Exception would be re-raised by the LangGraph tool node, aborting
    the run instead of handing the failure back to the model."""
    with pytest.raises(ToolException) as exc:
        fail("the thing is unavailable")
    assert "the thing is unavailable" in str(exc.value)


def test_fail_includes_status_when_given():
    with pytest.raises(ToolException) as exc:
        fail("denied", status=403)
    assert "403" in str(exc.value)


def test_check_api_error_fails_on_api_error():
    """An upstream provider failure arrives as a *successful* response with zero
    results and api_error set. Nothing raises in that path, so this check is the
    only thing standing between it and "this business has no results"."""
    with pytest.raises(ToolException) as exc:
        check_api_error(_Out(api_error="Serper quota exceeded", api_status_code=429))
    assert "Serper quota exceeded" in str(exc.value)
    assert "429" in str(exc.value)


def test_check_api_error_passes_a_genuine_zero_result():
    """results=[] with no api_error is a real negative answer. Must not raise."""
    check_api_error(_Out(results=[]))


def test_check_api_error_tolerates_models_without_the_field():
    """place_search / address_validation outputs carry no api_error field."""
    check_api_error(object())


def test_fail_if_error_payload_reraises_an_error_object():
    with pytest.raises(ToolException) as exc:
        fail_if_error_payload(json.dumps({"error": "repo is required", "status_code": 400}))
    # Verbatim: the service writes these to be actionable for the model.
    assert str(exc.value) == "repo is required"


def test_fail_if_error_payload_ignores_success_payloads():
    fail_if_error_payload(json.dumps({"results": [], "count": 0}))
    fail_if_error_payload("not json at all")
    fail_if_error_payload(json.dumps(["a", "list"]))


# ----------------------------------------------------------------------
# End to end through the tool boundary
# ----------------------------------------------------------------------


async def test_failure_becomes_tool_message_status_error():
    """The property cardiff-sales-agent needs: detectable without text matching."""
    from pydantic import BaseModel
    from langchain_core.tools import tool

    class In(BaseModel):
        query: str

    @tool(args_schema=In)
    async def searcher(query: str) -> str:
        """Search."""
        fail(f"No search results available for query: {query} (error: ConnectError: refused)")

    msg = await _invoke(reports_failures(searcher), query="acme")
    assert msg.status == "error"
    # Same text as before the change, so the model behaves identically.
    assert msg.content == "No search results available for query: acme (error: ConnectError: refused)"


async def test_genuine_zero_result_stays_success():
    from pydantic import BaseModel
    from langchain_core.tools import tool

    class In(BaseModel):
        query: str

    @tool(args_schema=In)
    async def searcher(query: str) -> str:
        """Search."""
        return f"Search returned no results for query: {query}"

    msg = await _invoke(reports_failures(searcher), query="nonexistent llc")
    assert msg.status == "success"
    assert "no results" in msg.content


async def test_without_reports_failures_the_exception_escapes():
    """Guards the reason reports_failures exists: unset handle_tool_error makes
    langchain-core re-raise, which aborts the run instead of reporting."""
    from pydantic import BaseModel
    from langchain_core.tools import tool

    class In(BaseModel):
        query: str

    @tool(args_schema=In)
    async def searcher(query: str) -> str:
        """Search."""
        fail("boom")

    with pytest.raises(ToolException):
        await _invoke(searcher, query="x")


# ----------------------------------------------------------------------
# Every agent-facing tool is wired into the contract
# ----------------------------------------------------------------------


def test_all_patched_tools_set_handle_tool_error():
    """A tool that raises but lacks handle_tool_error aborts the run. Keep this
    list in step with the tools that call fail()."""
    import pathlib
    import re

    root = pathlib.Path("kgraphplanner/tools")
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        if re.search(r"\bfail\(|\bfail_if_error_payload\(", text) and path.name != "tool_errors.py":
            if "reports_failures(" not in text:
                offenders.append(str(path))
    assert not offenders, f"call fail() but never reports_failures(): {offenders}"


# ----------------------------------------------------------------------
# The client layer: a tool that caught its own exception
# ----------------------------------------------------------------------


def test_client_honours_body_success_over_http_200():
    """The service reports a caught tool failure as HTTP 200 with success=false,
    and its own `to_dict()` replaces the null tool_output with {"results": []}.
    Every parse branch called create_success unconditionally, so that failure
    was stamped successful and reached the agent as a well-formed empty result.
    This is the one row no wrapper-level change can reach.
    """
    from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
        VitalAgentRestResourceClient,
    )

    client = VitalAgentRestResourceClient({"tool_endpoint": "http://localhost:8008"}, None)
    body = {
        "success": False,
        "error_message": "googlemaps: API key not authorised",
        "tool_output": {"results": []},   # synthesised by the service, not real
    }
    parsed = client._parse_tool_response("place_search_tool", None, body)

    assert parsed.success is False
    assert "not authorised" in parsed.error_message
    assert parsed.tool_output is None


def test_client_still_parses_a_successful_body():
    from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
        VitalAgentRestResourceClient,
    )

    client = VitalAgentRestResourceClient({"tool_endpoint": "http://localhost:8008"}, None)
    body = {"success": True, "tool_output": {"tool": "place_search_tool", "results": []}}
    parsed = client._parse_tool_response("place_search_tool", None, body)
    assert parsed.success is True


# ----------------------------------------------------------------------
# The real wrappers
# ----------------------------------------------------------------------


async def test_place_search_reports_an_unreachable_server_as_an_error():
    """The 2026-10-05 incident: :8008 stopped. A stopped server emits nothing, so
    this is only ever detectable here."""
    import httpx

    from kgraphplanner.tools.place_search.place_search_tool import PlaceSearchTool

    tool_fn = PlaceSearchTool({"tool_endpoint": "http://localhost:8008"}).get_tool_function()

    async def boom(*a, **k):
        raise httpx.ConnectError("[Errno 61] Connection refused")

    import kgraphplanner.tools.place_search.place_search_tool as mod
    original = mod.VitalAgentRestResourceClient.handle_tool_request
    mod.VitalAgentRestResourceClient.handle_tool_request = boom
    try:
        msg = await _invoke(tool_fn, place_search_string="Acme LLC")
    finally:
        mod.VitalAgentRestResourceClient.handle_tool_request = original

    assert msg.status == "error"
    assert "unavailable" in msg.content
    assert "Connection refused" in msg.content


async def test_unconfigured_endpoint_is_an_error_not_an_empty_result():
    from kgraphplanner.tools.place_search.place_search_tool import PlaceSearchTool

    tool_fn = PlaceSearchTool({}).get_tool_function()
    msg = await _invoke(tool_fn, place_search_string="Acme LLC")
    assert msg.status == "error"
    assert "not configured" in msg.content


async def test_github_error_payload_becomes_status_error_without_changing_run():
    """run() keeps returning the {"error": ...} JSON its callers and tests rely on;
    only the agent boundary gains the status."""
    from kgraphplanner.tools.github.issue_tools import GitHubGetIssueTool

    # No endpoint configured: reaches _dispatch and returns an error payload.
    # (An out-of-scope repo cannot be used here -- the schema makes `repo` a
    # Literal of the configured set, so the agent boundary rejects it first,
    # which is that design working as intended.)
    tool = GitHubGetIssueTool({"tool_endpoint": "", "repos": ["a/b"]})

    # Internal contract unchanged.
    raw = await tool.run(repo="a/b", issue_number=1)
    assert "endpoint is not configured" in json.loads(raw)["error"]

    # Agent boundary now reports it.
    msg = await _invoke(tool.get_tool_function(), repo="a/b", issue_number=1)
    assert msg.status == "error"
    assert "endpoint is not configured" in msg.content


async def test_client_converts_http_error_to_failure(monkeypatch):
    """A 4xx/5xx from the tool server must become a failed ToolResponse carrying
    the server's text. Not reachable live -- the client validates its own
    ToolRequest before sending, so a malformed payload never leaves the process --
    so the branch is exercised here with a stubbed transport."""
    import httpx

    from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
        VitalAgentRestResourceClient,
    )

    client = VitalAgentRestResourceClient({"tool_endpoint": "http://localhost:8008"}, None)
    monkeypatch.setattr(
        client, "_build_request",
        lambda *a, **k: ("http://localhost:8008/tool", {}, {}),
    )

    class _Resp:
        status_code = 503
        text = "upstream unavailable"

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client())

    parsed = await client.handle_tool_request("place_search_tool", None)
    assert parsed.success is False
    assert "503" in parsed.error_message
    assert "upstream unavailable" in parsed.error_message
    assert parsed.tool_output is None


async def test_client_raises_on_auth_failure(monkeypatch):
    """401/403 raises rather than returning, so a wrapper's `except` turns it into
    a ToolException instead of it being mistaken for an empty result."""
    import httpx

    from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
        VitalAgentRestResourceClient,
    )

    client = VitalAgentRestResourceClient({"tool_endpoint": "http://localhost:8008"}, None)
    monkeypatch.setattr(client, "_build_request",
                        lambda *a, **k: ("http://localhost:8008/tool", {}, {}))

    class _Resp:
        status_code = 401
        text = "Malformed JWT token"

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client())

    with pytest.raises(PermissionError) as exc:
        await client.handle_tool_request("place_search_tool", None)
    assert "401" in str(exc.value)
