"""
Tool logs follow ToolConfig.log_content (0.2.1, K5).

The bug: tools had no content switch. They logged the user's query at INFO
(web search and Serper on no results, the knowledge document tool on every
call), the loan lookup's identifier, service error bodies, exception text
(httpx errors carry the request URL) and full tracebacks, and the REST client
logged service error bodies and whole responses.

What must hold now:
  - with log_content off (the default), no query, identifier, error body or
    exception text reaches a log record at any level;
  - with log_content on, it does;
  - what a tool returns to the model is unchanged in both modes: the model
    still needs the reason to recover (the 0.1.5 tool-failure contract).
"""
import logging
from types import SimpleNamespace

import httpx
import pytest

from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
    VitalAgentRestResourceClient,
)

QUERY = "QUERY-SENTINEL-555"
BODY = "BODY-SENTINEL-777"
ENDPOINT = "http://localhost:8008"


async def invoke(tool_fn, **args):
    """Invoke as the agent does, so failures come back as a ToolMessage."""
    return await tool_fn.ainvoke({"type": "tool_call", "name": tool_fn.name, "args": args, "id": "c1"})


def logs(caplog):
    return "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("kgraphplanner"))


@pytest.fixture
def server_error(monkeypatch):
    """The tool server answers 503 with an error body holding BODY."""

    class _Resp:
        status_code = 503
        text = f"upstream failed for {BODY}"

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client())


def make(tool_cls, log_content):
    return tool_cls({"tool_endpoint": ENDPOINT, "log_content": log_content}).get_tool_function()


SEARCH_TOOLS = [
    ("kgraphplanner.tools.websearch.web_search_tool", "WebSearchTool", {"search_query": QUERY}),
    ("kgraphplanner.tools.serper_websearch.serper_web_search_tool", "SerperWebSearchTool", {"search_query": QUERY}),
    ("kgraphplanner.tools.place_search.place_search_tool", "PlaceSearchTool", {"place_search_string": QUERY}),
    ("kgraphplanner.tools.address_validation.address_validation_tool", "AddressValidationTool", {"address": QUERY}),
    ("kgraphplanner.tools.weather.weather_tool", "WeatherTool", {"latitude": 1.0, "longitude": 2.0,
                                                                  "place_label": QUERY}),
]


def tool_class(module, name):
    import importlib
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("module,cls,args", SEARCH_TOOLS, ids=[c for _, c, _ in SEARCH_TOOLS])
async def test_service_failure_logs_no_content(module, cls, args, server_error, caplog):
    caplog.set_level(logging.DEBUG)
    msg = await invoke(make(tool_class(module, cls), False), **args)
    assert msg.status == "error"
    assert BODY in msg.content                        # the model still sees the reason
    text = logs(caplog)
    assert "503" in text                              # the status is still logged
    assert QUERY not in text and BODY not in text


@pytest.mark.parametrize("module,cls,args", SEARCH_TOOLS, ids=[c for _, c, _ in SEARCH_TOOLS])
async def test_log_content_logs_it(module, cls, args, server_error, caplog):
    caplog.set_level(logging.DEBUG)
    await invoke(make(tool_class(module, cls), True), **args)
    assert BODY in logs(caplog)


@pytest.mark.parametrize("module,cls,args", SEARCH_TOOLS, ids=[c for _, c, _ in SEARCH_TOOLS])
async def test_exception_text_and_traceback_are_not_logged(module, cls, args, monkeypatch, caplog):
    # httpx errors carry the request URL; a query string would leak with it.
    caplog.set_level(logging.DEBUG)

    async def boom(*a, **k):
        raise httpx.ConnectError(f"connection refused for /search?q={QUERY}")

    monkeypatch.setattr(VitalAgentRestResourceClient, "handle_tool_request", boom)
    msg = await invoke(make(tool_class(module, cls), False), **args)
    assert msg.status == "error"
    text = logs(caplog)
    assert "ConnectError" in text and "connection" in text
    assert QUERY not in text and "Traceback" not in text


@pytest.mark.parametrize("module,cls", [
    ("kgraphplanner.tools.websearch.web_search_tool", "WebSearchTool"),
    ("kgraphplanner.tools.serper_websearch.serper_web_search_tool", "SerperWebSearchTool"),
])
async def test_no_results_logs_no_query(module, cls, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    async def empty(*a, **k):
        return SimpleNamespace(success=True, tool_output=SimpleNamespace(results=[], api_error=None))

    monkeypatch.setattr(VitalAgentRestResourceClient, "handle_tool_request", empty)
    msg = await invoke(make(tool_class(module, cls), False), search_query=QUERY)
    assert QUERY in msg.content                       # returned to the model as before
    text = logs(caplog)
    assert "no results" in text.lower() or "empty results" in text
    assert QUERY not in text


async def test_knowledge_get_document_logs_no_query(caplog):
    from kgraphplanner.tools.knowledge.knowledge_get_document_tool import KnowledgeGetDocumentTool

    caplog.set_level(logging.DEBUG)
    tool = KnowledgeGetDocumentTool({"log_content": False})
    no_objects = SimpleNamespace(objects=[])
    collection = SimpleNamespace(query=SimpleNamespace(bm25=lambda **k: no_objects,
                                                       fetch_objects=lambda **k: no_objects))
    tool._ensure_client = lambda: SimpleNamespace(collections=SimpleNamespace(get=lambda name: collection))
    tool._wv_config = SimpleNamespace(collection_prefix="")
    await invoke(tool.get_tool_function(), query=QUERY)
    await invoke(tool.get_tool_function(), source=QUERY)
    assert "knowledge_get_document_tool" in logs(caplog)
    assert QUERY not in logs(caplog)


async def test_knowledge_search_logs_no_query_or_error_text(caplog):
    from kgraphplanner.tools.knowledge.knowledge_search_tool import KnowledgeSearchTool

    caplog.set_level(logging.DEBUG)
    tool = KnowledgeSearchTool({"log_content": False})

    def down():
        raise RuntimeError(f"weaviate unreachable while searching {QUERY}")

    tool._ensure_init = down
    msg = await invoke(tool.get_tool_function(), query=QUERY)
    assert msg.status == "error"
    text = logs(caplog)
    assert "RuntimeError" in text and QUERY not in text


async def test_loan_lookup_logs_no_identifier(caplog):
    from kgraphplanner.tools.mock_loan_lookup.loan_lookup_tool import LoanLookupTool

    caplog.set_level(logging.DEBUG)
    fn = LoanLookupTool({"log_content": False}).get_tool_function()
    await invoke(fn, identifier_type="email", identifier_value=f"{QUERY}@example.com")
    assert "loan_lookup_tool" in logs(caplog) and QUERY not in logs(caplog)


async def test_github_service_failure_logs_no_exception_text(monkeypatch, caplog):
    from kgraphplanner.tools.github.github_service_tool import GitHubServiceTool

    caplog.set_level(logging.DEBUG)

    async def boom(*a, **k):
        raise httpx.ConnectError(f"refused for {QUERY}")

    monkeypatch.setattr(VitalAgentRestResourceClient, "handle_tool_request", boom)
    tool = next(c for c in GitHubServiceTool.__subclasses__())(
        {"tool_endpoint": ENDPOINT, "repos": ["vital-ai/example"]})
    _, error = await tool.call_service(None)
    assert QUERY in error                             # returned for the model
    assert "ConnectError" in logs(caplog) and QUERY not in logs(caplog)


async def test_rest_client_withholds_bodies_and_responses(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    class _Resp:
        def __init__(self, status, body):
            self.status_code, self.text, self._body = status, body, body

        def json(self):
            return {"success": True, "tool_output": {"note": self._body}}

    def client_with(resp):
        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return resp
        return _Client()

    for log_content in (False, True):
        caplog.clear()
        client = VitalAgentRestResourceClient({"tool_endpoint": ENDPOINT, "log_content": log_content}, None)
        monkeypatch.setattr(client, "_build_request", lambda *a, **k: (f"{ENDPOINT}/tool", {}, {}))
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: client_with(_Resp(500, BODY)))
        failed = await client.handle_tool_request("place_search_tool", None)
        assert BODY in failed.error_message           # returned to the caller either way
        assert (BODY in logs(caplog)) is log_content


def test_flag_reaches_every_tool_through_the_shared_settings():
    on = AgentConfig.from_dict({"tools": {"endpoint": ENDPOINT, "log_content": True}})
    off = AgentConfig.from_dict({"tools": {"endpoint": ENDPOINT}})
    assert on.get_config_for_tool("weather_tool")["log_content"] is True
    assert off.get_config_for_tool("weather_tool")["log_content"] is False
    assert AgentConfig.from_dict({"tools": on.get_tool_config()}).tools.log_content is True
