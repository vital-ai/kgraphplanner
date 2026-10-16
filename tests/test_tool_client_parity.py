"""The tool client keeps every field the tool service returns.

The client's models used to fall behind the service's silently: pydantic
ignores undeclared fields, so a field the service added (Serper's `category`,
place search's `rating`) never reached the agent and nothing failed.

- ``tests/data/tool_service_fields.json`` records the service's field names
  per model. Regenerate it with ``scripts/snapshot_tool_service_fields.py``
  after a service change.
- The parity test checks every client model has those fields.
- The handler tests check that a field present on the wire survives the
  handler, so a field-by-field handler can't drop one either.
"""
from __future__ import annotations

import importlib
import json
import logging
import pkgutil
from pathlib import Path

import pytest
from pydantic import BaseModel

from kgraphplanner.vital_agent_rest_resource_client.tools.place_search.tool_handler import PlaceSearchToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.serper_web_search.models import (
    SerperSearchResult, SerperWebSearchInput,
)
from kgraphplanner.vital_agent_rest_resource_client.tools.serper_web_search.tool_handler import (
    SerperWebSearchToolHandler,
)

SNAPSHOT = json.loads((Path(__file__).parent / "data" / "tool_service_fields.json").read_text())
CLIENT = "kgraphplanner.vital_agent_rest_resource_client.tools."

# Service model -> client class, where the names differ.
SERVICE_TO_CLIENT = {
    "serper_web_search.SerperWebSearchResult": "SerperSearchResult",
    "serper_web_search.SerperPeopleAlsoAsk": "SerperRelatedQuestion",
}


def client_models(package: str) -> dict[str, type[BaseModel]]:
    """Models in the client modules that mirror the service: models.py and
    *_models.py. The older *_request.py / *_response.py modules are not
    mirrors and may reuse the same class names."""
    pkg = importlib.import_module(CLIENT + package)
    out = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name != "models" and not info.name.endswith("_models"):
            continue
        module = importlib.import_module(f"{CLIENT}{package}.{info.name}")
        for name, value in vars(module).items():
            if isinstance(value, type) and issubclass(value, BaseModel) and value.__module__ == module.__name__:
                out[name] = value
    return out


@pytest.mark.parametrize("key", sorted(SNAPSHOT["models"]))
def test_client_model_has_every_service_field(key):
    package, name = key.split(".")
    models = client_models(package)
    client_name = SERVICE_TO_CLIENT.get(key, name)
    assert client_name in models, f"no client model {client_name} for service {key}"
    missing = [f for f in SNAPSHOT["models"][key] if f not in models[client_name].model_fields]
    assert not missing, (f"{CLIENT}{package} {client_name} lacks {missing} (service "
                         f"{SNAPSHOT['service_commit']}); add them to the model")


def test_rename_map_points_at_real_models():
    for key, client_name in SERVICE_TO_CLIENT.items():
        assert key in SNAPSHOT["models"]
        assert client_name in client_models(key.split(".")[0])


# ---------------------------------------------------------------------------
# Handlers keep what's on the wire
# ---------------------------------------------------------------------------

# Recorded from the local service (61f1900), values replaced.
SERPER_PLACE = {
    "title": "ALM RV LLC", "link": "", "position": 1, "result_type": "place",
    "rating": 4.9, "rating_count": 196, "address": "1 Main St", "phone_number": "(555) 010-0000",
    "website": "https://example.test", "category": "RV repair shop", "cid": "123",
    "latitude": 39.1, "longitude": -84.5, "price_level": "$$",
    "sitelinks": [{"title": "Hours", "link": "https://example.test/hours"}],
}
PLACE = {
    "name": "Roto-Rooter", "address": "2 Main St", "place_id": "ChIJx", "latitude": 39.1,
    "longitude": -84.5, "business_status": "OPERATIONAL", "icon": "https://example.test/i.png",
    "types": ["plumber", "point_of_interest"], "url": "https://maps.example.test/?cid=1",
    "vicinity": "Cincinnati", "formatted_phone_number": "(555) 010-0001",
    "website": "https://example.test", "rating": 4.6, "user_ratings_total": 812,
}


def serper(results, **extra):
    return SerperWebSearchToolHandler().handle_response(
        SerperWebSearchInput(search_query="q"),
        {"tool_output": {"results": results, **extra}})


def test_serper_handler_keeps_every_place_field():
    result = serper([SERPER_PLACE]).results[0]
    kept = result.model_dump()
    assert {k: kept[k] for k in SERPER_PLACE} == SERPER_PLACE


def test_serper_phone_alias_is_filled_but_not_serialised():
    result = serper([SERPER_PLACE]).results[0]
    assert result.phone == result.phone_number == "(555) 010-0000"
    assert "phone" not in result.model_dump()
    # An older payload that only has `phone` still fills phone_number.
    assert SerperSearchResult(title="t", phone="(555) 1").phone_number == "(555) 1"


def test_serper_handler_keeps_knowledge_graph_fields():
    kg = {"title": "ALM RV", "type": "RV repair shop", "website": "https://example.test",
          "image_url": "https://example.test/kg.png", "attributes": {"Hours": "9-5"}}
    out = serper([], knowledge_graph=kg).knowledge_graph.model_dump()
    assert {k: out[k] for k in kg} == kg


def test_serper_handler_still_reads_older_key_spellings():
    result = serper([{"title": "t", "url": "https://a.test", "display_url": "a.test"}]).results[0]
    assert (result.link, result.displayed_link, result.result_type) == ("https://a.test", "a.test", "organic")


def test_a_bad_optional_field_is_dropped_not_fatal(caplog):
    with caplog.at_level(logging.WARNING):
        result = serper([{**SERPER_PLACE, "latitude": "not a number"}]).results[0]
    assert result.latitude is None and result.category == "RV repair shop"
    assert "latitude" in caplog.text and "not a number" not in caplog.text


def test_place_handler_keeps_every_field():
    out = PlaceSearchToolHandler().handle_response(None, {"tool_output": {"results": [PLACE]}})
    kept = out.results[0].model_dump()
    assert {k: kept[k] for k in PLACE} == PLACE


def test_place_results_past_the_details_cap_parse():
    # Past the service's Details limit, phone/website/url/vicinity are null.
    capped = {k: v for k, v in PLACE.items()
              if k not in ("formatted_phone_number", "website", "url", "vicinity")}
    out = PlaceSearchToolHandler().handle_response(None, {"tool_output": {"results": [capped]}})
    assert out.results[0].rating == 4.6 and out.results[0].website is None


async def test_serper_tool_forwards_language_country_and_page(monkeypatch):
    from kgraphplanner.tools.serper_websearch.serper_web_search_tool import SerperWebSearchTool
    from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
        VitalAgentRestResourceClient,
    )
    sent = {}

    async def fake_request(self, tool_name, tool_parameters):
        sent.update(tool_parameters.model_dump())
        raise RuntimeError("stop here")

    monkeypatch.setattr(VitalAgentRestResourceClient, "handle_tool_request", fake_request)
    tool_fn = SerperWebSearchTool({"tool_endpoint": "http://localhost:8008"}).get_tool_function()
    await tool_fn.ainvoke({"type": "tool_call", "name": tool_fn.name, "id": "c1", "args": {
        "search_query": "q", "language": "en", "country": "us", "page": 2}})
    assert (sent["language"], sent["country"], sent["page"]) == ("en", "us", 2)
