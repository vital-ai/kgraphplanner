"""
Live check of the tool failure-reporting contract against the tool server on :8008.

Offline unit tests live in tests/test_tool_failure_reporting.py. This script
exercises the paths that can only be observed against a real server:

  1. a healthy call still succeeds and is NOT reported as an error;
  2. a genuine zero-result stays status="success";
  3. a real connection failure (bad port) is reported status="error";
  4. an auth failure (bad JWT) is reported status="error";
  5. a schema rejection (HTTP 422) is reported status="error";
  6. the server's own success/error body shape is what the client assumes.

Not covered here, deliberately:

  - HTTP 200 with success=false. No tool on this server can be made to produce it
    on demand -- bad inputs are rejected at 422 by pydantic before the tool runs,
    and the tools that would self-report a failure (loop_*) take a different
    schema. That row is covered offline in tests/test_tool_failure_reporting.py,
    which feeds the exact body shape to _parse_tool_response.
  - api_error on a successful response. Requires the upstream provider to fail,
    which cannot be arranged from here. Covered offline.

Run:  python -m test_scripts.test_tool_failure_reporting_live
"""
import asyncio
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

from test_scripts.cases.case_helpers import get_keycloak_token  # noqa: E402

ENDPOINT = os.getenv("TOOL_ENDPOINT", "http://localhost:8008")

PASS, FAIL = "PASS", "FAIL"
results = []


def record(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


async def invoke(tool_fn, **args):
    """Invoke as the agent does, so we observe ToolMessage.status."""
    return await tool_fn.ainvoke(
        {"type": "tool_call", "name": tool_fn.name, "args": args, "id": "live-1"}
    )


class _TM:
    """Stand-in for ToolManager: supplies the JWT, accepts registration.

    A real ToolManager is avoided deliberately -- the bad-JWT case below needs
    direct control over the token handed to the client.
    """

    def __init__(self, token):
        self._t = token

    def get_jwt_token(self):
        return self._t

    def add_tool(self, tool):
        pass


async def main():
    token, err = get_keycloak_token()
    if not token:
        print(f"Cannot obtain a JWT: {err}")
        return 2
    print(f"Tool server : {ENDPOINT}")
    print(f"JWT         : acquired ({len(token)} chars)\n")

    from kgraphplanner.tools.place_search.place_search_tool import PlaceSearchTool
    from kgraphplanner.tools.serper_websearch.serper_web_search_tool import SerperWebSearchTool

    good = {"tool_endpoint": ENDPOINT}
    tm = _TM(token)

    # ---- 1. raw server shape, so we know what the client is parsing ----
    print("1. Raw server response shape")
    body = {"tool": "place_search_tool", "request_id": "live", "timeout": 30,
            "tool_input": {"place_search_string": "Starbucks Union Square San Francisco"}}
    r = requests.post(f"{ENDPOINT}/tool", json=body,
                      headers={"Authorization": f"Bearer {token}"}, timeout=60)
    record("POST /tool returns 200", r.status_code == 200, f"status={r.status_code}")
    payload = r.json()
    record("body carries 'success'", "success" in payload, f"success={payload.get('success')}")
    if not payload.get("success"):
        record("server reported failure", False, f"error_message={payload.get('error_message')}")

    # ---- 2. healthy call is not reported as an error ----
    print("\n2. Healthy call stays successful")
    place = PlaceSearchTool(good, tool_manager=tm).get_tool_function()
    msg = await invoke(place, place_search_string="Starbucks Union Square San Francisco")
    record("status == success", msg.status == "success", f"status={msg.status}")
    record("content is non-empty", bool(str(msg.content).strip()),
           f"{str(msg.content)[:70]}...")

    # ---- 3. genuine zero-result stays successful ----
    # place_search is used rather than a web search: Serper returns fuzzy matches
    # for any string at all, so a web search can never reach zero results and
    # would make this assertion pass for the wrong reason.
    print("\n3. Genuine zero-result is not an error")
    nonsense = "zzqxkjvbnm998877 nonexistent business llc"
    raw = requests.post(f"{ENDPOINT}/tool",
                        json={"tool": "place_search_tool", "request_id": "live",
                              "timeout": 30, "tool_input": {"place_search_string": nonsense}},
                        headers={"Authorization": f"Bearer {token}"}, timeout=60).json()
    n = len((raw.get("tool_output") or {}).get("results") or [])
    record("the query really does match nothing", n == 0, f"server returned {n} results")
    msg = await invoke(place, place_search_string=nonsense)
    record("status == success", msg.status == "success", f"status={msg.status}")
    record("not reported as a failure", "unavailable" not in str(msg.content),
           str(msg.content)[:70])

    # ---- 4. real connection failure is reported ----
    print("\n4. Unreachable server is reported as an error")
    dead = PlaceSearchTool({"tool_endpoint": "http://localhost:9"}, tool_manager=tm).get_tool_function()
    msg = await invoke(dead, place_search_string="Acme LLC")
    record("status == error", msg.status == "error", f"status={msg.status}")
    record("message names the failure", "unavailable" in str(msg.content),
           str(msg.content)[:80])

    # ---- 5. auth failure is reported ----
    print("\n5. Auth failure is reported as an error")
    badauth = PlaceSearchTool(good, tool_manager=_TM("not-a-valid-jwt")).get_tool_function()
    msg = await invoke(badauth, place_search_string="Acme LLC")
    record("status == error", msg.status == "error", f"status={msg.status}")
    record("message is not 'no results'", "no results" not in str(msg.content).lower(),
           str(msg.content)[:90])

    # ---- 5b. what the server actually does with a payload it rejects ----
    # Observed by raw HTTP, not through the client: the client builds a pydantic
    # ToolRequest and so rejects a malformed payload locally before any request is
    # made. That is a good property, but it means a server 4xx is only reachable
    # under client/server schema skew. The client's >=400 handling is asserted
    # offline instead (test_client_converts_http_error_to_failure).
    print("\n5b. Server rejects a malformed payload with 4xx (raw HTTP)")
    r = requests.post(f"{ENDPOINT}/tool",
                      json={"tool": "place_search_tool", "request_id": "live", "timeout": 30,
                            "tool_input": {"place_search_string": 12345}},
                      headers={"Authorization": f"Bearer {token}"}, timeout=30)
    record("server returns 4xx", 400 <= r.status_code < 500, f"status={r.status_code}")

    # ---- 6. unconfigured endpoint ----
    print("\n6. Unconfigured endpoint is reported as an error")
    msg = await invoke(PlaceSearchTool({}, tool_manager=tm).get_tool_function(),
                       place_search_string="Acme LLC")
    record("status == error", msg.status == "error", f"status={msg.status}")
    record("message says not configured", "not configured" in str(msg.content),
           str(msg.content)[:80])

    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{'-'*60}\n{len(results)-len(failed)}/{len(results)} passed")
    for _, name, detail in failed:
        print(f"  FAIL: {name} -- {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
