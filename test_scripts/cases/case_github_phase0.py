"""
Case: GitHub tools phase-0 spike.

Validates the wire contract between kgraphplanner and the github_* tools in the
vital-agent-resource REST service, before any of the 33 agent-facing tools are
written. See planning/kg_tools/github_tools_plan.md section 11.

What it proves:
  1. ToolRequest's plain Union accepts the GitHub input models and does not
     coerce them to a sibling  (plan section 9.1 -- this was an open risk)
  2. A real round trip through VitalAgentRestResourceClient with a JWT
  3. api_error survives intact rather than being flattened  (plan section 4)
  4. The pagination fields survive  (plan section 5.2)

Requires the service running with GitHub configured, and a JWT. Set:

    VITAL_TOOL_ENDPOINT     default http://localhost:8008
    VITAL_TOOL_JWT          bearer token; without it the service returns 401
    GITHUB_TEST_OWNER       default vital-ai
    GITHUB_TEST_REPO        default vital-ai-sandbox

Run:  <conda>/bin/python test_scripts/cases/case_github_phase0.py
"""

from __future__ import annotations

import asyncio
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from kgraphplanner.vital_agent_rest_resource_client.tools.tool_request import ToolRequest
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName
from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
    VitalAgentRestResourceClient
)
from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
    GitHubIssueGetInput, GitHubIssueListInput, GitHubIssueSearchInput,
    GitHubIssueToolOutput
)

ENDPOINT = os.getenv("VITAL_TOOL_ENDPOINT", "http://localhost:8008")
JWT = os.getenv("VITAL_TOOL_JWT")
OWNER = os.getenv("GITHUB_TEST_OWNER", "vital-ai")
REPO = os.getenv("GITHUB_TEST_REPO", "vital-ai-sandbox")

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + ("" if condition else f" -- {detail}"))


def _client():
    return VitalAgentRestResourceClient({"tool_endpoint": ENDPOINT}, JWT)


# ---------------------------------------------------------------------------
# 1. Union behaviour -- offline, no service needed
# ---------------------------------------------------------------------------

def test_union():
    """Plan section 9.1: 33 models in a plain Union risks mis-coercion.

    The GitHub inputs carry a Literal `operation`, so pydantic's smart union
    should match by exact type rather than by field overlap. Confirm, don't assume.
    """
    print("\n1. ToolRequest union resolution")

    cases = [
        (GitHubIssueGetInput(operation="get_issue", owner=OWNER, repo=REPO, issue_number=1),
         "GitHubIssueGetInput"),
        (GitHubIssueListInput(operation="list_issues", owner=OWNER, repo=REPO, max_results=3),
         "GitHubIssueListInput"),
        (GitHubIssueSearchInput(operation="search_issues", owner=OWNER, repo=REPO, query="test"),
         "GitHubIssueSearchInput"),
    ]

    for vi, expected in cases:
        try:
            tr = ToolRequest(tool=ToolName.github_issue_tool.value, tool_input=vi)
        except Exception as e:
            check(f"{expected} accepted by the union", False, f"{type(e).__name__}: {e}")
            continue
        check(f"{expected} accepted by the union", True)
        check(f"{expected} not coerced to a sibling model",
              type(tr.tool_input).__name__ == expected,
              f"became {type(tr.tool_input).__name__}")

    # The payload must carry the operation discriminator, or the service cannot
    # pick a handler.
    tr = ToolRequest(tool=ToolName.github_issue_tool.value,
                     tool_input=cases[0][0])
    payload = tr.to_dict()
    check("payload carries the operation discriminator",
          payload.get("tool_input", {}).get("operation") == "get_issue", str(payload)[:160])
    check("payload carries the service tool name",
          payload.get("tool") == "github_issue_tool", str(payload.get("tool")))


# ---------------------------------------------------------------------------
# 2-4. Live round trip
# ---------------------------------------------------------------------------

async def test_round_trip():
    print("\n2. Round trip through VitalAgentRestResourceClient")

    vi = GitHubIssueListInput(operation="list_issues", owner=OWNER, repo=REPO,
                              state="all", max_results=3)
    resp = await _client().handle_tool_request(ToolName.github_issue_tool.value, vi)

    check("response returned", resp is not None and resp.success, str(resp))
    if not resp or not resp.success:
        return None

    out = resp.tool_output
    check("output parsed into GitHubIssueToolOutput",
          isinstance(out, GitHubIssueToolOutput), type(out).__name__)
    if not isinstance(out, GitHubIssueToolOutput):
        return None

    check("no api_error on a valid request", out.api_error is None, str(out.api_error))
    check("repository echoed back", out.repository == f"{OWNER}/{REPO}", str(out.repository))
    check("returned_count matches the payload",
          out.returned_count == len(out.issues),
          f"returned_count={out.returned_count} len={len(out.issues)}")

    print("\n3. Pagination fields survive the wrapper")
    check("truncated is present", isinstance(out.truncated, bool), str(out.truncated))
    check("truncated implies next_page",
          (out.next_page is not None) if out.truncated else True,
          f"truncated={out.truncated} next_page={out.next_page}")
    check("list_issues sets no total_count", out.total_count is None, str(out.total_count))
    print(f"     returned={out.returned_count} truncated={out.truncated} "
          f"next_page={out.next_page} rate_limit={out.rate_limit_remaining}")
    for issue in out.issues:
        print(f"     #{issue.number} [{issue.state}] {issue.title[:48]}")

    return out.issues[0].number if out.issues else None


async def test_error_contract():
    """Plan section 4: expected failures arrive as HTTP 200 with api_error set,
    and the text is actionable. Flattening it to 'no results' wastes the design."""
    print("\n4. Error contract preserved")

    cases = [
        ("repo outside the allowlist",
         GitHubIssueListInput(operation="list_issues", owner="some-other-org", repo="not-allowed"),
         "not in the allowed"),
        ("scope-widening search qualifier",
         GitHubIssueSearchInput(operation="search_issues", owner=OWNER, repo=REPO,
                                query="is:issue org:elsewhere"),
         "qualifier"),
        ("issue that does not exist",
         GitHubIssueGetInput(operation="get_issue", owner=OWNER, repo=REPO, issue_number=999999),
         "404"),
    ]

    for label, vi, expected in cases:
        resp = await _client().handle_tool_request(ToolName.github_issue_tool.value, vi)
        out = resp.tool_output if resp else None
        err = getattr(out, "api_error", None) or ""
        check(f"{label}: reported as api_error, not a transport failure",
              resp is not None and resp.success and bool(err), str(resp)[:120])
        check(f"{label}: message is actionable",
              expected.lower() in err.lower(), err[:140] or "<empty>")


async def test_get_issue(number):
    print("\n5. Single-issue read")
    if number is None:
        check("a known issue number was available", False, "list returned nothing to read")
        return
    vi = GitHubIssueGetInput(operation="get_issue", owner=OWNER, repo=REPO, issue_number=number)
    resp = await _client().handle_tool_request(ToolName.github_issue_tool.value, vi)
    out = resp.tool_output if resp else None
    check("get_issue returns a single issue", getattr(out, "issue", None) is not None,
          str(getattr(out, "api_error", None)))
    if getattr(out, "issue", None):
        i = out.issue
        check("issue number matches the request", i.number == number, f"{i.number} != {number}")
        check("body_truncated flag is present", isinstance(i.body_truncated, bool), str(i.body_truncated))
        print(f"     #{i.number} [{i.state}] {i.title[:50]}  by {i.user}")


async def main():
    print("GitHub tools -- phase 0 spike")
    print("=" * 62)
    print(f"endpoint: {ENDPOINT}   repo: {OWNER}/{REPO}   jwt: {'yes' if JWT else 'NO'}")

    test_union()

    if not JWT:
        print("\nVITAL_TOOL_JWT unset -- skipping the live sections. The service "
              "requires a bearer token and will answer 401.")
    else:
        number = await test_round_trip()
        await test_error_contract()
        await test_get_issue(number)

    print("\n" + "=" * 62)
    print(f"Passed: {len(PASSED)}   Failed: {len(FAILED)}")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
