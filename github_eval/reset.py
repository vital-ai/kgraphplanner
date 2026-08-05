"""
Sandbox reset: close ALL open issues and PRs, and delete leftover eval branches.

Blunter than a marker sweep on purpose (plan section 14.4.1) -- it also clears
leftovers from a crashed run that never got marked, and anything an agent created
off-script during a write case.

Run it before a run for a known baseline, and after for courtesy. It does NOT
change how correctness is decided: grading stays judge-driven consistency between
tool results and the agent's reply, never a fixed expected count (section 14.4.2).

    python -m github_eval.reset --confirm
    python -m github_eval.reset                 # dry run, lists what it would close

Guards, because "close every open issue" is exactly the operation you never want
pointed at the wrong repository:

  - hard-pinned to one literal repo, checked here, not the agent's subset and not
    the service allowlist
  - branches are only deleted in two narrow cases: they carry the harness prefix,
    or they are the head branch of a pull request this script just closed. A
    branch that is neither is never touched, however stale it looks -- deleting
    one destroys its unmerged commits, and unlike closing an issue that cannot be
    undone. The default branch is additionally refused by the service.

    The head-branch rule exists because an agent names its own branch during a
    write case, so the prefix cannot catch it; its PR is the only thing tying it
    back to the harness.
  - --confirm required; nothing happens by default and nothing triggers it as a
    side effect of running the eval
  - reports what it closed, so an unexpectedly large number is visible
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager

from github_eval.agent import get_jwt_token
from github_eval.fixtures import BRANCH_PREFIX

# Hard-pinned. Not read from the agent config, not from the service allowlist.
SANDBOX_REPO = "vital-ai/vital-ai-sandbox"

TOOLS = ["github_list_issues", "github_close_issue", "github_list_prs", "github_update_pr",
         "github_list_branches", "github_delete_branch"]


async def build_tools(endpoint: str, repo: str):
    config = AgentConfig.from_dict({"tools": {
        "endpoint": endpoint,
        "enabled": TOOLS,
        "tool_configs": {name: {"repos": [repo]} for name in TOOLS},
    }})
    manager = ToolManager(config=config)
    manager.load_tools_from_config()
    token, error = await get_jwt_token()
    if error:
        print(f"  WARNING: no JWT ({error})")
    else:
        manager.set_jwt_token(token)
    return {t.name: t for t in manager.get_enabled_tool_functions()}


async def call(tools, name, **kwargs):
    return json.loads(await tools[name].ainvoke(kwargs))


async def collect_open(tools, list_tool, key):
    """Page through everything open. next_page may repeat a partly-read page for
    list_issues, so dedupe by number."""
    seen, page, guard = {}, None, 0
    while guard < 20:
        guard += 1
        result = await call(tools, list_tool, state="open", max_results=100,
                            **({"page": page} if page else {}))
        if "error" in result:
            print(f"  ERROR listing: {result['error']}")
            break
        for record in result.get(key, []):
            seen[record["number"]] = record
        if not result.get("truncated") or not result.get("next_page"):
            break
        page = result["next_page"]
    return list(seen.values())


async def collect_eval_branches(tools):
    """Leftover branches this harness created, by prefix.

    Deliberately not "every branch that is not main": a branch is the only thing
    this script can destroy irreversibly, so the sweep is opt-in by naming rather
    than opt-out by exclusion.
    """
    seen, page, guard = {}, None, 0
    while guard < 20:
        guard += 1
        result = await call(tools, "github_list_branches", max_results=100,
                            **({"page": page} if page else {}))
        if "error" in result:
            print(f"  ERROR listing branches: {result['error']}")
            break
        for branch in result.get("branches", []):
            if branch["name"].startswith(BRANCH_PREFIX) and not branch.get("is_default"):
                seen[branch["name"]] = branch
        if not result.get("truncated") or not result.get("next_page"):
            break
        page = result["next_page"]
    return list(seen.values())


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reset the eval sandbox repository")
    parser.add_argument("--confirm", action="store_true",
                        help="actually close things; without it this is a dry run")
    parser.add_argument("--repo", default=SANDBOX_REPO)
    parser.add_argument("--endpoint",
                        default=os.getenv("VITAL_TOOL_ENDPOINT", "http://localhost:8008"))
    args = parser.parse_args(argv)

    if args.repo != SANDBOX_REPO:
        print(f"REFUSED: this script only operates on {SANDBOX_REPO}, not {args.repo!r}.")
        print("Closing every open issue is not something to point at another repository.")
        return 2

    print(f"Sandbox reset -- {args.repo}")
    print(f"  mode: {'CLOSING' if args.confirm else 'dry run (pass --confirm to act)'}\n")

    tools = await build_tools(args.endpoint, args.repo)

    issues = await collect_open(tools, "github_list_issues", "issues")
    prs = await collect_open(tools, "github_list_prs", "pull_requests")
    branches = await collect_eval_branches(tools)

    print(f"  open issues:    {len(issues)}")
    print(f"  open PRs:       {len(prs)}")
    print(f"  eval branches:  {len(branches)} (prefix {BRANCH_PREFIX!r})")

    if not args.confirm:
        for i in issues[:20]:
            print(f"    would close issue #{i['number']} {i['title'][:52]}")
        for p in prs[:20]:
            print(f"    would close PR    #{p['number']} {p['title'][:52]}")
        for b in branches[:20]:
            print(f"    would DELETE branch {b['name']}")
        if len(issues) + len(prs) > 40:
            print("    ...")
        return 0

    closed_issues = failed = 0
    for issue in issues:
        result = await call(tools, "github_close_issue", issue_number=issue["number"],
                            state_reason="not_planned")
        if result.get("closed"):
            closed_issues += 1
        else:
            failed += 1
            print(f"    FAILED issue #{issue['number']}: {result.get('error', 'unknown')}")

    closed_prs = 0
    orphaned_heads = []
    for pr in prs:
        result = await call(tools, "github_update_pr", pr_number=pr["number"], state="closed")
        if result.get("updated"):
            closed_prs += 1
            # An agent names its own branch, so the prefix sweep cannot find it.
            # The PR is the only link back to the harness, which is why this is
            # collected here rather than by scanning branches.
            head = pr.get("head")
            if head and head != "main":
                orphaned_heads.append(head)
        else:
            failed += 1
            print(f"    FAILED PR #{pr['number']}: {result.get('error', 'unknown')}")

    # Branches last: deleting one closes any PR against it, and the PR close
    # above should have been attempted on its own terms first.
    deleted_branches = 0
    for name in orphaned_heads:
        result = await call(tools, "github_delete_branch", branch=name)
        if result.get("deleted"):
            deleted_branches += 1
            print(f"    deleted orphaned head branch {name}")
        else:
            failed += 1
            print(f"    FAILED head branch {name}: {result.get('error', 'unknown')}")

    for branch in branches:
        result = await call(tools, "github_delete_branch", branch=branch["name"])
        if result.get("deleted"):
            deleted_branches += 1
        else:
            failed += 1
            print(f"    FAILED branch {branch['name']}: {result.get('error', 'unknown')}")

    print(f"\n  closed {closed_issues} issues, {closed_prs} PRs, "
          f"deleted {deleted_branches} branches, {failed} failures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
