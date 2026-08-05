"""
Eval runner. Owns the `finally`, so teardown happens whatever the outcome.

    python -m github_eval.run --pool read
    python -m github_eval.run --pool all --groups selection,error
    python -m github_eval.run --dry-run          # no model calls, prints the plan

Needs the tool service, a Keycloak JWT, a model API key, and the sandbox repo.
Not part of the pytest suite for exactly that reason.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message=".*NotRequired.*", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from github_eval import agent as agent_mod
from github_eval import cases as cases_mod
from github_eval.fixtures import FixtureFactory
from github_eval.judge import Judge
from github_eval.report import CaseResult, print_case, print_summary, write_results

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_REPO = os.getenv("GITHUB_EVAL_REPO", "vital-ai/vital-ai-sandbox")
DEFAULT_ENDPOINT = os.getenv("VITAL_TOOL_ENDPOINT", "http://localhost:8008")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GitHub tools deep-agent eval")
    p.add_argument("--pool", choices=["read", "write", "code", "all", "minimal"],
                   default="read",
                   help="which tools the agent sees. 'read' needs no teardown; 'write' "
                        "excludes tools that can alter repository contents; 'code' and "
                        "'all' include them; 'minimal' offers only the tools the selected "
                        "cases expect, as the control for the section 2.0.1 comparison")
    p.add_argument("--tool-pool", default="",
                   help="offer this pool's tools while keeping --pool's case selection. "
                        "Separates the two so the same cases can run at different pool "
                        "sizes -- the section 2.0.1 measurement.")
    p.add_argument("--groups", default="",
                   help=f"comma-separated subset of {','.join(cases_mod.GROUPS)}")
    p.add_argument("--case", default="", help="run one case by name")
    # Terra is the balanced tier -- the realistic profile for a production agent
    # doing everyday agentic work, which is what these tools are for.
    p.add_argument("--model", default=os.getenv("GITHUB_EVAL_MODEL", "openai:gpt-5.6-terra"))
    # Sol is the flagship tier, and deliberately a different model from the one
    # under test: a model grading its own output is a weaker check (plan 14.2).
    p.add_argument("--judge-model", default=os.getenv("GITHUB_EVAL_JUDGE", "openai:gpt-5.6-sol"))
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--timeout", type=int, default=agent_mod.DEFAULT_TIMEOUT)
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit; no model or service calls")
    p.add_argument("--out", default="", help="results file (default: logs/github_eval_<run>.json)")
    return p.parse_args(argv)


def choose_cases(args):
    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or None
    # `minimal` is read-only by construction: it offers only the tools the cases
    # name in expect_tools, and fixture-backed cases need write tools the fixtures
    # use but the cases do not name -- so those would fail for a harness reason.
    read_only = args.pool in ("read", "minimal")
    selected = cases_mod.select(
        groups=groups,
        include_writes=not read_only,
        include_code=agent_mod.pool_has_code_writes(args.pool),
    )
    if args.case:
        selected = [c for c in selected if c.name == args.case]
    return selected


def choose_tools(args, selected):
    """The minimal pool is derived from the cases, so it is the honest control:
    same cases, same judge, only the distractor tools removed. If selection is
    equal at 50 tools and at N, pool size is not costing anything (section 2.0.1).
    """
    if args.tool_pool:
        return agent_mod.select_tool_names(args.tool_pool)
    if args.pool != "minimal":
        return agent_mod.select_tool_names(args.pool)
    names = sorted({t for c in selected for t in c.expect_tools})
    if not names:
        raise SystemExit("--pool minimal needs cases that declare expect_tools")
    return names


async def main(argv=None) -> int:
    args = parse_args(argv)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected = choose_cases(args)
    tool_names = choose_tools(args, selected)

    print("=" * 72)
    print("GitHub tools -- deep-agent eval")
    print("=" * 72)
    print(f"  repo={args.repo}  endpoint={args.endpoint}")
    print(f"  pool={args.pool} ({len(tool_names)} tools)  model={args.model}  "
          f"judge={args.judge_model}")
    print(f"  cases={len(selected)}  run_id={run_id}")

    if args.pool in ("read", "minimal"):
        print("  read-only pool: no fixtures created, no teardown needed")
    if args.pool == "minimal":
        print(f"  minimal pool: only the {len(tool_names)} tools these cases expect. "
              f"A case needing an unnamed tool fails as a harness gap, not an agent error.")
    if agent_mod.pool_has_code_writes(args.pool):
        # Stated rather than assumed. This pool can commit to the repository,
        # and the repository is named two lines above -- if that is not the
        # sandbox, this is the line that should stop someone.
        print(f"  CODE WRITES ENABLED: this agent can commit to {args.repo}.")
        print("  Fixture branches are deleted on teardown, taking their commits with them.")

    if args.dry_run:
        print("\n  cases:")
        for c in selected:
            print(f"    {c.group:10s} {c.name:28s} {c.request[:60]}")
        print(f"\n  tools offered:\n    {', '.join(tool_names)}")
        return 0

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("\n  ERROR: no model API key in the environment.")
        return 2

    agent, manager, functions = await agent_mod.build(
        args.repo, args.endpoint, tool_names, args.model
    )
    print(f"  registered {len(functions)} tools\n")

    judge = Judge(args.judge_model)
    factory = FixtureFactory(functions, run_id)

    results = []
    fixtures = []
    teardown_problems = []

    try:
        for i, case in enumerate(selected, 1):
            print(f"[{i}/{len(selected)}] {case.group}/{case.name}")

            context = {}
            fixture_error = None
            if case.fixture:
                fixture = await factory.create(case.fixture)
                fixtures.append(fixture)
                context = fixture.context
                if fixture.errors:
                    fixture_error = "; ".join(fixture.errors)

            request = case.render(context)
            started = time.time()

            if fixture_error:
                # A fixture failure is a harness problem, not a verdict on the
                # agent -- recording it as `incorrect` would be a lie.
                results.append(CaseResult(
                    name=case.name, group=case.group, request=request,
                    verdict="tool_error", reason=f"fixture failed: {fixture_error}",
                    tools_expected=case.expect_tools, elapsed=0.0,
                    error=fixture_error, note=case.note,
                ))
                print_case(results[-1])
                continue

            transcript, run_error = await agent_mod.run_case(agent, request, args.timeout)
            elapsed = time.time() - started

            if run_error:
                results.append(CaseResult(
                    name=case.name, group=case.group, request=request,
                    verdict="tool_error", reason=f"agent run failed: {run_error}",
                    tools_expected=case.expect_tools, elapsed=elapsed,
                    error=run_error, note=case.note,
                ))
                print_case(results[-1])
                continue

            verdict = await judge.judge(transcript)
            results.append(CaseResult(
                name=case.name, group=case.group, request=request,
                verdict=verdict.verdict, reason=verdict.reason,
                tools_called=[c.name for c in transcript.tool_calls],
                tools_expected=case.expect_tools,
                reply=transcript.reply, elapsed=elapsed, note=case.note,
            ))
            print_case(results[-1])

    finally:
        for fixture in fixtures:
            teardown_problems.extend(await factory.teardown(fixture))

    meta = {
        "run_id": run_id, "repo": args.repo, "pool": args.pool,
        "tool_count": len(functions), "tools": sorted(functions),
        "model": args.model, "judge_model": args.judge_model,
        "case_count": len(selected),
    }
    print_summary(results, meta, teardown_problems)

    out = Path(args.out) if args.out else Path("logs") / f"github_eval_{run_id}.json"
    write_results(out, results, meta, teardown_problems)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
