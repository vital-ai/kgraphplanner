"""
Per-case and summary output, plus a machine-readable results file.

Runs get compared over time rather than only eyeballed -- this is meant to be
evidence about the design decisions in the plan, not a pass/fail smoke test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from github_eval.judge import Transcript, Verdict


@dataclass
class CaseResult:
    name: str
    group: str
    request: str
    verdict: str
    reason: str
    tools_called: List[str] = field(default_factory=list)
    tools_expected: List[str] = field(default_factory=list)
    reply: str = ""
    elapsed: float = 0.0
    error: Optional[str] = None
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "correct"

    @property
    def used_expected_tool(self) -> Optional[bool]:
        """Whether any expected tool was called. None when the case named none.

        Reported, never asserted: an agent reaching the right answer another way
        has not failed. It is a signal about tool selection, not a verdict.
        """
        if not self.tools_expected:
            return None
        return any(t in self.tools_called for t in self.tools_expected)


def print_case(result: CaseResult) -> None:
    symbol = {"correct": "PASS", "incorrect": "FAIL",
              "no_tool_call": "NOTOOL", "tool_error": "TOOLERR"}.get(result.verdict, "?")
    print(f"  [{symbol:7s}] {result.name:28s} {result.elapsed:5.1f}s  "
          f"tools={result.tools_called or '-'}")
    if result.reason:
        print(f"              {result.reason}")
    if result.error:
        print(f"              harness error: {result.error}")


def print_summary(results: List[CaseResult], meta: Dict[str, Any],
                  teardown_problems: List[str]) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    by_verdict: Dict[str, int] = {}
    for r in results:
        by_verdict[r.verdict] = by_verdict.get(r.verdict, 0) + 1

    print(f"\n  model={meta.get('model')}  judge={meta.get('judge_model')}  "
          f"pool={meta.get('pool')} ({meta.get('tool_count')} tools)  repo={meta.get('repo')}")
    print(f"  run_id={meta.get('run_id')}\n")

    for verdict in ("correct", "incorrect", "no_tool_call", "tool_error", "unparseable"):
        if verdict in by_verdict:
            print(f"    {verdict:14s} {by_verdict[verdict]}")

    groups: Dict[str, List[CaseResult]] = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)
    print("\n  by group:")
    for group in sorted(groups):
        rs = groups[group]
        print(f"    {group:12s} {sum(1 for r in rs if r.passed)}/{len(rs)}")

    # Tool selection is the direct evidence about the section 2 decision.
    scored = [r for r in results if r.used_expected_tool is not None]
    if scored:
        hit = sum(1 for r in scored if r.used_expected_tool)
        print(f"\n  expected tool used: {hit}/{len(scored)} "
              f"(reported, not asserted -- another route can still be correct)")
        missed = [r for r in scored if not r.used_expected_tool]
        for r in missed:
            print(f"    {r.name}: expected {r.tools_expected}, called {r.tools_called or '-'}")

    if teardown_problems:
        print(f"\n  TEARDOWN PROBLEMS ({len(teardown_problems)}) -- fixtures may be left open:")
        for problem in teardown_problems:
            print(f"    {problem}")

    total = len(results)
    passed = by_verdict.get("correct", 0)
    print(f"\n  {passed}/{total} correct")


def write_results(path: Path, results: List[CaseResult], meta: Dict[str, Any],
                  teardown_problems: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "teardown_problems": teardown_problems,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  results: {path}")
