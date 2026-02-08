#!/usr/bin/env python3
"""
Planner Agent Test Runner.

Discovers and runs test cases from the cases/ package.
Each case module exposes an async run() -> TestResult function.

Usage:
    python test_planner_agent.py              # run all cases
    python test_planner_agent.py 1            # run case 1 only
    python test_planner_agent.py 2 3          # run cases 2 and 3
    python test_planner_agent.py simple       # substring match on case name
    python test_planner_agent.py --list       # list available cases
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse
import asyncio
import importlib
import logging

from test_scripts_planner.cases.test_result import TestResult, run_case

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

# Registry: (short_key, module_path, display_name)
CASES = [
    ("simple",      "test_scripts_planner.cases.case_simple_linear",          "Simple Linear Planner"),
    ("pipeline",    "test_scripts_planner.cases.case_full_pipeline",          "Full Pipeline (templates + fan-out/fan-in)"),
    ("full",        "test_scripts_planner.cases.case_planner_full_features",  "LLM Planner Full Features"),
    ("expander",    "test_scripts_planner.cases.case_unit_expander",          "Unit: program_expander"),
    ("hop",         "test_scripts_planner.cases.case_unit_graph_hop",         "Unit: graph_hop"),
    ("tool",        "test_scripts_planner.cases.case_tool_worker",            "Tool Worker Pipeline"),
    ("conditional", "test_scripts_planner.cases.case_conditional_routing",    "Conditional Routing"),
    ("research",    "test_scripts_planner.cases.case_planner_tool_research", "Planner + Tool Research (5 companies)"),
    ("diag",        "test_scripts_planner.cases.case_diagnostic_plan_only",  "Diagnostic: Plan Only"),
]


def resolve_selection(selectors: list[str]) -> list[tuple[str, str, str]]:
    """Resolve CLI selectors to matching cases.  Accepts 1-based indices or substring matches."""
    if not selectors:
        return list(CASES)

    selected = []
    for sel in selectors:
        # Try as 1-based index
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(CASES):
                selected.append(CASES[idx])
                continue
        except ValueError:
            pass
        # Substring match on key or display name (case-insensitive)
        matches = [c for c in CASES if sel.lower() in c[0].lower() or sel.lower() in c[2].lower()]
        if matches:
            selected.extend(matches)
        else:
            print(f"WARNING: no case matches '{sel}'")

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for c in selected:
        if c[0] not in seen:
            seen.add(c[0])
            deduped.append(c)
    return deduped


async def run_cases(cases: list[tuple[str, str, str]]) -> list[TestResult]:
    results: list[TestResult] = []

    for idx, (key, module_path, display_name) in enumerate(cases, 1):
        print(f"\n{'=' * 60}")
        print(f"CASE {idx}/{len(cases)}: {display_name}  [{key}]")
        print(f"{'=' * 60}")

        mod = importlib.import_module(module_path)
        result = await run_case(display_name, mod.run)
        results.append(result)

        print(f"\n{result.summary_line()}")
        if result.traceback and not result.passed:
            print(result.traceback)

    return results


def print_summary(results: list[TestResult]):
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_time = sum(r.duration_seconds for r in results)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed} passed, {failed} failed, {total_time:.1f}s total")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r.summary_line()}")
        if r.details:
            for k, v in r.details.items():
                print(f"    {k}: {v}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Planner Agent Test Runner")
    parser.add_argument("cases", nargs="*",
                        help="Case selectors: 1-based index or name substring (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List available cases and exit")
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for i, (key, _, display) in enumerate(CASES, 1):
            print(f"  {i}. [{key}] {display}")
        return

    cases = resolve_selection(args.cases)
    if not cases:
        print("No cases selected. Use --list to see available cases.")
        return

    results = asyncio.run(run_cases(cases))
    print_summary(results)


if __name__ == "__main__":
    main()
