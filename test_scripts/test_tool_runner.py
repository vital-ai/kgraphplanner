#!/usr/bin/env python3
"""
Tool Test Runner.

Runs tool-related test cases from the cases/ package.

Usage:
    python test_tool_runner.py              # run all cases
    python test_tool_runner.py 1            # run case 1 only
    python test_tool_runner.py direct       # substring match on case name
    python test_tool_runner.py --list       # list available cases
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse
import asyncio
import importlib
import logging

from test_scripts.cases.test_result import TestResult, run_case

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

# Registry: (short_key, module_path, display_name)
CASES = [
    ("direct",    "test_scripts.cases.case_tool_direct",          "Direct Tool Tests"),
    ("manager",   "test_scripts.cases.case_tool_manager",         "Tool Manager"),
    ("weather",   "test_scripts.cases.case_multi_weather",        "Multi-Weather Agent"),
    ("times_sq",  "test_scripts.cases.case_times_square",          "Times Square"),
]


def resolve_selection(selectors: list[str]) -> list[tuple[str, str, str]]:
    """Resolve CLI selectors to matching cases."""
    if not selectors:
        return list(CASES)

    selected = []
    for sel in selectors:
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(CASES):
                selected.append(CASES[idx])
                continue
        except ValueError:
            pass
        matches = [c for c in CASES if sel.lower() in c[0].lower() or sel.lower() in c[2].lower()]
        if matches:
            selected.extend(matches)
        else:
            print(f"WARNING: no case matches '{sel}'")

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
    parser = argparse.ArgumentParser(description="Tool Test Runner")
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
