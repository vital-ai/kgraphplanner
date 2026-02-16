#!/usr/bin/env python3
"""
Agent Test Runner.

Discovers and runs test cases from the cases/ package.
Each case module exposes an async run() -> TestResult function.

Usage:
    python test_agent_runner.py              # run all cases
    python test_agent_runner.py 1            # run case 1 only
    python test_agent_runner.py case         # substring match on case name
    python test_agent_runner.py --list       # list available cases
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

import datetime as _dt

_log_dir = os.path.join(project_root, "test_output")
os.makedirs(_log_dir, exist_ok=True)
_log_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = os.path.join(_log_dir, f"agent_runner_{_log_ts}.log")

_file_handler = logging.FileHandler(_log_file, mode="w")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s - %(levelname)s - %(message)s"))
# Flush after every log record so data survives process kills
_file_handler.stream.reconfigure(write_through=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        _file_handler,
    ],
)
for _name in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Registry: (short_key, module_path, display_name)
CASES = [
    ("case_agent",   "test_scripts.cases.case_case_agent",   "Case Agent Routing"),
    ("case_worker",  "test_scripts.cases.case_case_worker",  "Case Worker"),
    ("sales_agent",  "test_scripts.cases.case_sales_agent",  "Sales Agent Pipeline"),
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
    parser = argparse.ArgumentParser(description="Agent Test Runner")
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
