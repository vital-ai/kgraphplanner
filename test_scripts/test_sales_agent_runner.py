#!/usr/bin/env python3
"""
Sales Agent Test Runner.

Dedicated runner for sales-agent test cases (memory and Redis checkpointer).

Usage:
    python -m test_scripts.test_sales_agent_runner              # run all variants
    python -m test_scripts.test_sales_agent_runner memory       # memory only
    python -m test_scripts.test_sales_agent_runner redis        # redis only
    python -m test_scripts.test_sales_agent_runner --list       # list variants
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse
import asyncio
import datetime as _dt
import importlib
import logging

from test_scripts.cases.test_result import TestResult, run_case

# ---------------------------------------------------------------------------
# Logging — timestamped file + stderr, flush-on-write for crash safety
# ---------------------------------------------------------------------------

_log_dir = os.path.join(project_root, "test_output")
os.makedirs(_log_dir, exist_ok=True)
_log_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = os.path.join(_log_dir, f"sales_agent_runner_{_log_ts}.log")

_file_handler = logging.FileHandler(_log_file, mode="w")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s - %(levelname)s - %(message)s"))
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

# ---------------------------------------------------------------------------
# Case registry: (short_key, module_path, display_name)
# ---------------------------------------------------------------------------

CASES = [
    ("memory",     "test_scripts.cases.case_sales_agent",          "Sales Agent (memory checkpointer)"),
    ("redis",      "test_scripts.cases.case_sales_agent_redis",     "Sales Agent (Redis checkpointer)"),
    ("parse_json", "test_scripts.cases.case_parse_json_response",   "Parse JSON Response (Gap #2)"),
]


def resolve_selection(selectors: list[str]) -> list[tuple[str, str, str]]:
    """Resolve CLI selectors to matching cases."""
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
        # Substring match
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
    parser = argparse.ArgumentParser(description="Sales Agent Test Runner")
    parser.add_argument("cases", nargs="*",
                        help="Case selectors: 'memory', 'redis', or 1-based index (default: all)")
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
