#!/usr/bin/env python3
"""
KGraphPlanner Agent Test Runner.

Runs the KGraphPlannerAgent classification + continuity test case.

Usage:
    python test_kgraphplanner_agent_runner.py
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import asyncio
import logging

from test_scripts.cases.test_result import TestResult, run_case
from test_scripts.cases import case_kgraphplanner_agent

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

CASE_NAME = "KGraphPlanner Agent"


async def run_all() -> TestResult:
    result = await run_case(CASE_NAME, case_kgraphplanner_agent.run)
    return result


def print_summary(result: TestResult):
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {'1 passed' if result.passed else '1 failed'}, {result.duration_seconds:.1f}s total")
    print(f"{'=' * 60}")
    print(f"  {result.summary_line()}")
    if result.details:
        for k, v in result.details.items():
            print(f"    {k}: {v}")
    if result.traceback and not result.passed:
        print(result.traceback)
    print(f"{'=' * 60}")


def main():
    print(f"\n{'=' * 60}")
    print(f"CASE: {CASE_NAME}")
    print(f"{'=' * 60}")

    result = asyncio.run(run_all())
    print_summary(result)


if __name__ == "__main__":
    main()
