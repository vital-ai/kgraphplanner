#!/usr/bin/env python3
"""
Planner + Tool Research Test Runner.

Runs the case_planner_tool_research case: LLM planner with a tool-based
research worker (web search), analyst tracks, and aggregator — closely
mirroring the original test_planner.py demo().

Usage:
    python test_planner_tool_research.py
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import asyncio
import logging

from test_scripts_planner.cases.test_result import run_case
from test_scripts_planner.cases.case_planner_tool_research import run

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


async def main():
    print(f"\n{'=' * 60}")
    print("Planner + Tool Research (5 companies)")
    print(f"{'=' * 60}")

    result = await run_case("Planner + Tool Research (5 companies)", run)

    print(f"\n{result.summary_line()}")
    if result.traceback and not result.passed:
        print(result.traceback)

    if result.details:
        print(f"\nDetails:")
        for k, v in result.details.items():
            print(f"  {k}: {v}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
