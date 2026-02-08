"""
Runner for the planner + restaurant recommendation case.

Usage:
    python test_scripts_planner/test_planner_restaurant.py
"""
from __future__ import annotations

import os
import sys
import asyncio
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from test_scripts_planner.cases.test_result import run_case
from test_scripts_planner.cases.case_planner_restaurant import run

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


async def main():
    result = await run_case("Planner + Restaurant Recommendations", run)
    print("\n" + "=" * 60)
    if result.passed:
        print(f"✅ {result.name} ({result.duration_seconds:.1f}s)")
    else:
        print(f"❌ {result.name} ({result.duration_seconds:.1f}s) — {result.error}")
    print(f"\nDetails:")
    for k, v in (result.details or {}).items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
