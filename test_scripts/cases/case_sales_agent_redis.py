"""
Sales Agent — Redis Checkpointer variant.

Runs the same sales-agent test suite as case_sales_agent but forces the
Redis checkpointer (KGraphRedisCheckpointer).  Requires a running Redis
instance.  URL and TTL are read from KGPLAN__CHECKPOINTING__REDIS_URL
and KGPLAN__CHECKPOINTING__REDIS_TTL (defaults: redis://localhost:6381, 3600).
"""
import os

from test_scripts.cases.test_result import TestResult

# Override backend to redis via the standard KGPLAN__ config pattern
os.environ["KGPLAN__CHECKPOINTING__BACKEND"] = "redis"

from test_scripts.cases.case_sales_agent import run as _run_sales_agent


async def run() -> TestResult:
    """Run the sales-agent test with Redis checkpointer."""
    return await _run_sales_agent()
