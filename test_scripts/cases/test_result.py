"""Structured test result for test cases."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TestResult:
    """Outcome of a single test case."""
    name: str
    passed: bool = False
    duration_seconds: float = 0.0
    error: Optional[str] = None
    traceback: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def summary_line(self) -> str:
        status = "\u2705" if self.passed else "\u274c"
        timing = f"{self.duration_seconds:.1f}s"
        msg = f"{status} {self.name} ({timing})"
        if self.error:
            msg += f" \u2014 {self.error}"
        return msg


class TestTimer:
    """Context manager that measures elapsed wall-clock time."""

    def __init__(self):
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.start


async def run_case(name: str, coro_fn) -> TestResult:
    """Run an async test case function and capture the result."""
    timer = TestTimer()
    try:
        with timer:
            result = await coro_fn()
        if isinstance(result, TestResult):
            result.duration_seconds = timer.elapsed
            return result
        # If the case returned a plain TestResult-like dict, wrap it
        return TestResult(name=name, passed=True, duration_seconds=timer.elapsed,
                          details=result if isinstance(result, dict) else {})
    except AssertionError as e:
        return TestResult(name=name, passed=False, duration_seconds=timer.elapsed,
                          error=str(e), traceback=traceback.format_exc())
    except Exception as e:
        return TestResult(name=name, passed=False, duration_seconds=timer.elapsed,
                          error=str(e), traceback=traceback.format_exc())
