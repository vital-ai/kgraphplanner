"""
Errors raised by the exec graph agent while building or routing a graph.

``GraphSpecError`` also subclasses ``ValueError`` so callers that already catch
``ValueError`` from spec validation keep working.
"""
from __future__ import annotations

from typing import Any, List, Optional


class ExecGraphError(Exception):
    """Base class for exec graph build and routing failures."""


class GraphSpecError(ExecGraphError, ValueError):
    """The GraphSpec is invalid for KGraphExecGraphAgent. Lists every error."""

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        bullet = "\n  - "
        super().__init__(f"Invalid GraphSpec ({len(errors)} error(s)):{bullet}{bullet.join(errors)}")


class ExecGraphConditionError(ExecGraphError):
    """A routing condition raised while it was evaluated."""

    def __init__(self, source: str, condition: str, cause: BaseException):
        super().__init__(
            f"Condition {condition!r} on edges from '{source}' raised "
            f"{type(cause).__name__}: {cause}"
        )
        self.source = source
        self.condition = condition
        self.cause = cause


class ExecGraphRoutingError(ExecGraphError):
    """No condition matched and the source has no ``__default__`` edge."""

    def __init__(self, source: str, result: Any):
        keys = sorted(result.keys()) if isinstance(result, dict) else type(result).__name__
        super().__init__(
            f"No condition matched for edges from '{source}' and there is no "
            f"__default__ edge (result keys: {keys})"
        )
        self.source = source


class ExecGraphLoopExhausted(ExecGraphError):
    """An edge hit ``max_traversals`` and its ``on_exhausted`` policy is ``error``."""

    def __init__(self, source: str, destination: str, max_traversals: int):
        super().__init__(
            f"Edge '{source}' -> '{destination}' reached max_traversals "
            f"({max_traversals})"
        )
        self.source = source
        self.destination = destination
        self.max_traversals = max_traversals


class ConditionSyntaxError(ExecGraphError, ValueError):
    """A condition string could not be parsed or compiled."""

    def __init__(self, condition: str, reason: str, source: Optional[str] = None):
        where = f" on edge from '{source}'" if source else ""
        super().__init__(f"Invalid condition {condition!r}{where}: {reason}")
        self.condition = condition
        self.reason = reason
