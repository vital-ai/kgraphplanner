"""
Content-safe log text for tools.

Tool logs follow ``ToolConfig.log_content`` (default off), which reaches each
tool through its config (``AbstractTool.log_content``). With it off, tool logs
carry tool names, lengths, status codes and error classes, but not queries,
identifiers, service error bodies, raw responses or exception text.

This only affects logging. What a tool returns to the model, including the
error text in a failed ``ToolMessage``, is unchanged: the model needs the
reason to recover.
"""
from typing import Any

from kgraphplanner.worker.error_info import describe_error

__all__ = ["describe_error", "logged"]


def logged(value: Any, log_content: bool, *, label: str = "value") -> str:
    """``value`` for a log line: itself with ``log_content``, else its size."""
    if log_content:
        return repr(value)
    size = 0 if value is None else len(str(value))
    return f"<{label}: {size} chars>"
