"""
Typed worker failures.

Built-in workers raise these instead of storing error text as a normal result.
The exec graph agent then applies the node's ``on_error`` policy: ``raise``
propagates the exception out of ``ainvoke``; ``record`` writes a typed entry to
``agent_data.errors`` and an ``{"status": "error", ...}`` result so the graph
can route on it.

All of these subclass ``Exception`` directly, never the builtin ``TimeoutError``
(an ``OSError``), so retry predicates can tell them apart from OS-level errors.
"""
from __future__ import annotations

from typing import Optional


class WorkerError(Exception):
    """Base class for a worker node that could not produce a result."""

    error_type: str = "worker_error"

    def __init__(self, node_id: str, message: str, *, cause: Optional[BaseException] = None):
        super().__init__(f"{node_id}: {message}")
        self.node_id = node_id
        self.message = message
        self.cause = cause
        # Set by the exec graph agent's retry loop on the final attempt.
        self.attempts: int = 1


class WorkerLLMError(WorkerError):
    """The LLM call failed (provider error, network, auth, ...)."""

    error_type = "llm_error"


class WorkerParseError(WorkerError):
    """The LLM answered, but the answer did not parse or validate."""

    error_type = "parse_error"


class WorkerTimeoutError(WorkerError):
    """The node ran past its own budget or the run deadline."""

    error_type = "timeout"


def error_type_of(exc: BaseException) -> str:
    """Classify any exception into the ``errors[node].type`` vocabulary."""
    if isinstance(exc, WorkerError):
        return exc.error_type
    return "worker_error"
