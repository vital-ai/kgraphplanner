from __future__ import annotations

import copy
import inspect
import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional, Tuple

from langgraph.graph import StateGraph

from kgraphplanner.agent.node_runtime import run_context
from kgraphplanner.worker.kgraph_worker import KGraphWorker
from kgraphplanner.worker.worker_errors import WorkerError

logger = logging.getLogger(__name__)

# Keys the framework writes into results[node]; a function may not return them
# unless allow_reserved_keys is set.
RESERVED_FUNCTION_KEYS = frozenset({"status", "error_type", "error", "usage"})


@dataclass(frozen=True)
class FunctionContext:
    """Read-only context passed as the optional second argument to ``fn``."""
    node_id: str
    prompt: str
    results: Mapping[str, Any]  # every node's result so far (a copy)
    run: Mapping[str, Any]      # started_at, deadline, now, time_left


@dataclass
class KGraphFunctionWorker(KGraphWorker):
    """
    A worker that runs deterministic Python code instead of an LLM.

    ``fn(args) -> dict`` or ``fn(args, ctx) -> dict``, sync or async. ``args``
    is a copy of the node's activation args (the bindings on its incoming
    edge), and ``ctx`` is a ``FunctionContext``. The returned dict becomes
    ``results[node]``, so downstream edges bind its keys with ``$.<key>`` and
    conditions read them as ``result``.

    Exceptions go through the node's ``retry`` and ``on_error`` policy like any
    other worker. A sync ``fn`` runs inline unless the node has
    ``timeout_seconds``, in which case it runs in a thread; the timeout then
    stops waiting for the thread but can't stop it. Long-running work should
    be async.

    Example::

        def validate(args):
            problems = check(args["draft"])
            return {"ok": not problems, "retry": bool(problems), "problems": problems}

        KGraphFunctionWorker(name="validate", fn=validate)
    """

    llm: Any = None
    fn: Optional[Callable[..., Any]] = None
    allow_reserved_keys: bool = False
    _takes_ctx: bool = field(default=False, init=False, repr=False)

    uses_llm: ClassVar[bool] = False

    def __post_init__(self):
        super().__post_init__()
        if not callable(self.fn):
            raise ValueError("KGraphFunctionWorker needs fn=<callable>")
        try:
            params = [
                p for p in inspect.signature(self.fn).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):  # builtins without a signature, e.g. dict
            params = []
        self._takes_ctx = len(params) >= 2

    def _call_args(self, state: Dict[str, Any], occurrence_id: str):
        activation = self._get_activation(state, occurrence_id)
        args = copy.deepcopy(activation.get("args", {}))
        if not self._takes_ctx:
            return (args,)
        results = (state.get("agent_data") or {}).get("results") or {}
        ctx = FunctionContext(
            node_id=occurrence_id,
            prompt=activation.get("prompt", ""),
            results=MappingProxyType(copy.deepcopy(results)),
            run=MappingProxyType(run_context(state)),
        )
        return (args, ctx)

    def _to_update(self, out: Any, occurrence_id: str) -> Dict[str, Any]:
        if not isinstance(out, dict):
            raise WorkerError(
                occurrence_id, f"function returned {type(out).__name__}, expected a dict")
        if not self.allow_reserved_keys:
            bad = sorted(k for k in out if k in RESERVED_FUNCTION_KEYS)
            if bad:
                raise WorkerError(
                    occurrence_id,
                    f"function returned reserved key(s) {bad}; set allow_reserved_keys=True "
                    f"if this is intended")
        return {"agent_data": {"results": {occurrence_id: dict(out)}}}

    def build_subgraph(self, graph_builder: StateGraph, occurrence_id: str) -> Tuple[str, str]:
        node_id = self._safe_node_id(occurrence_id, "fn")

        if inspect.iscoroutinefunction(self.fn):
            async def function_node(state: Dict[str, Any]) -> Dict[str, Any]:
                out = await self.fn(*self._call_args(state, occurrence_id))
                return self._to_update(out, occurrence_id)
        else:
            # Registered as a sync node so the exec graph agent can run it in
            # a thread when the node has a timeout.
            def function_node(state: Dict[str, Any]) -> Dict[str, Any]:
                out = self.fn(*self._call_args(state, occurrence_id))
                if inspect.isawaitable(out):
                    raise WorkerError(
                        occurrence_id,
                        "fn returned an awaitable; define it with 'async def'")
                return self._to_update(out, occurrence_id)

        graph_builder.add_node(node_id, function_node)
        return node_id, node_id
