"""
Per-node runtime policy for KGraphExecGraphAgent: timeout, retry, on_error.

Workers add their own LangGraph nodes in ``build_subgraph``. The agent hands
them a ``NodePolicyGraph`` proxy instead of the real ``StateGraph``; the proxy
wraps every node the worker adds, so built-in and custom workers get the same
behaviour without changing their code.

Why our own timeout and retry rather than LangGraph's ``add_node(timeout=,
retry_policy=)``:

- LangGraph's timeout is fixed when the graph is built. It can't be capped at
  the time left before a per-run deadline.
- LangGraph's retry backoff doesn't know the deadline, and doesn't expose the
  attempt count, which we record on errors.

The run deadline and its grace window:

- LLM workers get ``min(timeout_seconds, time_left)``. Once the deadline has
  passed they fail without calling the model: the deadline stops model calls.
- Other workers (function nodes) get ``time_left + grace`` instead, so a
  fallback or finish node can still produce a reply after a timeout.

``on_error="record"`` is handled here too, not with LangGraph's
``error_handler``: that handler doesn't run when the failing node is one of
several running in parallel (langgraph 1.2.10 and 1.2.14), so a fan-out with
one failed branch would fail the whole run. After the last attempt the wrapper
returns the error result as a normal state update, and the node's edges carry
the run on to the worker's done node (a pass-through node the agent inserts
after the worker's exit), where the graph's routing for the node hangs. It
also marks the node failed for this activation, so a multi-node worker's
remaining internal nodes do nothing, and clears the worker's scratch state
(``work[node]``, ``decisions[node]``) so its internal routing ends.

Failed attempts are counted in usage: each one that reached the model (an LLM
error, a timeout during the call, a parse error) adds a ``usage_log`` entry
with its ``outcome``. Calls that never returned have zero token counts.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from langgraph.errors import GraphBubbleUp

from kgraphplanner.graph.exec_graph import RetrySpec, WorkerNodeSpec
from kgraphplanner.worker.llm_usage import failed_attempt_usage, usage_state_update
from kgraphplanner.worker.worker_errors import WorkerError, WorkerTimeoutError, error_type_of

logger = logging.getLogger(__name__)

DEFAULT_DEADLINE_GRACE = 1.0


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------

def run_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """The ``run`` dict seen by conditions: started_at, deadline, now, time_left."""
    run = dict((state.get("agent_data") or {}).get("run") or {})
    now = time.time()
    deadline = run.get("deadline")
    run["now"] = now
    run["time_left"] = (deadline - now) if deadline is not None else float("inf")
    return run


def time_left(state: Dict[str, Any], *, include_grace: bool = False) -> Optional[float]:
    """Seconds left before the run deadline (plus its grace window if asked),
    or None if the run has no deadline."""
    run = (state.get("agent_data") or {}).get("run") or {}
    deadline = run.get("deadline")
    if deadline is None:
        return None
    left = deadline - time.time()
    if include_grace:
        left += run.get("grace", DEFAULT_DEADLINE_GRACE)
    return left


# ---------------------------------------------------------------------------
# Retry classification
# ---------------------------------------------------------------------------

def _chain(exc: BaseException):
    """Yield exc and the exceptions it wraps (WorkerError.cause, __cause__)."""
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        nxt = getattr(cur, "cause", None)
        cur = nxt if isinstance(nxt, BaseException) else cur.__cause__


def _class_names(exc: BaseException) -> set:
    return {c.__name__ for c in type(exc).__mro__}


def _status_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def failure_kinds(exc: BaseException) -> set:
    """Classify an exception into the RetrySpec.retry_on vocabulary.

    Duck-typed on class names and status codes so it covers the OpenAI and
    Anthropic SDKs and httpx without importing them.
    """
    kinds = set()
    for e in _chain(exc):
        names = _class_names(e)
        if isinstance(e, WorkerError):
            kinds.add(e.error_type)
        if isinstance(e, (WorkerTimeoutError, asyncio.TimeoutError)) or "APITimeoutError" in names:
            kinds.add("timeout")
        code = _status_code(e)
        if code == 429 or "RateLimitError" in names:
            kinds.add("rate_limit")
        elif code is not None and code >= 500:
            kinds.add("server_error")
        elif code in (401, 403) or names & {"AuthenticationError", "PermissionDeniedError"}:
            kinds.add("auth_error")
        elif code is not None and 400 <= code < 500:
            kinds.add("client_error")
        if names & {"APIConnectionError", "ConnectError", "ConnectTimeout",
                    "ReadError", "RemoteProtocolError", "TransportError"} \
                or isinstance(e, ConnectionError):
            kinds.add("connection")
    return kinds


def is_retryable(exc: BaseException, retry_on: Sequence[str]) -> bool:
    if isinstance(exc, GraphBubbleUp):  # interrupts and other control flow
        return False
    return bool(failure_kinds(exc) & set(retry_on))


def _backoff(spec: RetrySpec, attempt: int) -> float:
    delay = spec.initial_interval * (spec.backoff_factor ** (attempt - 1))
    if spec.jitter:
        delay += random.uniform(0, delay * 0.25)
    return min(delay, spec.max_interval)


# ---------------------------------------------------------------------------
# Node wrapper
# ---------------------------------------------------------------------------

def _merge_failed_usage(result: Any, state: Dict[str, Any], node_id: str,
                        failed: List[Dict[str, Any]]) -> Any:
    """Put failed-attempt usage ahead of the worker's own entries in a
    successful node update."""
    if not isinstance(result, dict):
        logger.warning(f"Node '{node_id}': can't add failed-attempt usage to a "
                       f"{type(result).__name__} update")
        return result
    agent_data = dict(result.get("agent_data") or {})
    own = agent_data.get("usage_log") or {}
    own_entries = [own[k] for k in sorted(own, key=lambda k: int(k.rsplit("#", 1)[1]))]
    agent_data.update(usage_state_update(state, node_id, failed + own_entries))
    return {**result, "agent_data": agent_data}


def is_failed(state: Dict[str, Any], node_id: str) -> bool:
    """True if ``node_id`` recorded a failure in its current activation."""
    return bool(((state.get("agent_data") or {}).get("failed_nodes") or {}).get(node_id))


def record_update(state: Dict[str, Any], node_id: str, exc: BaseException) -> Dict[str, Any]:
    """The state update for ``on_error="record"``: the typed error, an error
    result, the failed attempts' usage, the failed flag, and cleared scratch
    state for the worker."""
    etype = error_type_of(exc)
    attempts = getattr(exc, "attempts", 1)
    logger.warning(f"Node '{node_id}' failed ({etype}, {attempts} attempt(s)); recorded")
    agent_data: Dict[str, Any] = {
        "errors": {node_id: {
            "type": etype,
            "message": str(exc),
            "exception": type(exc).__name__,
            "attempts": attempts,
        }},
        "results": {node_id: {
            "status": "error",
            "error_type": etype,
            "error": str(exc),
            "result_text": "",
            "attempts": attempts,
        }},
        "failed_nodes": {node_id: True},
        "decisions": {node_id: {}},
    }
    agent_data.update(usage_state_update(state, node_id, getattr(exc, "attempt_usage", [])))
    return {"agent_data": agent_data, "work": {node_id: {}}}


def wrap_node(fn: Callable, node_spec: WorkerNodeSpec, lg_node_id: str, *,
              uses_llm: bool = True, model: Optional[str] = None,
              record: bool = False) -> Callable:
    """Wrap one LangGraph node function with timeout + retry for ``node_spec``.

    The node always becomes async. A sync ``fn`` runs inline when no budget
    applies, and in a thread when one does (node timeout or run deadline): the
    timeout then fires on time, but the thread itself can't be stopped.

    ``uses_llm`` selects the deadline rule (see the module docstring);
    ``model`` labels failed attempts in usage (None: don't count them);
    ``record`` turns the final failure into a recorded error result.
    """
    retry = node_spec.retry
    max_attempts = retry.max_attempts if retry else 1
    node_timeout = node_spec.timeout_seconds
    is_async = inspect.iscoroutinefunction(fn)
    params = inspect.signature(fn).parameters

    async def call(state, config, budget):
        kwargs = {"config": config} if "config" in params else {}
        if is_async:
            return await fn(state, **kwargs)
        if budget is not None:
            return await asyncio.to_thread(fn, state, **kwargs)
        return fn(state, **kwargs)

    def fail(exc: BaseException, attempts: int, failed: List[Dict[str, Any]]):
        exc.attempts = attempts  # type: ignore[attr-defined]
        exc.attempt_usage = list(failed)  # type: ignore[attr-defined]
        exc.kg_node_id = node_spec.id  # type: ignore[attr-defined]
        return exc

    async def wrapped(state, config):
        if record and is_failed(state, node_spec.id):
            return {}  # an earlier node of this worker failed: pass through
        try:
            return await attempts_loop(state, config)
        except GraphBubbleUp:
            raise
        except Exception as e:
            if record:
                return record_update(state, node_spec.id, e)
            raise

    async def attempts_loop(state, config):
        failed: List[Dict[str, Any]] = []
        attempt = 0
        while True:
            attempt += 1
            left = time_left(state, include_grace=not uses_llm)
            if left is not None and left <= 0:
                raise fail(WorkerTimeoutError(node_spec.id, "run deadline already passed"),
                           attempt - 1, failed)
            budget = node_timeout if left is None else (
                left if node_timeout is None else min(node_timeout, left))
            t0 = time.monotonic()
            try:
                if budget is None:
                    result = await call(state, config, None)
                else:
                    cm = asyncio.timeout(budget)
                    try:
                        async with cm:
                            result = await call(state, config, budget)
                    except TimeoutError as e:
                        if not cm.expired():  # raised by the node itself, not our budget
                            raise
                        raise WorkerTimeoutError(
                            node_spec.id, f"no result within {budget:.2f}s", cause=e) from e
                return _merge_failed_usage(result, state, node_spec.id, failed) if failed else result
            except GraphBubbleUp:
                raise
            except Exception as e:
                entry = failed_attempt_usage(e, latency_ms=(time.monotonic() - t0) * 1000,
                                             model=model)
                if entry is not None:
                    failed.append(entry)
                if attempt < max_attempts and is_retryable(e, retry.retry_on):
                    delay = _backoff(retry, attempt)
                    left = time_left(state, include_grace=not uses_llm)
                    if left is None or delay < left:
                        logger.warning(
                            f"Node '{node_spec.id}' attempt {attempt}/{max_attempts} "
                            f"failed ({type(e).__name__}); retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue
                    logger.warning(
                        f"Node '{node_spec.id}' attempt {attempt} failed; no retry, "
                        f"backoff {delay:.2f}s would pass the run deadline")
                try:
                    fail(e, attempt, failed)
                except Exception:  # exceptions that refuse new attributes
                    pass
                raise

    wrapped.__name__ = getattr(fn, "__name__", lg_node_id)
    return wrapped


class NodePolicyGraph:
    """StateGraph proxy that applies a node's runtime policy to every node a
    worker adds. Everything other than ``add_node`` is forwarded unchanged."""

    def __init__(self, graph, node_spec: WorkerNodeSpec, done_node: Optional[str], *,
                 uses_llm: bool = True, model: Optional[str] = None):
        self._graph = graph
        self._spec = node_spec
        self._done = done_node
        self._uses_llm = uses_llm
        self._model = model
        self.added: list = []

    def add_node(self, node, action=None, **kwargs):
        if action is None:  # add_node(fn) form: name comes from the function
            action = node
            node = getattr(action, "__name__", str(action))
        if callable(action) and not hasattr(action, "invoke"):
            extra = [p for p in list(inspect.signature(action).parameters)[1:] if p != "config"]
            if extra:
                # LangGraph injects these by parameter name; a wrapper would hide them.
                logger.warning(
                    f"Node '{node}' for '{self._spec.id}' takes {extra}; timeout, "
                    f"retry and run deadline are not applied to it")
            else:
                action = wrap_node(action, self._spec, node, uses_llm=self._uses_llm,
                                   model=self._model, record=self._done is not None)
        self.added.append(node)
        return self._graph.add_node(node, action, **kwargs)

    def __getattr__(self, name):
        return getattr(self._graph, name)
