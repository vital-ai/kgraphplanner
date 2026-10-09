"""
Token and model usage, normalised across OpenAI and Anthropic responses.

Workers store usage in state, not just in logs:

- ``results[node]["usage"]``: the latest call(s) for that node
- ``agent_data.usage_log["<node>#<n>"]``: every LLM call this run, in order,
  including failed attempts
- ``agent_data.usage_total``: run totals, overall, ``by_model`` and
  ``outcomes``

Every entry has an ``outcome``: ``ok``, or the error type of a failed attempt
(``timeout``, ``llm_error``, ``parse_error``). A call that never returned has
zero token counts; a parse failure did return, so it has real ones.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

USAGE_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens",
                "cache_read_tokens", "cache_creation_tokens", "latency_ms")


def _int(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def model_name(llm: Any) -> Optional[str]:
    inner = getattr(llm, "bound", llm)  # unwrap RunnableBinding
    for attr in ("model_name", "model", "model_id"):
        val = getattr(inner, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


def extract_usage(response: Any, *, latency_ms: float, llm: Any = None) -> Dict[str, Any]:
    """Normalise one LLM response's usage into
    ``{model, input_tokens, output_tokens, reasoning_tokens, cache_read_tokens,
    cache_creation_tokens, latency_ms, outcome}``. Missing counts are 0."""
    um = getattr(response, "usage_metadata", None) or {}
    meta = getattr(response, "response_metadata", None) or {}
    raw = meta.get("token_usage") or meta.get("usage") or {}
    if not isinstance(raw, dict):
        raw = {}
    in_details = um.get("input_token_details") or {}
    out_details = um.get("output_token_details") or {}
    raw_out_details = raw.get("completion_tokens_details") or {}
    raw_in_details = raw.get("prompt_tokens_details") or {}

    return {
        "model": meta.get("model_name") or meta.get("model") or model_name(llm) or "unknown",
        "input_tokens": _int(um.get("input_tokens", raw.get("prompt_tokens", raw.get("input_tokens")))),
        "output_tokens": _int(um.get("output_tokens", raw.get("completion_tokens", raw.get("output_tokens")))),
        "reasoning_tokens": _int(out_details.get("reasoning",
                                                 raw_out_details.get("reasoning_tokens")
                                                 if isinstance(raw_out_details, dict) else 0)),
        "cache_read_tokens": _int(in_details.get("cache_read",
                                                 raw.get("cache_read_input_tokens",
                                                         raw_in_details.get("cached_tokens")
                                                         if isinstance(raw_in_details, dict) else 0))),
        "cache_creation_tokens": _int(in_details.get("cache_creation",
                                                     raw.get("cache_creation_input_tokens"))),
        "latency_ms": int(round(latency_ms)),
        "outcome": "ok",
    }


def failed_attempt_usage(exc: BaseException, *, latency_ms: float,
                         model: Optional[str]) -> Optional[Dict[str, Any]]:
    """Usage entry for a failed attempt, or None if it never reached the model.

    Counted: parse errors (with the reply's real usage, ``exc.usage``), LLM
    call failures, and timeouts that fired during the call. Not counted:
    failures before any call (no input, run deadline already passed).
    """
    from kgraphplanner.worker.worker_errors import (
        WorkerLLMError, WorkerTimeoutError, error_type_of,
    )

    outcome = error_type_of(exc)
    usage = getattr(exc, "usage", None)
    if isinstance(usage, dict) and usage:
        return {**usage, "outcome": outcome}
    reached_model = isinstance(exc, WorkerLLMError) or (
        isinstance(exc, WorkerTimeoutError) and exc.cause is not None)
    if model is None or not reached_model:
        return None
    entry: Dict[str, Any] = {"model": model}
    entry.update({f: 0 for f in USAGE_FIELDS})
    entry["latency_ms"] = int(round(latency_ms))
    entry["outcome"] = outcome
    return entry


def usage_state_update(state: Dict[str, Any], node_id: str,
                       usages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """agent_data partial update that logs ``usages`` and adds them to the run total."""
    from kgraphplanner.agent.kgraph_base_agent import add_values

    if not usages:
        return {}
    log = (state.get("agent_data") or {}).get("usage_log") or {}
    prefix = f"{node_id}#"
    n = sum(1 for k in log if k.startswith(prefix))
    entries = {}
    total: Dict[str, Any] = {"calls": 0, "by_model": {}, "outcomes": {}}
    for u in usages:
        n += 1
        entries[f"{prefix}{n}"] = u
        total["calls"] += 1
        outcome = u.get("outcome", "ok")
        total["outcomes"][outcome] = total["outcomes"].get(outcome, 0) + 1
        by_model = total["by_model"].setdefault(u["model"], {"calls": 0})
        by_model["calls"] += 1
        for f in USAGE_FIELDS:
            total[f] = total.get(f, 0) + u.get(f, 0)
            by_model[f] = by_model.get(f, 0) + u.get(f, 0)
    return {"usage_log": entries, "usage_total": add_values(total)}


def sum_usage(usages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine several calls' usage into one entry (for ``results[node]["usage"]``)."""
    if not usages:
        return {}
    if len(usages) == 1:
        return dict(usages[0])
    out: Dict[str, Any] = {"model": usages[-1]["model"], "calls": len(usages)}
    for f in USAGE_FIELDS:
        out[f] = sum(u.get(f, 0) for u in usages)
    return out
