"""
Routing conditions for conditional edges.

Conditions are parsed and compiled once, when the graph is built, so a typo
fails at construction instead of silently evaluating to False at runtime.

Supported forms:

- ``__default__``           fallback branch, taken when nothing else matches
- ``true`` / ``always``     always matches
- ``has:<key>``             ``result[key]`` is truthy
- ``eq:<key>:<value>``      ``str(result[key]) == value``
- ``on_error``              the source node recorded an error (``on_error="record"``)
- anything else             a Python expression

``<key>`` may be a dotted path (``data.action``); an exact key match is tried
first so keys that contain dots still work.

Expressions are evaluated with these names in scope:

- ``result``      the source node's result
- ``results``     every node's result, keyed by node id
- ``traversals``  how many times each conditional edge has been taken this run,
                  keyed ``"source->destination"``
- ``run``         ``{"started_at", "deadline", "now", "time_left"}``; ``time_left``
                  is ``inf`` when the run has no deadline

``eval`` with restricted builtins is not a sandbox. Only run expressions from
trusted specs; ``KGraphExecGraphAgent(allow_expressions=False)`` rejects them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from types import CodeType
from typing import Any, Dict, Mapping, Optional

from kgraphplanner.graph.exec_errors import ConditionSyntaxError, ExecGraphConditionError

logger = logging.getLogger(__name__)

DEFAULT = "__default__"
ON_ERROR = "on_error"

_SAFE_BUILTINS = {
    "len": len, "min": min, "max": max, "abs": abs, "any": any, "all": all,
    "int": int, "float": float, "str": str, "bool": bool,
    "True": True, "False": False, "None": None,
}


@dataclass(frozen=True)
class ParsedCondition:
    """A condition string, parsed once at build time."""

    text: str
    kind: str  # "default" | "always" | "has" | "eq" | "on_error" | "expr"
    key: Optional[str] = None
    value: Optional[str] = None
    code: Optional[CodeType] = None

    @property
    def is_default(self) -> bool:
        return self.kind == "default"


def parse_condition(text: str, *, source: Optional[str] = None,
                    allow_expressions: bool = True) -> ParsedCondition:
    """Parse a condition string. Raises ``ConditionSyntaxError`` if invalid."""
    if not isinstance(text, str) or not text.strip():
        raise ConditionSyntaxError(str(text), "empty condition", source)
    t = text.strip()
    if t == DEFAULT:
        return ParsedCondition(text, "default")
    if t in ("true", "always"):
        return ParsedCondition(text, "always")
    if t == ON_ERROR:
        return ParsedCondition(text, "on_error")
    if t.startswith("has:"):
        key = t[4:]
        if not key:
            raise ConditionSyntaxError(text, "has: needs a key", source)
        return ParsedCondition(text, "has", key=key)
    if t.startswith("eq:"):
        parts = t[3:].split(":", 1)
        if len(parts) != 2 or not parts[0]:
            raise ConditionSyntaxError(text, "eq: needs the form eq:<key>:<value>", source)
        return ParsedCondition(text, "eq", key=parts[0], value=parts[1])
    if not allow_expressions:
        raise ConditionSyntaxError(text, "expression conditions are disabled", source)
    try:
        code = compile(t, f"<condition {t!r}>", "eval")
    except SyntaxError as e:
        raise ConditionSyntaxError(text, f"not a valid expression ({e.msg})", source) from e
    return ParsedCondition(text, "expr", code=code)


def _lookup(result: Any, key: str) -> Any:
    if not isinstance(result, dict):
        return None
    if key in result:
        return result[key]
    cur: Any = result
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def evaluate_condition(cond: ParsedCondition, *, source: str,
                       context: Mapping[str, Any]) -> bool:
    """Evaluate a parsed condition.

    ``context`` must contain ``result``; expressions also see ``results``,
    ``traversals`` and ``run``. Expression errors raise
    ``ExecGraphConditionError`` instead of counting as False.
    """
    result = context.get("result")
    if cond.kind == "default":
        return False
    if cond.kind == "always":
        return True
    if cond.kind == "on_error":
        return isinstance(result, dict) and result.get("status") == "error"
    if cond.kind in ("has", "eq"):
        if not isinstance(result, dict):
            logger.warning(
                f"Condition {cond.text!r} on '{source}' got a "
                f"{type(result).__name__} result, not a dict; treating as no match"
            )
            return False
        val = _lookup(result, cond.key)
        if cond.kind == "has":
            return bool(val)
        return str(val if val is not None else "") == cond.value
    scope: Dict[str, Any] = {
        "result": result,
        "results": context.get("results", {}),
        "traversals": context.get("traversals", {}),
        "run": context.get("run", {}),
    }
    try:
        return bool(eval(cond.code, {"__builtins__": _SAFE_BUILTINS}, scope))
    except Exception as e:
        raise ExecGraphConditionError(source, cond.text, e) from e
