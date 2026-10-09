"""
Deterministic gate workers.

A gate routes on the request's own inputs without an LLM call. Bind the
inputs to check, then branch on its result with conditional edges::

    EdgeSpec(source="start", destination="gate",
             bindings={"inbound_message": [Binding(from_node="start", path="$.inbound_message")]}),
    EdgeSpec(source="gate", destination="classifier",
             condition="has:present.inbound_message",
             bindings={"input": [Binding(from_node="gate", path="$.inputs.inbound_message")]}),
    EdgeSpec(source="gate", destination="end", condition="__default__"),

In a hand-written GraphSpec a start-edge guard does the same job without the
extra node. A gate is how KGraphPlannerAgent's generated plans route on the
inputs, since their conditionals can't originate from ``start``.
"""
from typing import Any, Dict, Optional

from kgraphplanner.worker.kgraph_function_worker import KGraphFunctionWorker
from kgraphplanner.worker.kgraph_worker import DIRECTIVE_KEYS, is_present  # noqa: F401 (re-exported)

DEFAULT_GATE_DESCRIPTION = (
    "Deterministic gate, no LLM call. Bind the inputs to check; its result "
    "present.<arg> is true when that input is non-empty, and inputs.<arg> "
    "holds the value. Branch with conditionals on has:present.<arg>."
)


def _presence(args: Dict[str, Any]) -> Dict[str, Any]:
    inputs = {k: v for k, v in args.items() if k not in DIRECTIVE_KEYS}
    present = {k: is_present(v) for k, v in inputs.items()}
    summary = ", ".join(f"{k}={'present' if p else 'empty'}" for k, p in present.items())
    return {"present": present, "inputs": inputs, "result_text": summary or "no inputs"}


def presence_gate(name: str = "gate", *, description: Optional[str] = None) -> KGraphFunctionWorker:
    """A gate whose result says which of its bound inputs are non-empty:
    ``{"present": {arg: bool}, "inputs": {arg: value}, "result_text": ...}``.
    """
    return KGraphFunctionWorker(
        name=name, fn=_presence,
        description=DEFAULT_GATE_DESCRIPTION if description is None else description,
    )
