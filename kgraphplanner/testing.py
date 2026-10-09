"""
Offline test helpers for exec graphs.

``ScriptedLLM`` is a fake chat model whose responses are scripted per graph
node, in call order. It supports ``with_structured_output`` (which
``GenericFakeChatModel`` doesn't handle cleanly), ``bind_tools``, scripted
exceptions and delays, and fake usage metadata.

``assert_node_order`` checks the order nodes were activated in a run.

Example::

    llm = ScriptedLLM(script={
        "plan":    [{"steps": ["a", "b"]}],                 # structured output
        "content": ["first draft", "second draft"],         # one per call
        "format":  [RateLimitError(...), {"text": "..."}],  # fail, then succeed
    })
    result = await agent.run({"case_id": "c1"})
    assert_node_order(result, ["context", "plan", "content", "validate",
                               "content", "validate", "format", "finish"])
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, PrivateAttr, ValidationError


@dataclass
class Delay:
    """Scripted response that waits ``seconds`` before producing ``then``."""
    seconds: float
    then: Any = "ok"


def _current_node() -> Optional[str]:
    try:
        from langgraph.config import get_config
        return (get_config().get("metadata") or {}).get("langgraph_node")
    except Exception:
        return None


class ScriptedLLM(BaseChatModel):
    """Fake chat model with responses scripted per exec-graph node.

    ``script`` maps a spec node id to the responses for its calls, in order.
    The LangGraph node a call comes from (e.g. ``content__chat``) is matched to
    the longest script key it equals or starts with followed by ``__``.
    ``default`` serves nodes not in the script; those calls are still logged
    under the spec node id (``content__chat`` → ``content``), and each such
    node keeps its own position in ``default``.

    Cursors and the call log persist across runs; call ``reset()`` between
    runs that should each start from the top of the script.

    A response can be: a str (reply text), a dict or pydantic model (JSON
    text, or the structured result), an ``AIMessage``, an exception instance
    (raised), a ``Delay``, or a callable ``(messages) -> response``.
    """

    script: Dict[str, List[Any]] = {}
    default: List[Any] = []
    model_name: str = "scripted-model"
    input_tokens: int = 10
    output_tokens: int = 5

    _cursor: Dict[str, int] = PrivateAttr(default_factory=dict)
    _calls: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    # --- inspection ---

    @property
    def calls(self) -> List[Dict[str, Any]]:
        """Every call: ``{"node", "messages", "structured"}``."""
        return list(self._calls)

    def calls_for(self, node: str) -> List[Dict[str, Any]]:
        return [c for c in self._calls if c["node"] == node]

    def reset(self) -> None:
        """Start every node from the top of its script and clear the call log."""
        self._cursor.clear()
        self._calls.clear()

    # --- scripting ---

    def _node_key(self) -> str:
        lg = _current_node() or ""
        keys = [k for k in self.script if lg == k or lg.startswith(k + "__")]
        if keys:
            return max(keys, key=len)
        # Worker node names are "<spec node id>__<part>".
        return lg.rsplit("__", 1)[0] if "__" in lg else lg

    async def _next(self, messages: Sequence[BaseMessage], structured: bool) -> Any:
        node = self._node_key()
        self._calls.append({"node": node, "messages": list(messages), "structured": structured})
        queue = self.script.get(node, self.default)
        i = self._cursor.get(node, 0)
        self._cursor[node] = i + 1
        if i >= len(queue):
            raise AssertionError(f"ScriptedLLM: no scripted response #{i + 1} for node '{node}'")
        item = queue[i]
        while True:
            if isinstance(item, Delay):
                await asyncio.sleep(item.seconds)
                item = item.then
            elif callable(item) and not isinstance(item, (BaseModel, type)):
                item = item(list(messages))
            else:
                break
        if isinstance(item, BaseException):
            raise item
        return item

    def _message(self, content: str) -> AIMessage:
        return AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
            response_metadata={"model_name": self.model_name, "finish_reason": "stop"},
        )

    @staticmethod
    def _text(item: Any) -> str:
        if isinstance(item, BaseModel):
            return item.model_dump_json()
        if isinstance(item, (dict, list)):
            return json.dumps(item)
        return str(item)

    # --- BaseChatModel ---

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("ScriptedLLM is async-only; use ainvoke")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        item = await self._next(messages, structured=False)
        msg = item if isinstance(item, AIMessage) else self._message(self._text(item))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        async def run(messages):
            msgs = messages.to_messages() if hasattr(messages, "to_messages") else messages
            item = await self._next(msgs, structured=True)
            raw = self._message(self._text(item))
            parsed, error = None, None
            try:
                if isinstance(item, schema):
                    parsed = item
                elif isinstance(item, str):
                    parsed = schema.model_validate_json(item)
                else:
                    parsed = schema.model_validate(item)
            except ValidationError as e:
                error = e
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": error}
            if error is not None:
                raise error
            return parsed

        return RunnableLambda(run)


def rate_limit_error(message: str = "rate limited") -> Exception:
    """An exception the retry policy classifies as ``rate_limit`` (HTTP 429),
    for scripting a transient provider failure without the provider SDK."""

    class RateLimitError(Exception):
        status_code = 429

    return RateLimitError(message)


def node_trace(run_or_state: Any) -> List[str]:
    """The node activation order from a ``RunResult``, a final state, or a list."""
    if isinstance(run_or_state, list):
        return list(run_or_state)
    trace = getattr(run_or_state, "trace", None)
    if trace is not None:
        return list(trace)
    return list(((run_or_state or {}).get("agent_data") or {}).get("trace") or [])


def assert_node_order(run_or_state: Any, expected: Sequence[str], *, exact: bool = True) -> None:
    """Assert the nodes a run activated, in order.

    ``exact=False`` only requires ``expected`` to appear as a subsequence.
    """
    trace = node_trace(run_or_state)
    if exact:
        assert trace == list(expected), f"node order {trace} != expected {list(expected)}"
        return
    it = iter(trace)
    missing = [n for n in expected if not any(t == n for t in it)]
    assert not missing, f"node order {trace} does not contain {list(expected)} in order"
