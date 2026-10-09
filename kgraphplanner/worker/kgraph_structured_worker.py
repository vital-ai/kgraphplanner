from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.runtime import get_runtime
from pydantic import BaseModel, ValidationError

from kgraphplanner.worker.kgraph_worker import KGraphWorker
from kgraphplanner.worker.llm_usage import extract_usage, usage_state_update
from kgraphplanner.worker.error_info import describe_error
from kgraphplanner.worker.worker_errors import WorkerLLMError, WorkerParseError

logger = logging.getLogger(__name__)


def render_request(prompt: str, args: Dict[str, Any]) -> str:
    """The user-turn text for a worker: ``args["request"]`` if present,
    otherwise the prompt followed by each arg as ``key: value`` (JSON for
    lists and dicts)."""
    if "request" in args:
        return str(args["request"])
    parts = [prompt] if prompt else []
    for k, v in args.items():
        if isinstance(v, (dict, list)):
            parts.append(f"{k}: {json.dumps(v, default=str, ensure_ascii=False)}")
        else:
            parts.append(f"{k}: {v}")
    return "\n".join(parts) or "Respond based on the task instructions."


def describe_validation_error(e: BaseException) -> str:
    """Validation failure summary without the input values (which may hold
    request content)."""
    if isinstance(e, ValidationError):
        problems = [
            f"{'.'.join(str(p) for p in err.get('loc', ())) or '<root>'}: {err.get('type')}"
            for err in e.errors(include_input=False, include_url=False)
        ]
        return f"{e.error_count()} validation error(s): {'; '.join(problems)}"
    return type(e).__name__


@dataclass
class KGraphStructuredWorker(KGraphWorker):
    """
    A worker that makes one LLM call and returns a validated pydantic model.

    Stores ``results[node] = {"data": model.model_dump(mode="json"),
    "result_text": <JSON, or the result_text_field value>, "usage": {...}}``.
    Bind fields downstream with ``path="$.data.<field>"``.

    A reply that doesn't validate against ``schema`` raises
    ``WorkerParseError`` (``parse_error``); a failed call raises
    ``WorkerLLMError``. The exec graph agent applies the node's ``on_error``
    and ``retry`` policy.

    Args:
        schema: the pydantic model to return.
        method: passed to ``with_structured_output`` ("function_calling" or
            "json_schema"). None picks per model: "json_schema" for Anthropic
            models, the LangChain default otherwise. LangChain's Anthropic
            default, "function_calling", forces a tool call, which current
            Claude models reject (400) and which isn't honoured with thinking
            on; "json_schema" uses Claude's native structured outputs
            (``output_config.format``) instead.
        result_text_field: use this field's value as ``result_text`` instead
            of the JSON dump.
        include_conversation: also send ``state["messages"]``. Off by default:
            pipeline steps see only their own inputs.
    """

    schema: Optional[Type[BaseModel]] = None
    method: Optional[str] = None
    result_text_field: Optional[str] = None
    include_conversation: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not (isinstance(self.schema, type) and issubclass(self.schema, BaseModel)):
            raise ValueError("KGraphStructuredWorker needs schema=<pydantic BaseModel subclass>")
        if self.result_text_field and self.result_text_field not in self.schema.model_fields:
            raise ValueError(
                f"result_text_field '{self.result_text_field}' is not a field of "
                f"{self.schema.__name__}")

    def resolved_method(self) -> Optional[str]:
        """The ``with_structured_output`` method this worker uses (None: LangChain default)."""
        if self.method:
            return self.method
        inner = getattr(self.llm, "bound", self.llm)  # unwrap RunnableBinding
        if "Anthropic" in type(inner).__name__:
            return "json_schema"
        return None

    def structured_runnable(self):
        """The runnable this worker calls: the LLM with structured output.

        ``llm.bind(max_tokens=...).with_structured_output(...)`` drops the
        bound kwargs (the binding delegates to the bare model), so bound
        kwargs that are model fields are copied onto the model instead.
        """
        llm = self.llm
        inner = getattr(llm, "bound", None)
        bound_kwargs = getattr(llm, "kwargs", None) or {}
        if inner is not None and hasattr(inner, "model_copy"):
            fields = getattr(type(inner), "model_fields", {})
            dropped = sorted(k for k in bound_kwargs if k not in fields)
            if dropped:
                logger.warning(
                    f"Structured worker '{self.name}': bound kwargs {dropped} are not "
                    f"fields of {type(inner).__name__} and are dropped")
            llm = inner.model_copy(update={k: v for k, v in bound_kwargs.items() if k in fields})
        kwargs: Dict[str, Any] = {"include_raw": True}
        method = self.resolved_method()
        if method:
            kwargs["method"] = method
        return llm.with_structured_output(self.schema, **kwargs)

    def build_subgraph(self, graph_builder: StateGraph, occurrence_id: str) -> Tuple[str, str]:
        node_id = self._safe_node_id(occurrence_id, "structured")
        structured = self.structured_runnable()

        async def structured_node(state: Dict[str, Any]) -> Dict[str, Any]:
            writer = get_runtime().stream_writer
            activation = self._get_activation(state, occurrence_id)
            prompt = activation.get("prompt", "")
            directive, args = self._split_directive(activation.get("args", {}))

            messages = self._directive_messages(directive)
            if prompt:
                messages.append(SystemMessage(content=f"Task instructions: {prompt}"))
            conversation = list(state.get("messages", [])) if self.include_conversation else []
            messages.extend(conversation)
            if conversation:
                # The current user message is usually also bound into args
                # (e.g. start.input); don't send it twice.
                last = str(conversation[-1].content)
                args = {k: v for k, v in args.items() if str(v) != last}
            if args or not conversation:
                # The prompt is already the system "Task instructions".
                messages.append(HumanMessage(content=render_request("", args)))

            event: Dict[str, Any] = {"phase": "structured_start", "node": occurrence_id,
                                     "worker": self.name, "arg_keys": sorted(args)}
            if self.stream_payloads:
                event["activation"] = activation
            writer(event)
            if self.log_content:
                logger.debug(f"Structured worker '{occurrence_id}' messages: {messages}")

            t0 = time.time()
            try:
                out = await structured.ainvoke(messages)
            except Exception as e:
                raise WorkerLLMError(occurrence_id, f"LLM call failed: {describe_error(e, log_content=self.log_content)}",
                                     cause=e) from e
            elapsed_ms = (time.time() - t0) * 1000

            raw = out.get("raw") if isinstance(out, dict) else None
            parsed = out.get("parsed") if isinstance(out, dict) else out
            parse_err = out.get("parsing_error") if isinstance(out, dict) else None
            try:
                if parse_err is not None:
                    raise parse_err
                if isinstance(parsed, dict):
                    parsed = self.schema.model_validate(parsed)
                if not isinstance(parsed, self.schema):
                    raise TypeError(f"expected {self.schema.__name__}, got {type(parsed).__name__}")
            except Exception as e:
                detail = str(e) if self.log_content else describe_validation_error(e)
                err = WorkerParseError(
                    occurrence_id, f"reply did not validate against {self.schema.__name__}: {detail}",
                    cause=e)
                # The call returned, so its tokens were spent: count them.
                err.usage = extract_usage(raw, latency_ms=elapsed_ms, llm=self.llm)
                raise err from e

            data = parsed.model_dump(mode="json")
            if self.result_text_field:
                result_text = str(data.get(self.result_text_field) or "")
            else:
                result_text = json.dumps(data, ensure_ascii=False)
            usage = extract_usage(raw, latency_ms=elapsed_ms, llm=self.llm)

            update = self._finalize_result(state, occurrence_id, result_text)
            result = update["agent_data"]["results"][occurrence_id]
            result["data"] = data
            result["usage"] = usage
            update["agent_data"].update(usage_state_update(state, occurrence_id, [usage]))

            event = {"phase": "structured_complete", "node": occurrence_id,
                     "worker": self.name, "fields": sorted(data)}
            if self.stream_payloads:
                event["data"] = data
            writer(event)
            logger.info(f"Structured worker '{occurrence_id}' → {self.schema.__name__} "
                        f"({usage['input_tokens']} in / {usage['output_tokens']} out, "
                        f"{usage['latency_ms']} ms)")
            return update

        graph_builder.add_node(node_id, structured_node)
        return node_id, node_id
