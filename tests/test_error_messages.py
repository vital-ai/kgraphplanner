"""
Worker error messages follow log_content (0.2.1, K1).

The bug: every WorkerLLMError embedded the provider's error text
(f"LLM call failed: {type(e).__name__}: {e}"). Provider bodies are free text
and can echo the request, and the message is copied into errors[node],
results[node]["error"], RunResult.errors and consumers' logs. That broke
0.2.0's own rule that request content stays out of logs and state unless
log_content=True.

What must hold now:
  - by default the message names the class, HTTP status, failure category,
    provider code and request id, and never the provider's text;
  - log_content=True restores the full text;
  - the original exception is still on .cause, and retry classification is
    unchanged.
"""
import hashlib
import json
import logging

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from kgraphplanner.agent.kgraph_base_agent import AgentState
from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.graph.exec_graph import (
    Binding, EdgeSpec, EndNodeSpec, GraphSpec, RetrySpec, StartNodeSpec, WorkerNodeSpec,
)
from kgraphplanner.testing import ScriptedLLM
from kgraphplanner.worker.error_info import describe_error
from kgraphplanner.worker.kgraph_case_worker import KGCase, KGraphCaseWorker
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_function_worker import KGraphFunctionWorker
from kgraphplanner.worker.kgraph_structured_worker import KGraphStructuredWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.worker.worker_errors import WorkerLLMError

SENTINEL = "SENTINEL-PII-123"


class ProviderError(Exception):
    """Shaped like an OpenAI/Anthropic APIStatusError, with request text echoed."""

    def __init__(self, status=400, code="context_length_exceeded", request_id="req_abc123"):
        super().__init__(f"Your request mentioned {SENTINEL}; it was rejected")
        self.status_code = status
        self.code = code
        self.request_id = request_id


class Plan(BaseModel):
    steps: list[str]


@tool
async def lookup(query: str) -> str:
    """Look up a record."""
    return f"record for {query}"


class _Tools:
    def get_tool(self, tool_id):
        return type("T", (), {"get_tool_function": lambda self: lookup})() if tool_id == "lookup" else None


TOOL_CALL = AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": "x"}, "id": "c1"}])


def single(worker, *, on_error="raise", retry=None, start=None):
    nodes = [StartNodeSpec(initial_data={"args": start or {"query": "hello"}}), EndNodeSpec(),
             WorkerNodeSpec(id="work", worker_name=worker.name, on_error=on_error, retry=retry)]
    edges = [EdgeSpec(source="start", destination="work",
                      bindings={"query": [Binding(from_node="start", path="$.query")]})]
    workers = {worker.name: worker}
    if on_error == "record":
        workers["fallback"] = KGraphFunctionWorker(name="fallback", fn=lambda a: {})
        nodes.append(WorkerNodeSpec(id="fallback", worker_name="fallback"))
        edges += [EdgeSpec(source="work", destination="fallback", condition="on_error"),
                  EdgeSpec(source="work", destination="end", condition="__default__"),
                  EdgeSpec(source="fallback", destination="end")]
    else:
        edges.append(EdgeSpec(source="work", destination="end"))
    return KGraphExecGraphAgent(name="t", graph_spec=GraphSpec(graph_id="t", nodes=nodes, edges=edges),
                                worker_registry=workers)


def workers(llm, **kw):
    """One worker of each kind whose (first) LLM call fails with ProviderError."""
    return {
        "chat": KGraphChatWorker(name="chat", llm=llm, include_conversation=False, **kw),
        "case": KGraphCaseWorker(name="case", llm=llm, cases=[KGCase("a", "A", "a")], **kw),
        "structured": KGraphStructuredWorker(name="structured", llm=llm, schema=Plan, **kw),
        "tool_decision": KGraphToolWorker(name="tool_decision", llm=llm, tool_manager=_Tools(),
                                          available_tool_ids=["lookup"], summarize_results=False, **kw),
    }


def ours(caplog):
    return [r.getMessage() for r in caplog.records if r.name.startswith("kgraphplanner")]


@pytest.mark.parametrize("kind", ["chat", "case", "structured", "tool_decision"])
async def test_message_has_no_provider_text_by_default(kind, caplog):
    caplog.set_level(logging.DEBUG)
    worker = workers(ScriptedLLM(default=[ProviderError()]))[kind]
    with pytest.raises(WorkerLLMError) as exc:
        await single(worker).run({})
    message = str(exc.value)
    assert SENTINEL not in message
    for fact in ("ProviderError", "HTTP 400", "client_error",
                 "code=context_length_exceeded", "request_id=req_abc123"):
        assert fact in message
    assert isinstance(exc.value.cause, ProviderError)          # the original is kept
    assert SENTINEL not in repr(exc.value.run_result.errors)
    assert not [m for m in ours(caplog) if SENTINEL in m]


@pytest.mark.parametrize("kind", ["chat", "case", "structured", "tool_decision"])
async def test_log_content_restores_the_full_text(kind):
    worker = workers(ScriptedLLM(default=[ProviderError()]), log_content=True)[kind]
    with pytest.raises(WorkerLLMError) as exc:
        await single(worker).run({})
    assert SENTINEL in str(exc.value)


async def test_recorded_error_has_no_provider_text():
    agent = single(workers(ScriptedLLM(default=[ProviderError()]))["chat"], on_error="record")
    result = await agent.run({})
    assert result.errors["work"]["type"] == "llm_error"
    assert SENTINEL not in result.errors["work"]["message"]
    assert SENTINEL not in result.results["work"]["error"]
    assert "HTTP 400" in result.errors["work"]["message"]


async def test_tool_worker_max_iterations_summary():
    llm = ScriptedLLM(default=[TOOL_CALL, ProviderError()])
    worker = KGraphToolWorker(name="tw", llm=llm, tool_manager=_Tools(), available_tool_ids=["lookup"],
                              summarize_results=False, max_iters=1)
    with pytest.raises(WorkerLLMError, match="max-iterations") as exc:
        await single(worker).run({})
    assert SENTINEL not in str(exc.value)


async def test_tool_worker_final_answer_from_tool_history():
    # The branch where the worker has tool results but no prompt or args.
    llm = ScriptedLLM(default=[ProviderError()])
    worker = KGraphToolWorker(name="tw", llm=llm, tool_manager=_Tools(), available_tool_ids=["lookup"])
    g = StateGraph(AgentState)
    entry, exit_ = worker.build_subgraph(g, "tw")
    g.add_edge(START, entry)
    g.add_edge(exit_, END)
    key = hashlib.md5(json.dumps({}, sort_keys=True, default=str).encode()).hexdigest()
    state = {
        "messages": [],
        "agent_data": {"activation": {"tw": {"prompt": "", "args": {}}}},
        "work": {"tw": {"_activation_key": key, "iters": 0, "tool_history": [
            [TOOL_CALL, [ToolMessage(content="r", tool_call_id="c1", name="lookup")]]]}},
    }
    with pytest.raises(WorkerLLMError, match="final-answer") as exc:
        await g.compile().ainvoke(state)
    assert SENTINEL not in str(exc.value)


async def test_summarization_failure_log_has_no_provider_text(caplog):
    caplog.set_level(logging.DEBUG)
    llm = ScriptedLLM(default=[TOOL_CALL, ProviderError(), '{"include_summaries": ["c1"]}'])
    worker = KGraphToolWorker(name="tw", llm=llm, tool_manager=_Tools(), available_tool_ids=["lookup"])
    await single(worker).run({})
    warnings = [m for m in ours(caplog) if "Summarization failed" in m]
    assert warnings and SENTINEL not in warnings[0] and "ProviderError" in warnings[0]


async def test_retry_classification_is_unchanged():
    llm = ScriptedLLM(default=[ProviderError(status=429, code="rate_limit_exceeded"), "ok"])
    agent = single(workers(llm)["chat"], retry=RetrySpec(max_attempts=2, initial_interval=0))
    result = await agent.run({})
    assert result.results["work"]["result_text"] == "ok"


def test_prose_codes_and_ids_are_dropped():
    exc = ProviderError(code="something went badly wrong", request_id=f"id {SENTINEL}")
    message = describe_error(exc)
    assert "code=" not in message and "request_id=" not in message
    assert message == "ProviderError (HTTP 400, client_error)"


def test_anthropic_style_body_type_is_used():
    exc = ProviderError(code=None, request_id=None)
    exc.body = {"type": "error", "error": {"type": "overloaded_error", "message": SENTINEL}}
    assert "type=overloaded_error" in describe_error(exc)
    assert SENTINEL not in describe_error(exc)
