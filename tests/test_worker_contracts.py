"""
Contracts for workers running inside KGraphExecGraphAgent.

The bugs:
  - LLM failures became successes: "Chat error: ..." stored as the result text.
  - parse_json_response could overwrite result_text; parse failures were silent.
  - Every chat worker replayed the whole conversation, including earlier steps.
  - Request content was logged and streamed; args were copied into results.
  - Token usage was logged but not kept.
  - Directive keys were popped from the activation held in state.

What must hold now:
  - failures raise typed errors; on_error="record" records them and routes;
  - per-node timeout and retry, capped by the run deadline;
  - structured and function workers;
  - usage in state; no request content in logs or events by default.
"""
import asyncio
import logging
import time
import warnings

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.agent.node_runtime import failure_kinds
from kgraphplanner.graph.exec_graph import (
    Binding, EdgeSpec, EndNodeSpec, GraphSpec, RetrySpec, StartNodeSpec, WorkerNodeSpec,
)
from kgraphplanner.testing import Delay, ScriptedLLM, assert_node_order
from kgraphplanner.worker.kgraph_case_worker import KGCase, KGraphCaseWorker
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_function_worker import KGraphFunctionWorker
from kgraphplanner.worker.kgraph_structured_worker import KGraphStructuredWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.worker.llm_usage import extract_usage
from kgraphplanner.worker.worker_errors import (
    WorkerError, WorkerLLMError, WorkerParseError, WorkerTimeoutError,
)

SENTINEL = "PII-SENTINEL-4242"


def b(node, path="$"):
    return [Binding(from_node=node, path=path)]


def single(worker, *, node_kwargs=None, edges_after=None, extra=None, start=None, **agent_kwargs):
    """start -> work [-> ...]. ``extra`` adds (worker, node_kwargs) pairs."""
    nodes = [StartNodeSpec(initial_data={"args": start or {"q": "hello"}}), EndNodeSpec(),
             WorkerNodeSpec(id="work", worker_name=worker.name, **(node_kwargs or {}))]
    workers = {worker.name: worker}
    for w, kw in (extra or []):
        nodes.append(WorkerNodeSpec(id=w.name, worker_name=w.name, **kw))
        workers[w.name] = w
    edges = [EdgeSpec(source="start", destination="work", bindings={"q": b("start", "$.q")})]
    edges += edges_after if edges_after is not None else [EdgeSpec(source="work", destination="end")]
    spec = GraphSpec(graph_id="s", nodes=nodes, edges=edges)
    return KGraphExecGraphAgent(name="s", graph_spec=spec, worker_registry=workers, **agent_kwargs)


class Status503Error(Exception):
    status_code = 503


class Status401Error(Exception):
    status_code = 401


class RateLimitError(Exception):
    """Named like the OpenAI/Anthropic SDK class."""


def flaky(*failures):
    """Async fn raising each failure in turn, then succeeding. Counts calls."""
    calls = []

    async def fn(args):
        calls.append(time.monotonic())
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return {"result_text": "ok", "calls": len(calls)}

    fn.calls = calls
    return fn


# ---------------------------------------------------------------------------
# W1: errors stop or reroute the graph
# ---------------------------------------------------------------------------

ERROR_ROUTES = [
    EdgeSpec(source="work", destination="fallback", condition="on_error"),
    EdgeSpec(source="work", destination="end", condition="__default__"),
]


class TestOnError:
    async def test_llm_error_propagates_by_default(self):
        llm = ScriptedLLM(default=[RuntimeError("provider down")])
        agent = single(KGraphChatWorker(name="chat", llm=llm))
        with pytest.raises(WorkerLLMError) as exc:
            await agent.run({})
        assert exc.value.error_type == "llm_error"
        assert isinstance(exc.value.cause, RuntimeError)

    async def test_record_writes_typed_error_and_routes(self):
        llm = ScriptedLLM(default=[RuntimeError("provider down")])
        agent = single(
            KGraphChatWorker(name="chat", llm=llm),
            node_kwargs={"on_error": "record"},
            edges_after=ERROR_ROUTES,
            extra=[(KGraphFunctionWorker(name="fallback", fn=lambda a: {"result_text": "sorry"}), {})],
        )
        result = await agent.run({})
        assert result.errors["work"]["type"] == "llm_error"
        assert result.errors["work"]["exception"] == "WorkerLLMError"
        assert result.results["work"]["status"] == "error"
        assert result.results["work"]["result_text"] == ""
        assert_node_order(result, ["work", "fallback"])

    async def test_eq_status_error_route_also_works(self):
        agent = single(
            KGraphFunctionWorker(name="fn", fn=flaky(ValueError("bad"))),
            node_kwargs={"on_error": "record"},
            edges_after=[
                EdgeSpec(source="work", destination="fallback", condition="eq:status:error"),
                EdgeSpec(source="work", destination="end", condition="__default__"),
            ],
            extra=[(KGraphFunctionWorker(name="fallback", fn=lambda a: {}), {})],
        )
        result = await agent.run({})
        assert_node_order(result, ["work", "fallback"])
        assert result.errors["work"]["type"] == "worker_error"

    async def test_record_without_error_route_warns(self, caplog):
        caplog.set_level(logging.WARNING)
        single(KGraphFunctionWorker(name="fn", fn=lambda a: {}), node_kwargs={"on_error": "record"})
        assert "routes on the error" in caplog.text

    async def test_record_in_multi_node_tool_worker_skips_finalize(self):
        llm = ScriptedLLM(default=[RuntimeError("down")])
        agent = single(
            KGraphToolWorker(name="tw", llm=llm, tool_manager=FakeToolManager(),
                             available_tool_ids=["lookup"], summarize_results=False),
            node_kwargs={"on_error": "record"},
            edges_after=ERROR_ROUTES,
            extra=[(KGraphFunctionWorker(name="fallback", fn=lambda a: {}), {})],
        )
        result = await agent.run({})
        assert result.results["work"]["status"] == "error"
        assert result.errors["work"]["type"] == "llm_error"
        assert_node_order(result, ["work", "fallback"])

    async def test_case_worker_without_input_raises(self):
        llm = ScriptedLLM(default=["a"])
        agent = single(KGraphCaseWorker(name="case", llm=llm, cases=[KGCase("a", "A", "a")]),
                       start={"q": ""})
        with pytest.raises(WorkerError, match="No input"):
            await agent.run({})

    async def test_interrupt_passes_through_record_and_retry(self):
        def asks(args):
            return {"reply": interrupt({"question": "id?"})}

        agent = single(
            KGraphFunctionWorker(name="asker", fn=asks),
            node_kwargs={"on_error": "record", "retry": RetrySpec(max_attempts=3, initial_interval=0)},
            edges_after=ERROR_ROUTES,
            extra=[(KGraphFunctionWorker(name="fallback", fn=lambda a: {}), {})],
            checkpointer=MemorySaver(),
        )
        cfg = {"configurable": {"thread_id": "i1"}}
        first = await agent.arun(config=cfg)
        assert "__interrupt__" in first
        final = await agent.get_compiled_graph().ainvoke(Command(resume="42"), cfg)
        assert final["agent_data"]["results"]["work"] == {"reply": "42"}
        assert final["agent_data"]["errors"] == {}


# ---------------------------------------------------------------------------
# W2: timeout and retry
# ---------------------------------------------------------------------------

class TestTimeoutAndRetry:
    @pytest.mark.timing
    async def test_node_timeout(self):
        async def slow(args):
            await asyncio.sleep(2)
            return {}

        agent = single(KGraphFunctionWorker(name="slow", fn=slow), node_kwargs={"timeout_seconds": 0.1})
        t0 = time.monotonic()
        with pytest.raises(WorkerTimeoutError):
            await agent.run({})
        assert time.monotonic() - t0 < 1

    @pytest.mark.timing
    async def test_slow_llm_call_times_out(self):
        llm = ScriptedLLM(default=[Delay(2, "late")])
        agent = single(KGraphChatWorker(name="chat", llm=llm), node_kwargs={"timeout_seconds": 0.1})
        t0 = time.monotonic()
        with pytest.raises(WorkerTimeoutError):
            await agent.run({})
        assert time.monotonic() - t0 < 1

    @pytest.mark.timing
    async def test_timeout_capped_by_run_deadline(self):
        async def slow(args):
            await asyncio.sleep(2)
            return {}

        agent = single(KGraphFunctionWorker(name="slow", fn=slow), node_kwargs={"timeout_seconds": 5})
        t0 = time.monotonic()
        with pytest.raises(WorkerTimeoutError):
            await agent.arun(deadline=time.time() + 0.2, grace=0.1)
        assert time.monotonic() - t0 < 1

    async def test_deadline_passed_means_no_call(self):
        fn = flaky()
        agent = single(KGraphFunctionWorker(name="fn", fn=fn))
        with pytest.raises(WorkerTimeoutError, match="deadline"):
            await agent.arun(deadline=time.time() - 1)
        assert fn.calls == []

    async def test_retry_once_after_interval(self):
        fn = flaky(RateLimitError("429"))
        agent = single(KGraphFunctionWorker(name="fn", fn=fn), node_kwargs={
            "retry": RetrySpec(max_attempts=2, initial_interval=0.05, jitter=False)})
        result = await agent.run({})
        assert result.results["work"]["calls"] == 2
        assert fn.calls[1] - fn.calls[0] >= 0.05

    async def test_timeout_is_retried_with_a_fresh_budget(self):
        calls = []

        async def first_slow(args):
            calls.append(1)
            if len(calls) == 1:
                await asyncio.sleep(2)
            return {"calls": len(calls)}

        agent = single(KGraphFunctionWorker(name="fn", fn=first_slow), node_kwargs={
            "timeout_seconds": 0.1, "retry": RetrySpec(max_attempts=2, initial_interval=0)})
        result = await agent.run({})
        assert result.results["work"]["calls"] == 2

    @pytest.mark.parametrize("exc, retried", [
        (Status503Error("503"), True),
        (RateLimitError("429"), True),
        (ConnectionError("reset"), True),
        (Status401Error("401"), False),
        (ValueError("bug"), False),
    ])
    async def test_what_is_retried(self, exc, retried):
        fn = flaky(exc)
        agent = single(KGraphFunctionWorker(name="fn", fn=fn), node_kwargs={
            "retry": RetrySpec(max_attempts=2, initial_interval=0)})
        if retried:
            await agent.run({})
        else:
            with pytest.raises(type(exc)):
                await agent.run({})
        assert len(fn.calls) == (2 if retried else 1)

    async def test_parse_error_not_retried(self):
        class Out(BaseModel):
            text: str

        llm = ScriptedLLM(default=[{"wrong": 1}, {"text": "ok"}])
        agent = single(KGraphStructuredWorker(name="st", llm=llm, schema=Out), node_kwargs={
            "retry": RetrySpec(max_attempts=2, initial_interval=0)})
        with pytest.raises(WorkerParseError):
            await agent.run({})
        assert len(llm.calls) == 1

    @pytest.mark.timing
    async def test_backoff_past_deadline_is_skipped(self):
        fn = flaky(Status503Error("503"), Status503Error("503"))
        agent = single(KGraphFunctionWorker(name="fn", fn=fn), node_kwargs={
            "retry": RetrySpec(max_attempts=3, initial_interval=5, jitter=False)})
        t0 = time.monotonic()
        with pytest.raises(Status503Error):
            await agent.arun(deadline=time.time() + 1)
        assert len(fn.calls) == 1 and time.monotonic() - t0 < 1

    async def test_record_after_retries_keeps_attempts(self):
        fn = flaky(*[Status503Error("503")] * 3)
        agent = single(
            KGraphFunctionWorker(name="fn", fn=fn),
            node_kwargs={"on_error": "record",
                         "retry": RetrySpec(max_attempts=3, initial_interval=0)},
            edges_after=ERROR_ROUTES,
            extra=[(KGraphFunctionWorker(name="fallback", fn=lambda a: {}), {})],
        )
        result = await agent.run({})
        assert result.errors["work"] == {
            "type": "worker_error", "message": "503", "exception": "Status503Error", "attempts": 3,
        }

    def test_failure_kinds(self):
        assert failure_kinds(WorkerTimeoutError("n", "x")) == {"timeout"}
        wrapped = WorkerLLMError("n", "x", cause=Status503Error())
        assert failure_kinds(wrapped) == {"llm_error", "server_error"}
        assert "rate_limit" in failure_kinds(RateLimitError())
        assert failure_kinds(Status401Error()) == {"auth_error"}

        class Status404Error(Exception):
            status_code = 404
        assert failure_kinds(Status404Error()) == {"client_error"}


# ---------------------------------------------------------------------------
# W3 / W4: structured and function workers
# ---------------------------------------------------------------------------

class Plan(BaseModel):
    steps: list[str]
    note: str = ""


class TestStructuredWorker:
    async def test_round_trip(self):
        llm = ScriptedLLM(default=[{"steps": ["a", "b"]}])
        agent = single(KGraphStructuredWorker(name="plan", llm=llm, schema=Plan))
        result = await agent.run({})
        assert result.results["work"]["data"] == {"steps": ["a", "b"], "note": ""}
        assert result.results["work"]["usage"]["input_tokens"] == 10
        human = [m for m in llm.calls[0]["messages"] if isinstance(m, HumanMessage)]
        assert "q: hello" in human[0].content

    async def test_conversation_not_duplicated_in_args(self):
        llm = ScriptedLLM(default=[{"steps": []}])
        agent = single(KGraphStructuredWorker(name="plan", llm=llm, schema=Plan,
                                              include_conversation=True))
        agent.graph_spec.edges[0].bindings = {"q": b("start", "$.input"), "extra": b("start", "$.extra")}
        await agent.arun(messages=[HumanMessage(content="typed")], start_data={"extra": "ctx"})
        humans = [m.content for m in llm.calls[0]["messages"] if isinstance(m, HumanMessage)]
        assert humans == ["typed", "extra: ctx"]

    async def test_result_text_field(self):
        llm = ScriptedLLM(default=[{"steps": [], "note": "the note"}])
        agent = single(KGraphStructuredWorker(name="plan", llm=llm, schema=Plan,
                                              result_text_field="note"))
        assert (await agent.run({})).results["work"]["result_text"] == "the note"

    async def test_validation_failure_is_parse_error_without_input_values(self):
        llm = ScriptedLLM(default=[{"steps": SENTINEL}])
        agent = single(KGraphStructuredWorker(name="plan", llm=llm, schema=Plan))
        with pytest.raises(WorkerParseError) as exc:
            await agent.run({})
        assert "steps" in str(exc.value) and SENTINEL not in str(exc.value)

    @staticmethod
    def anthropic_payload(llm, method):
        """The Messages API request body with_structured_output would send."""
        raw = llm.with_structured_output(Plan, method=method, include_raw=True).first.steps__["raw"]
        # langchain-anthropic >=1.7 wraps the binding in a RunnableSequence.
        if not hasattr(raw, "kwargs"):
            raw = next(step for step in raw.steps if hasattr(step, "kwargs"))
        kwargs = {k: v for k, v in raw.kwargs.items() if k != "ls_structured_output_format"}
        return llm._get_request_payload([HumanMessage(content="x")], **kwargs)

    def test_anthropic_uses_native_structured_outputs(self):
        # LangChain's Anthropic default forces a tool call, which current
        # Claude models reject (400). The worker asks for output_config.format.
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-opus-5-5", api_key="test")
        worker = KGraphStructuredWorker(name="plan", llm=llm, schema=Plan)
        assert worker.resolved_method() == "json_schema"

        payload = self.anthropic_payload(llm, worker.resolved_method())
        assert "tool_choice" not in payload and "tools" not in payload
        assert payload["output_config"]["format"]["type"] == "json_schema"

        # The tool-call path, for contrast. Older langchain-anthropic forces
        # the tool (tool_choice); >=1.7 knows this model rejects that and only
        # offers it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tool_path = self.anthropic_payload(llm, "function_calling")
        assert [t["name"] for t in tool_path["tools"]] == ["Plan"]
        assert "output_config" not in tool_path
        assert tool_path.get("tool_choice") in (None, {"type": "tool", "name": "Plan"})

    def test_bound_kwargs_survive_structured_output(self):
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-opus-5-5", api_key="test").bind(max_tokens=321)
        worker = KGraphStructuredWorker(name="plan", llm=llm, schema=Plan)
        assert worker.resolved_method() == "json_schema"
        raw = worker.structured_runnable().first.steps__["raw"]
        kwargs = {k: v for k, v in raw.kwargs.items() if k != "ls_structured_output_format"}
        payload = raw.bound._get_request_payload([HumanMessage(content="x")], **kwargs)
        assert payload["max_tokens"] == 321
        assert "tool_choice" not in payload

    def test_explicit_method_wins_and_other_models_keep_default(self):
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-opus-5-5", api_key="test")
        assert KGraphStructuredWorker(name="p", llm=llm, schema=Plan,
                                      method="function_calling").resolved_method() == "function_calling"
        assert KGraphStructuredWorker(name="p", llm=ScriptedLLM(), schema=Plan).resolved_method() is None

    def test_schema_required(self):
        with pytest.raises(ValueError):
            KGraphStructuredWorker(name="x", llm=ScriptedLLM(), schema=None)


class TestFunctionWorker:
    async def test_sync_and_async_with_context(self):
        async def second(args, ctx):
            return {"seen": ctx.results["work"]["value"], "node": ctx.node_id,
                    "has_time": ctx.run["time_left"] > 0}

        agent = single(
            KGraphFunctionWorker(name="first", fn=lambda a: {"value": a["q"].upper()}),
            edges_after=[EdgeSpec(source="work", destination="second"),
                         EdgeSpec(source="second", destination="end")],
            extra=[(KGraphFunctionWorker(name="second", fn=second), {})],
        )
        result = await agent.run({}, timeout=10)
        assert result.results["second"] == {"seen": "HELLO", "node": "second", "has_time": True}

    async def test_function_cannot_change_state_through_args(self):
        def mutate(args):
            args["q"] = "changed"
            return {}

        agent = single(KGraphFunctionWorker(name="fn", fn=mutate))
        result = await agent.run({})
        assert result.state["agent_data"]["activation"]["work"]["args"]["q"] == "hello"

    @pytest.mark.parametrize("out, match", [
        ("text", "expected a dict"),
        ({"status": "ok"}, "reserved key"),
    ])
    async def test_bad_return_values(self, out, match):
        agent = single(KGraphFunctionWorker(name="fn", fn=lambda a: out))
        with pytest.raises(WorkerError, match=match):
            await agent.run({})

    async def test_reserved_keys_can_be_allowed(self):
        agent = single(KGraphFunctionWorker(name="fn", fn=lambda a: {"status": "ok"},
                                            allow_reserved_keys=True))
        assert (await agent.run({})).results["work"] == {"status": "ok"}

    @pytest.mark.timing
    async def test_sync_function_with_timeout_runs_in_a_thread(self):
        def blocking(args):
            time.sleep(1)
            return {}

        agent = single(KGraphFunctionWorker(name="fn", fn=blocking), node_kwargs={"timeout_seconds": 0.1})
        t0 = time.monotonic()
        with pytest.raises(WorkerTimeoutError):
            await agent.run({})
        assert time.monotonic() - t0 < 0.9

    def test_not_an_llm_worker(self):
        assert KGraphFunctionWorker(name="fn", fn=dict).uses_llm is False


# ---------------------------------------------------------------------------
# W5: usage
# ---------------------------------------------------------------------------

class TestUsage:
    async def test_usage_per_node_log_and_total(self):
        llm = ScriptedLLM(script={"work": ["one", "two"]})
        agent = single(
            KGraphChatWorker(name="chat", llm=llm, include_conversation=False),
            edges_after=[
                EdgeSpec(source="work", destination="work",
                         condition="result.get('result_text') == 'one'"),
                EdgeSpec(source="work", destination="end", condition="__default__"),
            ],
        )
        result = await agent.run({})
        assert sorted(result.usage_log) == ["work#1", "work#2"]
        assert result.usage["calls"] == 2
        assert result.usage["input_tokens"] == 20
        assert result.usage["by_model"]["scripted-model"]["output_tokens"] == 10
        assert result.results["work"]["usage"]["model"] == "scripted-model"

    def test_extract_openai_style(self):
        msg = AIMessage(content="x", response_metadata={
            "model_name": "gpt-x",
            "token_usage": {"prompt_tokens": 7, "completion_tokens": 3,
                            "completion_tokens_details": {"reasoning_tokens": 2},
                            "prompt_tokens_details": {"cached_tokens": 4}},
        })
        u = extract_usage(msg, latency_ms=12.4)
        assert (u["model"], u["input_tokens"], u["output_tokens"], u["reasoning_tokens"],
                u["cache_read_tokens"], u["latency_ms"]) == ("gpt-x", 7, 3, 2, 4, 12)

    def test_extract_anthropic_style(self):
        msg = AIMessage(content="x", response_metadata={"model": "claude-x"}, usage_metadata={
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "input_token_details": {"cache_read": 80, "cache_creation": 5}})
        u = extract_usage(msg, latency_ms=1)
        assert (u["model"], u["input_tokens"], u["cache_read_tokens"],
                u["cache_creation_tokens"]) == ("claude-x", 100, 80, 5)


# ---------------------------------------------------------------------------
# W6: no request content in logs or stream events
# ---------------------------------------------------------------------------

@tool
async def lookup(query: str) -> str:
    """Look up a record."""
    return f"record for {query}"


class _FakeTool:
    def get_tool_function(self):
        return lookup


class FakeToolManager:
    def get_tool(self, tool_id):
        return _FakeTool() if tool_id == "lookup" else None


class TestNoContentInLogs:
    def pii_agent(self):
        llm = ScriptedLLM(script={
            "chat": [f'{{"action": "go", "content": "{SENTINEL}"}}'],
            "case": ["a"],
            "plan": [{"steps": [SENTINEL]}],
            "tools": [
                AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": SENTINEL},
                                                   "id": "c1"}]),
                f"answer with {SENTINEL}",
            ],
        })
        workers = {
            "chat": KGraphChatWorker(name="chat", llm=llm, parse_json_response=True),
            "case": KGraphCaseWorker(name="case", llm=llm, cases=[KGCase("a", "A", "a")]),
            "plan": KGraphStructuredWorker(name="plan", llm=llm, schema=Plan),
            "tools": KGraphToolWorker(name="tools", llm=llm, tool_manager=FakeToolManager(),
                                      available_tool_ids=["lookup"], summarize_results=False),
        }
        spec = GraphSpec(graph_id="pii", nodes=[
            StartNodeSpec(), EndNodeSpec(),
            *[WorkerNodeSpec(id=n, worker_name=n) for n in workers],
        ], edges=[
            EdgeSpec(source="start", destination="chat", bindings={"msg": b("start", "$.msg")}),
            EdgeSpec(source="chat", destination="case", bindings={"query": b("chat", "$.content")}),
            EdgeSpec(source="case", destination="plan", bindings={"input": b("case", "$.input")}),
            EdgeSpec(source="plan", destination="tools", bindings={"request": b("start", "$.msg")}),
            EdgeSpec(source="tools", destination="end"),
        ])
        return KGraphExecGraphAgent(name="pii", graph_spec=spec, worker_registry=workers)

    async def test_sentinel_never_logged_or_streamed(self, caplog):
        caplog.set_level(logging.DEBUG)
        agent = self.pii_agent()
        graph = agent._get_compiled_graph_no_checkpointer()
        events = []
        async for mode, chunk in graph.astream(
                agent._initial_state(None, {"msg": f"my id is {SENTINEL}"}, None),
                stream_mode=["custom", "values"]):
            if mode == "custom":
                events.append(chunk)
            else:
                final = chunk
        assert final["agent_data"]["results"]["tools"]["result_text"] == f"answer with {SENTINEL}"
        ours = [r.getMessage() for r in caplog.records if r.name.startswith("kgraphplanner")]
        assert ours, "expected kgraphplanner log records"
        assert not [m for m in ours if SENTINEL in m]
        assert events and SENTINEL not in repr(events)
        assert all("args_used" not in r for r in final["agent_data"]["results"].values()
                   if isinstance(r, dict))

    async def test_log_content_and_record_args_opt_in(self, caplog):
        caplog.set_level(logging.DEBUG)
        llm = ScriptedLLM(default=["reply"])
        agent = single(KGraphChatWorker(name="chat", llm=llm, log_content=True, record_args=True),
                       start={"q": SENTINEL})
        result = await agent.run({})
        assert SENTINEL in caplog.text
        assert result.results["work"]["args_used"] == {"q": SENTINEL}


# ---------------------------------------------------------------------------
# W7: conversation replay
# ---------------------------------------------------------------------------

class TestIncludeConversation:
    async def run_pipeline(self, include):
        llm = ScriptedLLM(script={"first": ["first reply"], "second": ["second reply"]})
        workers = {n: KGraphChatWorker(name=n, llm=llm, include_conversation=include)
                   for n in ("first", "second")}
        spec = GraphSpec(graph_id="c", nodes=[
            StartNodeSpec(), EndNodeSpec(),
            WorkerNodeSpec(id="first", worker_name="first"),
            WorkerNodeSpec(id="second", worker_name="second"),
        ], edges=[
            EdgeSpec(source="start", destination="first", bindings={"q": b("start", "$.input")}),
            EdgeSpec(source="first", destination="second",
                     bindings={"draft": b("first", "$.result_text")}),
            EdgeSpec(source="second", destination="end"),
        ])
        agent = KGraphExecGraphAgent(name="c", graph_spec=spec, worker_registry=workers)
        state = await agent.arun(messages=[HumanMessage(content="hi")])
        return llm.calls_for("second")[0]["messages"], state["messages"]

    async def test_off_sends_only_own_inputs(self):
        sent, conversation = await self.run_pipeline(False)
        assert not [m for m in sent if isinstance(m, AIMessage)]
        assert "first reply" in sent[-1].content  # arrived through the binding
        assert [type(m).__name__ for m in conversation] == ["HumanMessage"]

    async def test_on_replays_earlier_steps(self):
        sent, conversation = await self.run_pipeline(True)
        assert [m.content for m in sent if isinstance(m, AIMessage)] == ["first reply"]
        assert len(conversation) == 3


# ---------------------------------------------------------------------------
# W8: parse_json_response
# ---------------------------------------------------------------------------

class TestParseJson:
    async def test_reserved_keys_protected_and_data_kept(self, caplog):
        caplog.set_level(logging.WARNING)
        reply = '{"action": "tool", "result_text": "hijack", "tool_request": "weather"}'
        agent = single(KGraphChatWorker(name="chat", llm=ScriptedLLM(default=[reply]),
                                        parse_json_response=True))
        res = (await agent.run({})).results["work"]
        assert res["result_text"] == reply
        assert res["action"] == "tool"
        assert res["data"]["result_text"] == "hijack"
        assert "reserved key(s) ['result_text']" in caplog.text

    async def test_top_level_routing_and_bindings_unchanged(self):
        # A common consumer shape: route on top-level action, bind $.content.
        reply = '{"action": "done", "content": "the message"}'
        agent = single(
            KGraphChatWorker(name="chat", llm=ScriptedLLM(default=[reply]), parse_json_response=True),
            edges_after=[
                EdgeSpec(source="work", destination="out", condition="result.get('action') == 'done'",
                         bindings={"message": b("work", "$.content")}),
                EdgeSpec(source="work", destination="end", condition="__default__"),
            ],
            extra=[(KGraphFunctionWorker(name="out", fn=lambda a: {"got": a["message"]}), {})],
        )
        result = await agent.run({})
        assert result.results["out"]["got"] == "the message"

    async def test_parse_failure_warns_or_raises(self, caplog):
        caplog.set_level(logging.WARNING)
        agent = single(KGraphChatWorker(name="chat", llm=ScriptedLLM(default=["not json"]),
                                        parse_json_response=True))
        assert (await agent.run({})).results["work"]["result_text"] == "not json"
        assert "not a JSON object" in caplog.text

        strict = single(KGraphChatWorker(name="chat", llm=ScriptedLLM(default=["not json"]),
                                         parse_json_response=True, parse_json_required=True))
        with pytest.raises(WorkerParseError):
            await strict.run({})


# ---------------------------------------------------------------------------
# H1 / N2: no in-place changes; per-request system prompts
# ---------------------------------------------------------------------------

class TestNoInPlaceChanges:
    async def test_caller_config_and_start_data_unchanged(self):
        agent = single(KGraphFunctionWorker(name="fn", fn=lambda a: {}))
        config = {"configurable": {}}
        start = {"q": "mine", "nested": {"k": 1}}
        await agent.arun(config=config, start_data=start)
        await agent.run(start)
        assert config == {"configurable": {}}
        assert start == {"q": "mine", "nested": {"k": 1}}

    async def test_directive_override_applies_on_every_loop_pass(self):
        llm = ScriptedLLM(script={"work": ["one", "two"]})
        agent = single(
            KGraphChatWorker(name="chat", llm=llm, system_directive="static", include_conversation=False),
            start={"q": "x", "outcome_prompt": "You explain declines."},
            edges_after=[
                EdgeSpec(source="work", destination="work",
                         condition="result.get('result_text') == 'one'",
                         bindings={"system_directive": b("start", "$.outcome_prompt")}),
                EdgeSpec(source="work", destination="end", condition="__default__"),
            ],
        )
        # The first pass gets the override from the start edge too.
        agent.graph_spec.edges[0].bindings["system_directive"] = b("start", "$.outcome_prompt")
        result = await agent.run({})
        for call in llm.calls_for("work"):
            system = [m.content for m in call["messages"] if isinstance(m, SystemMessage)]
            assert system[0] == "You explain declines."
            assert not any("system_directive" in str(m.content) for m in call["messages"])
        held = result.state["agent_data"]["activation"]["work"]["args"]
        assert held["system_directive"] == "You explain declines."

    async def test_last_human_injection(self):
        fn = KGraphFunctionWorker(name="fn", fn=lambda a: {"input": a.get("q")})
        agent = single(fn, start={"input": "from spec"})
        agent.graph_spec.edges[0].bindings = {"q": b("start", "$.input")}
        state = await agent.arun(messages=[HumanMessage(content="typed")])
        assert state["agent_data"]["results"]["work"]["input"] == "typed"
        assert (await agent.run({})).results["work"]["input"] == "from spec"

        off = single(KGraphFunctionWorker(name="fn", fn=lambda a: {"input": a.get("q")}),
                     start={"input": "from spec"}, inject_last_human_as=None)
        off.graph_spec.edges[0].bindings = {"q": b("start", "$.input")}
        state = await off.arun(messages=[HumanMessage(content="typed")])
        assert state["agent_data"]["results"]["work"]["input"] == "from spec"


# ---------------------------------------------------------------------------
# N3: prompt caching
# ---------------------------------------------------------------------------

class TestSegmentedDirective:
    SEGMENTS = [{"text": "stable rules", "cache": True}, {"text": "per run"}]

    def test_anthropic_gets_cache_control(self):
        from langchain_anthropic import ChatAnthropic
        worker = KGraphChatWorker(name="c", llm=ChatAnthropic(model="claude-x", api_key="test"),
                                  system_directive=self.SEGMENTS)
        [msg] = worker._directive_messages(worker.system_directive)
        assert msg.content == [
            {"type": "text", "text": "stable rules", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "per run"},
        ]

    def test_other_models_get_joined_text(self):
        worker = KGraphChatWorker(name="c", llm=ScriptedLLM(), system_directive=self.SEGMENTS)
        [msg] = worker._directive_messages(worker.system_directive)
        assert msg.content == "stable rules\n\nper run"

    def test_append_adds_an_uncached_segment(self):
        worker = KGraphChatWorker(name="c", llm=ScriptedLLM(), system_directive=self.SEGMENTS)
        directive, args = worker._split_directive({"system_directive_append": "extra", "x": 1})
        assert directive[-1] == {"text": "extra"} and args == {"x": 1}


# ---------------------------------------------------------------------------
# Interrupt orchestration: the reply joins the conversation
# ---------------------------------------------------------------------------

async def test_interrupt_reply_is_added_to_the_conversation():
    # The node runs again from the top on resume, so the question is drafted twice.
    llm = ScriptedLLM(default=["What is your loan ID?"] * 2)
    chat = KGraphChatWorker(name="orchestrator", llm=llm, enable_interrupt=True)
    spec = GraphSpec(graph_id="i", nodes=[
        StartNodeSpec(initial_data={"args": {"lookup": {"found": False, "tried": ["email"]}}}),
        EndNodeSpec(), WorkerNodeSpec(id="ask", worker_name="orchestrator")],
        edges=[EdgeSpec(source="start", destination="ask",
                        bindings={"lookup_result": [Binding(from_node="start", path="$.lookup")]}),
               EdgeSpec(source="ask", destination="end")])
    agent = KGraphExecGraphAgent(name="i", graph_spec=spec, worker_registry={"orchestrator": chat},
                                 checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "reply1"}}
    first = await agent.arun(messages=[HumanMessage(content="what's my balance?")], config=cfg)
    assert "__interrupt__" in first
    final = await agent.get_compiled_graph().ainvoke(Command(resume="LN-98765"), cfg)
    tail = [(type(m).__name__, m.content) for m in final["messages"][-2:]]
    assert tail == [("AIMessage", "What is your loan ID?"), ("HumanMessage", "LN-98765")]
    assert final["agent_data"]["results"]["ask"]["user_provided_info"] == "LN-98765"
