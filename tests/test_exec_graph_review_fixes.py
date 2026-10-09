"""
Regression tests for the consumer review of the exec graph changes
(planning/exec_graph_routing_and_worker_fixes.md, Phase 2, items P1-P15).

The bugs:
  - P1: with a run deadline, run() raised a bare TimeoutError instead of
    recording the error, and an on_error fallback could never run after the
    deadline.
  - P2: list-valued node defaults were mutated in place, so inputs grew across
    loop passes, across runs, and in the shared spec.
  - P3: a sync function node ignored the run deadline.
  - P9: on a reused checkpointed thread, results from an earlier turn were
    visible to the next one.

What must hold now:
  - the deadline stops model calls, and code nodes may still finish in a
    grace window;
  - a failed run still returns what it produced (exc.run_result), with usage
    that counts failed attempts;
  - runs are isolated from each other and from the spec.
"""
import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ValidationError

from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.agent.node_runtime import _backoff
from kgraphplanner.graph.exec_errors import GraphSpecError
from kgraphplanner.graph.exec_graph import (
    Binding, EdgeSpec, EndNodeSpec, GraphSpec, RetrySpec, StartNodeSpec, WorkerNodeSpec,
)
from kgraphplanner.program.program import ConditionalBranch
from kgraphplanner.testing import Delay, ScriptedLLM, assert_node_order, rate_limit_error
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_function_worker import KGraphFunctionWorker
from kgraphplanner.worker.kgraph_structured_worker import KGraphStructuredWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.worker.kgraph_worker import KGraphWorker
from kgraphplanner.worker.worker_errors import (
    WorkerError, WorkerLLMError, WorkerParseError, WorkerTimeoutError,
)


def b(node, path="$"):
    return [Binding(from_node=node, path=path)]


def graph(workers, nodes, edges, **agent_kwargs):
    spec = GraphSpec(graph_id="g", nodes=[StartNodeSpec(initial_data={"args": {"q": "hi"}}),
                                         EndNodeSpec(), *nodes], edges=edges)
    return KGraphExecGraphAgent(name="g", graph_spec=spec,
                                worker_registry={w.name: w for w in workers}, **agent_kwargs)


def chat_then_fallback(llm, fallback, **node_kwargs):
    """start -> chat (record) -> on_error: fallback | default: end."""
    return graph(
        [KGraphChatWorker(name="chat", llm=llm, include_conversation=False), fallback],
        [WorkerNodeSpec(id="chat", worker_name="chat", on_error="record", **node_kwargs),
         WorkerNodeSpec(id="fallback", worker_name=fallback.name)],
        [EdgeSpec(source="start", destination="chat", bindings={"q": b("start", "$.q")}),
         EdgeSpec(source="chat", destination="fallback", condition="on_error"),
         EdgeSpec(source="chat", destination="end", condition="__default__"),
         EdgeSpec(source="fallback", destination="end")],
    )


# ---------------------------------------------------------------------------
# P1: run deadline, grace window and on_error="record"
# ---------------------------------------------------------------------------

class TestDeadlineAndRecord:
    @pytest.mark.timing
    async def test_run_records_timeout_and_function_fallback_runs(self):
        agent = chat_then_fallback(
            ScriptedLLM(default=[Delay(5, "late")]),
            KGraphFunctionWorker(name="fallback", fn=lambda a: {"result_text": "canned reply"}))
        t0 = time.monotonic()
        result = await agent.run({}, deadline=time.time() + 0.3, grace=1.0)
        assert time.monotonic() - t0 < 1.0
        assert result.errors["chat"]["type"] == "timeout"
        assert result.results["fallback"]["result_text"] == "canned reply"
        assert_node_order(result, ["chat", "fallback"])

    async def test_spec_timeout_records_too(self):
        agent = chat_then_fallback(
            ScriptedLLM(default=[Delay(5, "late")]),
            KGraphFunctionWorker(name="fallback", fn=lambda a: {"result_text": "canned"}))
        agent.graph_spec.timeout_seconds = 0.3
        result = await agent.run({})
        assert result.errors["chat"]["type"] == "timeout"
        assert "fallback" in result.results

    @pytest.mark.timing
    async def test_llm_fallback_after_deadline_fails_fast(self):
        llm = ScriptedLLM(script={"chat": [Delay(5, "late")], "fallback": ["never sent"]})
        agent = chat_then_fallback(llm, KGraphChatWorker(name="fallback", llm=llm))
        t0 = time.monotonic()
        with pytest.raises(WorkerTimeoutError, match="deadline already passed") as exc:
            await agent.run({}, deadline=time.time() + 0.3, grace=1.0)
        assert time.monotonic() - t0 < 1.0
        assert llm.calls_for("fallback") == []
        assert exc.value.run_result.errors["chat"]["type"] == "timeout"

    @pytest.mark.timing
    async def test_stuck_function_node_times_out_at_deadline_plus_grace(self):
        # The node's own budget (time left + grace) fires, so the error is a
        # WorkerTimeoutError for the node, not the run backstop.
        async def stuck(args):
            await asyncio.sleep(5)
            return {}

        agent = graph([KGraphFunctionWorker(name="stuck", fn=stuck)],
                      [WorkerNodeSpec(id="stuck", worker_name="stuck")],
                      [EdgeSpec(source="start", destination="stuck"),
                       EdgeSpec(source="stuck", destination="end")])
        t0 = time.monotonic()
        with pytest.raises(WorkerTimeoutError):
            await agent.run({}, timeout=0.2, grace=0.2)
        assert time.monotonic() - t0 < 1.0

    @pytest.mark.timing
    async def test_backstop_cancels_an_unwrapped_node(self):
        # A node function that takes LangGraph-injected parameters (here
        # `writer`) isn't wrapped, so it has no node budget: only run()'s
        # backstop stops it. That raises the builtin TimeoutError.
        @dataclass
        class StuckWorker(KGraphWorker):
            llm: Any = None
            uses_llm: ClassVar[bool] = False

            def build_subgraph(self, graph_builder, occurrence_id):
                async def stuck(state, writer):
                    await asyncio.sleep(5)
                    return {}
                node_id = self._safe_node_id(occurrence_id, "stuck")
                graph_builder.add_node(node_id, stuck)
                return node_id, node_id

        agent = graph(
            [KGraphFunctionWorker(name="ok", fn=lambda a: {"done": True}), StuckWorker(name="stuck")],
            [WorkerNodeSpec(id="ok", worker_name="ok"), WorkerNodeSpec(id="stuck", worker_name="stuck")],
            [EdgeSpec(source="start", destination="ok"), EdgeSpec(source="ok", destination="stuck"),
             EdgeSpec(source="stuck", destination="end")])
        t0 = time.monotonic()
        with pytest.raises(TimeoutError) as exc:
            await agent.run({}, timeout=0.2, grace=0.1)
        assert not isinstance(exc.value, WorkerTimeoutError)
        assert time.monotonic() - t0 < 1.0
        partial = exc.value.run_result
        assert partial.results["ok"] == {"done": True}
        assert partial.errors["__run__"]["type"] == "timeout"


# ---------------------------------------------------------------------------
# P2: list defaults are never mutated
# ---------------------------------------------------------------------------

class TestListDefaults:
    def agent(self, seen):
        def content(args):
            seen.append(list(args["items"]))
            return {}

        return graph(
            [KGraphFunctionWorker(name="content", fn=content),
             KGraphFunctionWorker(name="validate", fn=lambda a: {"retry": True})],
            [WorkerNodeSpec(id="content", worker_name="content", reentry="reset",
                            defaults={"args": {"items": []}}),
             WorkerNodeSpec(id="validate", worker_name="validate")],
            [EdgeSpec(source="start", destination="content",
                      bindings={"items": [Binding(from_node="start", path="$.q", reduce="append_list")]}),
             EdgeSpec(source="content", destination="validate"),
             EdgeSpec(source="validate", destination="content", condition="has:retry",
                      max_traversals=2, on_exhausted="default",
                      bindings={"items": [Binding(literal=1, reduce="append_list")]}),
             EdgeSpec(source="validate", destination="end", condition="__default__")],
        )

    async def test_each_pass_and_each_run_starts_clean(self):
        seen = []
        agent = self.agent(seen)
        await agent.run({})
        await agent.run({})
        assert seen == [["hi"], [1], [1]] * 2
        assert agent.graph_spec.nodes[2].defaults == {"args": {"items": []}}

    async def test_concurrent_runs_dont_share_lists(self):
        seen = []
        agent = self.agent(seen)
        await asyncio.gather(*(agent.run({}) for _ in range(10)))
        assert Counter(map(tuple, seen)) == Counter({("hi",): 10, (1,): 20})

    async def test_bound_results_are_copies(self):
        def mutate(args):
            args["items"].append("x")
            return {}

        agent = graph(
            [KGraphFunctionWorker(name="src", fn=lambda a: {"items": [1]}),
             KGraphFunctionWorker(name="dst", fn=mutate)],
            [WorkerNodeSpec(id="src", worker_name="src"), WorkerNodeSpec(id="dst", worker_name="dst")],
            [EdgeSpec(source="start", destination="src"),
             EdgeSpec(source="src", destination="dst", bindings={"items": b("src", "$.items")}),
             EdgeSpec(source="dst", destination="end")])
        result = await agent.run({})
        assert result.results["src"]["items"] == [1]


# ---------------------------------------------------------------------------
# P3: sync function nodes respect the run deadline
# ---------------------------------------------------------------------------

@pytest.mark.timing
async def test_sync_function_respects_deadline_but_is_abandoned_not_cancelled():
    finished = []

    def blocking(args):
        time.sleep(0.6)
        finished.append(True)  # a side effect after the run has timed out
        return {}

    agent = graph([KGraphFunctionWorker(name="v", fn=blocking)],
                  [WorkerNodeSpec(id="v", worker_name="v")],
                  [EdgeSpec(source="start", destination="v"), EdgeSpec(source="v", destination="end")])
    t0 = time.monotonic()
    with pytest.raises(WorkerTimeoutError):
        await agent.run({}, timeout=0.2, grace=0.1)
    assert time.monotonic() - t0 < 0.55   # the run stopped waiting on time...
    assert finished == []
    await asyncio.sleep(0.6)
    assert finished == [True]             # ...but the thread ran to completion


# ---------------------------------------------------------------------------
# P4 / P5: partial results and usage on failure
# ---------------------------------------------------------------------------

class TestFailureAccounting:
    def two_step(self, llm, **node_kwargs):
        return graph(
            [KGraphChatWorker(name="first", llm=llm, include_conversation=False),
             KGraphChatWorker(name="second", llm=llm, include_conversation=False)],
            [WorkerNodeSpec(id="first", worker_name="first"),
             WorkerNodeSpec(id="second", worker_name="second", **node_kwargs)],
            [EdgeSpec(source="start", destination="first", bindings={"q": b("start", "$.q")}),
             EdgeSpec(source="first", destination="second", bindings={"d": b("first", "$.result_text")}),
             EdgeSpec(source="second", destination="end")])

    async def test_failed_run_carries_partial_result(self):
        llm = ScriptedLLM(script={"first": ["draft"], "second": [RuntimeError("down")]})
        with pytest.raises(WorkerLLMError) as exc:
            await self.two_step(llm).run({})
        partial = exc.value.run_result
        assert partial.results["first"]["result_text"] == "draft"
        assert partial.errors["second"]["type"] == "llm_error"
        assert partial.trace == ["first", "second"]
        assert partial.usage["calls"] == 2
        assert partial.usage["outcomes"] == {"ok": 1, "llm_error": 1}

    async def test_every_attempt_is_counted(self):
        llm = ScriptedLLM(script={"first": ["draft"],
                                  "second": [rate_limit_error(), Delay(2, "late"), "final"]})
        agent = self.two_step(llm, timeout_seconds=0.2,
                              retry=RetrySpec(max_attempts=3, initial_interval=0, jitter=False))
        result = await agent.run({})
        assert result.results["second"]["result_text"] == "final"
        assert result.usage["calls"] == 4
        assert result.usage["outcomes"] == {"ok": 2, "llm_error": 1, "timeout": 1}
        outcomes = [result.usage_log[k]["outcome"] for k in ("second#1", "second#2", "second#3")]
        assert outcomes == ["llm_error", "timeout", "ok"]
        assert result.usage_log["second#1"]["input_tokens"] == 0

    async def test_parse_failures_count_real_tokens(self):
        class Out(BaseModel):
            text: str

        llm = ScriptedLLM(default=[{"wrong": 1}, {"text": "ok"}])
        agent = graph([KGraphStructuredWorker(name="st", llm=llm, schema=Out)],
                      [WorkerNodeSpec(id="st", worker_name="st", retry=RetrySpec(
                          max_attempts=2, initial_interval=0, retry_on=["parse_error"]))],
                      [EdgeSpec(source="start", destination="st", bindings={"q": b("start", "$.q")}),
                       EdgeSpec(source="st", destination="end")])
        result = await agent.run({})
        assert result.usage["calls"] == 2
        assert result.usage["input_tokens"] == 2 * llm.input_tokens
        assert result.usage["outcomes"] == {"parse_error": 1, "ok": 1}

    async def test_record_includes_failed_attempt_usage(self):
        agent = chat_then_fallback(ScriptedLLM(default=[RuntimeError("down")]),
                                   KGraphFunctionWorker(name="fallback", fn=lambda a: {}))
        result = await agent.run({})
        assert result.usage["outcomes"] == {"llm_error": 1}

    async def test_failures_before_any_call_are_not_counted(self):
        agent = chat_then_fallback(ScriptedLLM(default=["x"]),
                                   KGraphFunctionWorker(name="fallback", fn=lambda a: {}))
        result = await agent.run({}, deadline=time.time() - 0.01, grace=1.0)
        assert result.errors["chat"]["type"] == "timeout"
        assert result.usage == {}

    async def test_node_timeout_carries_partial_result(self):
        async def stuck(args):
            await asyncio.sleep(5)

        agent = graph(
            [KGraphFunctionWorker(name="ok", fn=lambda a: {"done": True}),
             KGraphFunctionWorker(name="stuck", fn=stuck)],
            [WorkerNodeSpec(id="ok", worker_name="ok"), WorkerNodeSpec(id="stuck", worker_name="stuck")],
            [EdgeSpec(source="start", destination="ok"), EdgeSpec(source="ok", destination="stuck"),
             EdgeSpec(source="stuck", destination="end")])
        with pytest.raises(WorkerTimeoutError) as exc:
            await agent.run({}, timeout=0.2, grace=0.1)
        assert exc.value.run_result.results["ok"] == {"done": True}
        assert exc.value.run_result.errors["stuck"]["type"] == "timeout"

    async def test_arun_with_checkpointer_attaches_partial_result(self):
        llm = ScriptedLLM(script={"first": ["draft"], "second": [RuntimeError("down")]})
        agent = self.two_step(llm)
        agent.checkpointer = MemorySaver()
        with pytest.raises(WorkerLLMError) as exc:
            await agent.arun(config={"configurable": {"thread_id": "p4"}})
        assert exc.value.run_result.results["first"]["result_text"] == "draft"


# ---------------------------------------------------------------------------
# P6 / P8: tool worker
# ---------------------------------------------------------------------------

@tool
async def lookup(query: str) -> str:
    """Look up a record."""
    return f"record for {query}"


class _Tools:
    def get_tool(self, tool_id):
        return type("T", (), {"get_tool_function": lambda self: lookup})() if tool_id == "lookup" else None


def tool_graph(llm, **worker_kwargs):
    worker = KGraphToolWorker(name="tw", llm=llm, tool_manager=_Tools(), available_tool_ids=["lookup"],
                              summarize_results=False, **worker_kwargs)
    return graph([worker], [WorkerNodeSpec(id="tw", worker_name="tw")],
                 [EdgeSpec(source="start", destination="tw", bindings={"query": b("start", "$.q")}),
                  EdgeSpec(source="tw", destination="end")])


class TestToolWorker:
    TOOL_CALL = AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": "x"}, "id": "c1"}])

    async def test_result_has_usage(self):
        llm = ScriptedLLM(default=[self.TOOL_CALL, "answer"])
        result = await tool_graph(llm).run({})
        assert result.results["tw"]["result_text"] == "answer"
        assert result.results["tw"]["usage"]["calls"] == 2

    async def test_message_order_is_valid_for_anthropic(self):
        # Claude rejects system messages after the conversation starts, and
        # tool-call messages in a request that binds no tools. The tool worker
        # used to send both.
        from langchain_core.messages import ToolMessage

        def leading_system_only(messages):
            seen_other = False
            for m in messages:
                if isinstance(m, SystemMessage):
                    assert not seen_other, "system message after the conversation started"
                else:
                    seen_other = True

        def decision(response):
            def check(messages):
                leading_system_only(messages)
                return response
            return check

        def summary(messages):
            leading_system_only(messages)
            assert not any(isinstance(m, ToolMessage) or getattr(m, "tool_calls", None)
                           for m in messages), "tool-call messages without tools bound"
            assert "record for x" in messages[-1].content  # results passed as text
            return "summary"

        # Decision with history (iteration 2), then the max-iterations summary.
        llm = ScriptedLLM(default=[decision(self.TOOL_CALL), decision(self.TOOL_CALL), summary])
        result = await tool_graph(llm, max_iters=2).run({})
        assert result.results["tw"]["result_text"] == "summary"
        second_decision = llm.calls[1]["messages"]
        assert any(isinstance(m, ToolMessage) for m in second_decision)  # history after the request

    @pytest.mark.parametrize("max_iters", [1, 2, 3])
    async def test_max_iters_allows_that_many_decisions(self, max_iters):
        # Each decision asks for another tool call; after max_iters of them the
        # worker makes one summarization call instead of another decision.
        llm = ScriptedLLM(default=[self.TOOL_CALL] * max_iters + ["summary"])
        result = await tool_graph(llm, max_iters=max_iters).run({})
        assert len(llm.calls) == max_iters + 1
        assert result.results["tw"]["result_text"] == "summary"
        assert result.errors["tw"]["type"] == "max_iterations"

    async def test_max_iterations_summary_failure_raises(self):
        llm = ScriptedLLM(default=[self.TOOL_CALL, RuntimeError("summary down")])
        with pytest.raises(WorkerLLMError, match="max-iterations"):
            await tool_graph(llm, max_iters=1).run({})

    async def test_max_iterations_is_recorded_not_failed(self):
        llm = ScriptedLLM(default=[self.TOOL_CALL, "summary"])
        result = await tool_graph(llm, max_iters=1).run({})
        assert result.results["tw"]["result_text"] == "summary"
        assert result.errors["tw"]["type"] == "max_iterations"
        assert "status" not in result.results["tw"]


# ---------------------------------------------------------------------------
# P7: 4xx retry categories
# ---------------------------------------------------------------------------

async def test_client_error_retried_only_when_listed():
    class BadRequest(Exception):
        status_code = 400

    calls = []

    async def flaky(args):
        calls.append(1)
        if len(calls) == 1:
            raise BadRequest("400")
        return {}

    def agent(retry_on):
        return graph([KGraphFunctionWorker(name="f", fn=flaky)],
                     [WorkerNodeSpec(id="f", worker_name="f", retry=RetrySpec(
                         max_attempts=2, initial_interval=0, retry_on=retry_on))],
                     [EdgeSpec(source="start", destination="f"), EdgeSpec(source="f", destination="end")])

    with pytest.raises(BadRequest):
        await agent(["rate_limit"]).run({})
    calls.clear()
    await agent(["client_error"]).run({})
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# P9: per-run results on a checkpointed thread
# ---------------------------------------------------------------------------

async def test_no_stale_results_across_turns():
    agent = graph(
        [KGraphFunctionWorker(name="route", fn=lambda a: {"first": a["input"] == "one"}),
         KGraphFunctionWorker(name="a", fn=lambda a: {"ran": "a"}),
         KGraphFunctionWorker(name="b", fn=lambda a: {"ran": "b"})],
        [WorkerNodeSpec(id="route", worker_name="route"), WorkerNodeSpec(id="a", worker_name="a"),
         WorkerNodeSpec(id="b", worker_name="b")],
        [EdgeSpec(source="start", destination="route", bindings={"input": b("start", "$.input")}),
         EdgeSpec(source="route", destination="a", condition="has:first"),
         EdgeSpec(source="route", destination="b", condition="__default__"),
         EdgeSpec(source="a", destination="end"), EdgeSpec(source="b", destination="end")],
        checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "turns"}}
    await agent.arun(messages=[HumanMessage(content="one")], config=cfg)
    state = await agent.arun(messages=[HumanMessage(content="two")], config=cfg)
    results = state["agent_data"]["results"]
    assert "a" not in results and results["b"] == {"ran": "b"}
    assert len(state["messages"]) == 2  # the conversation is kept


# ---------------------------------------------------------------------------
# P11: WorkerNodeSpec fields
# ---------------------------------------------------------------------------

class TestNodeFields:
    def agent(self, llm, *, bind_override=False):
        bindings = {"q": b("start", "$.q")}
        if bind_override:
            bindings["system_directive"] = b("start", "$.override")
        return graph(
            [KGraphChatWorker(name="chat", llm=llm, system_directive="worker prompt",
                              include_conversation=False)],
            [WorkerNodeSpec(id="chat", worker_name="chat", system_directive="node prompt")],
            [EdgeSpec(source="start", destination="chat", bindings=bindings),
             EdgeSpec(source="chat", destination="end")])

    async def test_node_system_directive_is_used(self):
        llm = ScriptedLLM(default=["ok"])
        await self.agent(llm).run({})
        system = [m.content for m in llm.calls[0]["messages"] if isinstance(m, SystemMessage)]
        assert system[0] == "node prompt"

    async def test_binding_still_overrides_it(self):
        llm = ScriptedLLM(default=["ok"])
        await self.agent(llm, bind_override=True).run({"override": "per run"})
        system = [m.content for m in llm.calls[0]["messages"] if isinstance(m, SystemMessage)]
        assert system[0] == "per run"

    def test_ignored_fields_warn(self, caplog):
        caplog.set_level(logging.WARNING)
        graph([KGraphFunctionWorker(name="f", fn=dict)],
              [WorkerNodeSpec(id="f", worker_name="f", max_iterations=3, tools=["x"])],
              [EdgeSpec(source="start", destination="f"), EdgeSpec(source="f", destination="end")])
        warning = next(r.getMessage() for r in caplog.records if "ignores" in r.getMessage())
        assert "'tools'" in warning and "'max_iterations'" in warning
        assert "'required_inputs'" not in warning


# ---------------------------------------------------------------------------
# P12: minor
# ---------------------------------------------------------------------------

class TestMinor:
    def test_status_code_condition_is_not_an_error_route(self, caplog):
        caplog.set_level(logging.WARNING)
        graph([KGraphFunctionWorker(name="f", fn=dict)],
              [WorkerNodeSpec(id="f", worker_name="f", on_error="record")],
              [EdgeSpec(source="start", destination="f"),
               EdgeSpec(source="f", destination="end", condition="has:status_code"),
               EdgeSpec(source="f", destination="end", condition="__default__")])
        assert "routes on the error" in caplog.text

    def test_jitter_never_exceeds_max_interval(self):
        spec = RetrySpec(initial_interval=10, max_interval=10, jitter=True)
        assert all(_backoff(spec, 1) <= 10 for _ in range(200))

    async def test_chat_worker_that_was_never_activated_skips(self):
        # KGraphCaseAgent wires every candidate worker but activates only the
        # selected one; the others must do nothing rather than raise.
        from langgraph.graph import END, START, StateGraph
        from kgraphplanner.agent.kgraph_base_agent import AgentState

        llm = ScriptedLLM(default=["x"])
        g = StateGraph(AgentState)
        entry, exit_ = KGraphChatWorker(name="chat", llm=llm).build_subgraph(g, "chat")
        g.add_edge(START, entry)
        g.add_edge(exit_, END)
        state = await g.compile().ainvoke({"messages": [], "agent_data": {}, "work": {}})
        assert "chat" not in (state["agent_data"].get("results") or {})
        assert llm.calls == []

    async def test_chat_worker_without_inputs_raises(self):
        agent = graph([KGraphChatWorker(name="chat", llm=ScriptedLLM(default=["x"]))],
                      [WorkerNodeSpec(id="chat", worker_name="chat")],
                      [EdgeSpec(source="start", destination="chat"),
                       EdgeSpec(source="chat", destination="end")])
        with pytest.raises(WorkerError, match="no prompt or args"):
            await agent.run({})


# ---------------------------------------------------------------------------
# P13: ScriptedLLM
# ---------------------------------------------------------------------------

async def test_scripted_llm_default_calls_logged_by_node_and_reset():
    llm = ScriptedLLM(default=["one"])
    agent = graph([KGraphChatWorker(name="chat", llm=llm, include_conversation=False)],
                  [WorkerNodeSpec(id="chat", worker_name="chat")],
                  [EdgeSpec(source="start", destination="chat", bindings={"q": b("start", "$.q")}),
                   EdgeSpec(source="chat", destination="end")])
    await agent.run({})
    assert len(llm.calls_for("chat")) == 1
    llm.reset()
    assert llm.calls == []
    result = await agent.run({})  # starts from the top of the script again
    assert len(llm.calls_for("chat")) == 1
    assert result.results["chat"]["result_text"] == "one"


def test_old_node_merge_field_is_rejected():
    # WorkerNodeSpec.merge was renamed to reentry before 0.2.0. Specs forbid
    # extra fields, so the old name fails loudly instead of being ignored.
    with pytest.raises(ValidationError, match="merge"):
        WorkerNodeSpec(id="x", worker_name="x", merge="reset")


# ---------------------------------------------------------------------------
# P14 / P15: planner
# ---------------------------------------------------------------------------

class TestPlanner:
    def planner(self, llm, **kwargs):
        from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
        return KGraphPlannerAgent(
            name="p", planner_llm=llm,
            worker_registry={"researcher": KGraphFunctionWorker(name="researcher",
                                                                fn=lambda a: {"result_text": "found"})},
            **kwargs)

    LINEAR = {
        "program_id": "p",
        "static_nodes": [{"id": "start", "worker": "start", "defaults": {"args": {"topic": "t"}}},
                         {"id": "r", "worker": "researcher"}, {"id": "end", "worker": "end"}],
        "static_edges": [{"source": "start", "destination": "r",
                          "bindings": {"topic": [{"from_node": "start", "path": "$.topic"}]}},
                         {"source": "r", "destination": "end"}],
        "exit_nodes": ["end"],
    }

    def test_branch_conditions_must_be_declarative(self):
        ConditionalBranch(condition="has:ok", destination_tpl="x")
        with pytest.raises(ValidationError, match="expression conditions are disabled"):
            ConditionalBranch(condition="__import__('os').system('x')", destination_tpl="x")

    def test_generated_specs_reject_expressions(self):
        spec = GraphSpec(graph_id="x", nodes=[StartNodeSpec(), EndNodeSpec(),
                                             WorkerNodeSpec(id="r", worker_name="researcher")],
                         edges=[EdgeSpec(source="start", destination="r"),
                                EdgeSpec(source="r", destination="end", condition="result['x']")])
        with pytest.raises(GraphSpecError, match="expression conditions are disabled"):
            self.planner(ScriptedLLM()).exec_agent(spec)

    async def test_invalid_graph_is_replanned_with_the_errors(self):
        mixed = {**self.LINEAR,
                 "static_edges": self.LINEAR["static_edges"],
                 "conditionals": [{"source_tpl": "r",
                                   "branches": [{"condition": "has:x", "destination_tpl": "end"}]}]}
        llm = ScriptedLLM(default=[mixed, self.LINEAR])
        program = await self.planner(llm).plan([HumanMessage(content="research t")])
        assert program.conditionals == []
        retry_prompt = llm.calls[1]["messages"][-1].content
        assert "previous ProgramSpec was invalid" in retry_prompt
        assert "both conditional and unconditional" in retry_prompt

    FAN_IN = {
        "program_id": "fan",
        "static_nodes": [
            {"id": "start", "worker": "start", "defaults": {"args": {"topic": "t"}}},
            {"id": "r1", "worker": "researcher"}, {"id": "r2", "worker": "researcher_down"},
            {"id": "agg", "worker": "aggregator"}, {"id": "end", "worker": "end"}],
        "static_edges": [
            {"source": "start", "destination": "r1",
             "bindings": {"topic": [{"from_node": "start", "path": "$.topic"}]}},
            {"source": "start", "destination": "r2",
             "bindings": {"topic": [{"from_node": "start", "path": "$.topic"}]}},
            {"source": "r1", "destination": "agg",
             "bindings": {"a": [{"from_node": "r1", "path": "$.result_text"}]}},
            {"source": "r2", "destination": "agg",
             "bindings": {"b": [{"from_node": "r2", "path": "$.result_text"}]}},
            {"source": "agg", "destination": "end"}],
        "exit_nodes": ["end"],
    }

    async def test_one_failed_parallel_worker_doesnt_fail_the_plan(self):
        # LangGraph's error_handler doesn't run for a node that fails in
        # parallel with others, so on_error="record" is handled by our own
        # node wrapper. The aggregator must run once, with the good result.
        from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
        from kgraphplanner.program.program import ProgramSpec

        def down(args):
            raise RuntimeError("search down")

        agg_calls = []

        def aggregate(args):
            agg_calls.append(dict(args))
            return {"result_text": args.get("a", "")}

        planner = KGraphPlannerAgent(
            name="p", planner_llm=ScriptedLLM(), node_defaults={"on_error": "record"},
            worker_registry={
                "researcher": KGraphFunctionWorker(name="researcher", fn=lambda a: {"result_text": "found"}),
                "researcher_down": KGraphFunctionWorker(name="researcher_down", fn=down),
                "aggregator": KGraphFunctionWorker(name="aggregator", fn=aggregate),
            })
        result = await planner.execute(planner.expand(ProgramSpec.model_validate(self.FAN_IN)), {})
        assert agg_calls == [{"a": "found", "b": ""}]
        assert result.errors["r2"]["type"] == "worker_error"
        assert result.results["r2"]["status"] == "error"
        assert result.results["agg"]["result_text"] == "found"

    async def test_arun_returns_error_state_when_replanning_fails(self):
        mixed = {**self.LINEAR, "conditionals": [
            {"source_tpl": "r", "branches": [{"condition": "has:x", "destination_tpl": "end"}]}]}
        llm = ScriptedLLM(default=[mixed, mixed, mixed])   # 1 attempt + 2 retries
        state = await self.planner(llm).arun(messages=[HumanMessage(content="research t")])
        assert len(llm.calls) == 3
        assert state["program_spec"] is None
        assert "both conditional and unconditional" in state["agent_data"]["errors"]["planner"]

    async def test_execute_and_node_defaults(self):
        planner = self.planner(ScriptedLLM(), node_defaults={"on_error": "record", "timeout_seconds": 5})
        from kgraphplanner.program.program import ProgramSpec
        spec = planner.expand(ProgramSpec.model_validate(self.LINEAR))
        node = spec.get_node_by_id("r")
        assert (node.on_error, node.timeout_seconds) == ("record", 5)
        result = await planner.execute(spec, {"topic": "t"})
        assert result.results["r"] == {"result_text": "found"}
