"""
Regression tests for KGraphExecGraphAgent routing.

The bugs:
  - R1: a conditional edge to `end` was silently dropped, so `validate -> end`
    fell back to the first branch (retry content) until the recursion limit.
  - R2: the max_traversals counter lived on the compiled graph, shared by every
    run; at the limit a code-only source looped back to itself forever.
  - R3: a condition that raised counted as False and fell through to the first
    branch.
  - R6: a re-entered node kept its previous inputs; append_list grew each pass
    (now opt-in per node with reentry="reset").

What must hold now:
  - conditional edges to `end` route to END; no match and no default raises;
  - traversal counts are per run, and on_exhausted decides what happens at the
    limit;
  - bad conditions fail at build time, and raise at runtime;
  - conditions can see all results, traversal counts and the run deadline;
  - the spec is validated when the agent is built.
"""
import asyncio
import logging

import pytest
from langgraph.checkpoint.memory import MemorySaver

from kgraphplanner.agent.kgraph_base_agent import (
    add_values, append_values, merge_agent_data, replace_value,
)
from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.graph.exec_errors import (
    ExecGraphConditionError, ExecGraphLoopExhausted, ExecGraphRoutingError, GraphSpecError,
)
from kgraphplanner.graph.exec_graph import (
    AggregatorNodeSpec, Binding, EdgeSpec, EndNodeSpec, GraphSpec, StartNodeSpec,
    WorkerNodeSpec, validate_graph_spec,
)
from kgraphplanner.testing import ScriptedLLM, assert_node_order
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_function_worker import KGraphFunctionWorker


def b(node, path="$"):
    return [Binding(from_node=node, path=path)]


def make_agent(validate_edges, *, content_fn=None, validate_fn=None, extra_nodes=(),
               extra_workers=None, **agent_kwargs):
    """start -> content -> validate -> (validate_edges)."""
    def default_content(args):
        # Second pass gets attempt=1 from the retry edge binding.
        return {"result_text": "good" if args.get("attempt") else "bad", "attempt": args.get("attempt", 0)}

    def default_validate(args):
        ok = args["draft"] == "good"
        return {"ok": ok, "retry": not ok, "draft": args["draft"]}

    workers = {
        "content": KGraphFunctionWorker(name="content", fn=content_fn or default_content),
        "validate": KGraphFunctionWorker(name="validate", fn=validate_fn or default_validate),
        **(extra_workers or {}),
    }
    spec = GraphSpec(
        graph_id="t",
        nodes=[
            StartNodeSpec(),
            EndNodeSpec(),
            WorkerNodeSpec(id="content", worker_name="content", reentry="reset"),
            WorkerNodeSpec(id="validate", worker_name="validate"),
            *extra_nodes,
        ],
        edges=[
            EdgeSpec(source="start", destination="content"),
            EdgeSpec(source="content", destination="validate",
                     bindings={"draft": b("content", "$.result_text")}),
            *validate_edges,
        ],
    )
    return KGraphExecGraphAgent(name="t", graph_spec=spec, worker_registry=workers, **agent_kwargs)


RETRY_ONCE = EdgeSpec(
    source="validate", destination="content", condition="has:retry",
    max_traversals=1, on_exhausted="default",
    bindings={"attempt": [Binding(literal=1)]},
)


# ---------------------------------------------------------------------------
# R1: conditional edges to `end`
# ---------------------------------------------------------------------------

class TestEndRouting:
    async def test_default_to_end_after_one_pass(self):
        agent = make_agent(
            [RETRY_ONCE, EdgeSpec(source="validate", destination="end", condition="__default__")],
            validate_fn=lambda a: {"ok": True},
        )
        result = await agent.run({}, recursion_limit=30)
        assert_node_order(result, ["content", "validate"])

    async def test_retry_runs_content_exactly_twice(self):
        agent = make_agent(
            [RETRY_ONCE, EdgeSpec(source="validate", destination="end", condition="__default__")])
        result = await agent.run({}, recursion_limit=30)
        assert_node_order(result, ["content", "validate", "content", "validate"])
        assert result.results["validate"]["ok"] is True

    async def test_non_default_condition_to_end(self):
        agent = make_agent([
            EdgeSpec(source="validate", destination="end", condition="has:ok"),
            EdgeSpec(source="validate", destination="content", condition="has:retry",
                     max_traversals=3, on_exhausted="error",
                     bindings={"attempt": [Binding(literal=1)]}),
        ])
        result = await agent.run({}, recursion_limit=30)
        assert_node_order(result, ["content", "validate", "content", "validate"])

    async def test_no_match_and_no_default_raises(self):
        agent = make_agent(
            [EdgeSpec(source="validate", destination="end", condition="eq:ok:never")],
            validate_fn=lambda a: {"ok": True},
        )
        with pytest.raises(ExecGraphRoutingError, match="validate"):
            await agent.run({}, recursion_limit=30)

    def test_mixed_static_and_conditional_fails_at_build(self):
        with pytest.raises(GraphSpecError, match="both conditional and unconditional"):
            make_agent([
                EdgeSpec(source="validate", destination="end"),
                EdgeSpec(source="validate", destination="content", condition="has:retry"),
            ])

    def test_edge_that_would_be_dropped_fails_at_build(self):
        with pytest.raises(GraphSpecError, match="would be dropped"):
            make_agent(
                [EdgeSpec(source="validate", destination="agg")],
                extra_nodes=[AggregatorNodeSpec(id="agg", aggregation_function="f")],
            )

    async def test_condition_on_start_edge_is_a_guard(self):
        # Rejected in 0.2.0; a guard since 0.2.1 (see test_exec_graph_021.py).
        spec = GraphSpec(graph_id="t", nodes=[
            StartNodeSpec(), WorkerNodeSpec(id="a", worker_name="a")],
            edges=[EdgeSpec(source="start", destination="a", condition="has:go"),
                   EdgeSpec(source="start", destination="a", condition="__default__"),
                   EdgeSpec(source="a", destination="end")])
        agent = KGraphExecGraphAgent(name="t", graph_spec=spec, worker_registry={
            "a": KGraphFunctionWorker(name="a", fn=lambda a: {})})
        assert (await agent.run({"go": True})).trace == ["a"]


# ---------------------------------------------------------------------------
# R2: traversal counts are per run; on_exhausted
# ---------------------------------------------------------------------------

def always_retry(args):
    return {"retry": True}


class TestTraversalLimits:
    def agent(self, **edge_kwargs):
        return make_agent([
            EdgeSpec(source="validate", destination="content", condition="has:retry",
                     max_traversals=1, **edge_kwargs),
            EdgeSpec(source="validate", destination="end", condition="__default__"),
        ], validate_fn=always_retry)

    async def test_concurrent_runs_each_get_one_retry(self):
        agent = self.agent(on_exhausted="default")
        results = await asyncio.gather(*(agent.run({}, recursion_limit=30) for _ in range(20)))
        for r in results:
            assert r.trace.count("content") == 2, r.trace

    async def test_sequential_runs_each_get_their_retry(self):
        agent = self.agent(on_exhausted="default")
        for _ in range(3):
            r = await agent.run({}, recursion_limit=30)
            assert r.trace.count("content") == 2

    async def test_checkpointed_thread_resets_counts_per_run(self):
        agent = make_agent([
            EdgeSpec(source="validate", destination="content", condition="has:retry",
                     max_traversals=1, on_exhausted="default"),
            EdgeSpec(source="validate", destination="end", condition="__default__"),
        ], validate_fn=always_retry, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "t1"}}
        for _ in range(2):
            state = await agent.arun(config=cfg)
            assert state["agent_data"]["trace"].count("content") == 2

    async def test_exhausted_error_raises(self):
        with pytest.raises(ExecGraphLoopExhausted):
            await self.agent(on_exhausted="error").run({}, recursion_limit=30)

    async def test_unset_on_function_source_raises(self):
        with pytest.raises(ExecGraphLoopExhausted):
            await self.agent().run({}, recursion_limit=30)

    def test_source_final_turn_on_function_source_fails_at_build(self):
        with pytest.raises(GraphSpecError, match="not an LLM worker"):
            self.agent(on_exhausted="source_final_turn")

    async def test_unset_on_llm_source_gets_final_turn(self):
        # The general_agent shape: orchestrator <-> tool loop.
        llm = ScriptedLLM(script={"orchestrator": [
            '{"action": "tool"}', '{"action": "tool"}', '{"action": "respond", "answer": "done"}',
        ]})
        workers = {
            "orchestrator": KGraphChatWorker(name="orchestrator", llm=llm, parse_json_response=True),
            "tool": KGraphFunctionWorker(name="tool", fn=lambda a: {"result_text": "tool output"}),
        }
        spec = GraphSpec(graph_id="g", nodes=[
            StartNodeSpec(initial_data={"args": {"q": "x"}}), EndNodeSpec(),
            WorkerNodeSpec(id="orchestrator", worker_name="orchestrator"),
            WorkerNodeSpec(id="tool", worker_name="tool"),
        ], edges=[
            EdgeSpec(source="start", destination="orchestrator", bindings={"q": b("start", "$.q")}),
            EdgeSpec(source="orchestrator", destination="tool",
                     condition="result.get('action') == 'tool'", max_traversals=1),
            EdgeSpec(source="tool", destination="orchestrator", condition="always",
                     bindings={"tool_result": b("tool", "$.result_text")}),
            EdgeSpec(source="orchestrator", destination="end",
                     condition="result.get('action') == 'respond'"),
        ])
        agent = KGraphExecGraphAgent(name="g", graph_spec=spec, worker_registry=workers)
        result = await agent.run({}, recursion_limit=40)
        assert_node_order(result, ["orchestrator", "tool", "orchestrator", "orchestrator"])
        final_turn = llm.calls_for("orchestrator")[2]["messages"]
        assert any("maximum number of tool call cycles" in str(m.content) for m in final_turn)
        assert result.results["orchestrator"]["answer"] == "done"


# ---------------------------------------------------------------------------
# R3: conditions fail loudly
# ---------------------------------------------------------------------------

class TestConditions:
    def test_syntax_error_fails_at_build(self):
        with pytest.raises(GraphSpecError, match="Invalid condition"):
            make_agent([EdgeSpec(source="validate", destination="end",
                                 condition="result.get('ok' ==")])

    def test_malformed_eq_fails_at_build(self):
        with pytest.raises(GraphSpecError, match="eq:"):
            make_agent([EdgeSpec(source="validate", destination="end", condition="eq:ok")])

    async def test_runtime_error_raises(self):
        agent = make_agent([
            EdgeSpec(source="validate", destination="end", condition="result['retyr']"),
            EdgeSpec(source="validate", destination="content", condition="__default__"),
        ])
        with pytest.raises(ExecGraphConditionError, match="retyr"):
            await agent.run({}, recursion_limit=30)

    def test_expressions_can_be_disabled(self):
        with pytest.raises(GraphSpecError, match="expression conditions are disabled"):
            make_agent([EdgeSpec(source="validate", destination="end",
                                 condition="result.get('ok')")], allow_expressions=False)

    async def test_has_supports_dotted_paths(self):
        agent = make_agent([
            EdgeSpec(source="validate", destination="end", condition="eq:data.verdict:pass"),
        ], validate_fn=lambda a: {"data": {"verdict": "pass"}})
        result = await agent.run({}, recursion_limit=30)
        assert_node_order(result, ["content", "validate"])


# ---------------------------------------------------------------------------
# R4: condition context
# ---------------------------------------------------------------------------

class TestConditionContext:
    def budget_agent(self):
        return make_agent([
            EdgeSpec(source="validate", destination="content",
                     condition="result.get('retry') and run['time_left'] > 60",
                     bindings={"attempt": [Binding(literal=1)]}),
            EdgeSpec(source="validate", destination="end", condition="__default__"),
        ])

    async def test_retry_skipped_when_budget_too_low(self):
        result = await self.budget_agent().run({}, timeout=5, recursion_limit=30)
        assert result.trace.count("content") == 1

    async def test_retry_taken_when_budget_allows(self):
        result = await self.budget_agent().run({}, timeout=300, recursion_limit=30)
        assert result.trace.count("content") == 2

    async def test_no_deadline_means_unlimited_time(self):
        result = await self.budget_agent().run({}, recursion_limit=30)
        assert result.trace.count("content") == 2
        assert result.run["deadline"] is None

    async def test_traversals_and_results_are_visible(self):
        agent = make_agent([
            EdgeSpec(source="validate", destination="content",
                     condition="traversals.get('validate->content', 0) < 2 "
                               "and results['content']['result_text'] == 'bad'"),
            EdgeSpec(source="validate", destination="end", condition="__default__"),
        ], content_fn=lambda a: {"result_text": "bad"})
        result = await agent.run({}, recursion_limit=30)
        assert result.trace.count("content") == 3

    async def test_spec_timeout_is_the_default_deadline(self):
        agent = self.budget_agent()
        agent.graph_spec.timeout_seconds = 5
        result = await agent.run({}, recursion_limit=30)
        assert result.trace.count("content") == 1
        assert result.run["deadline"] - result.run["started_at"] == pytest.approx(5, abs=1)


# ---------------------------------------------------------------------------
# R5: spec validation
# ---------------------------------------------------------------------------

class TestSpecValidation:
    def test_on_exhausted_default_needs_a_default_edge(self):
        with pytest.raises(GraphSpecError, match="no __default__ edge"):
            make_agent([EdgeSpec(source="validate", destination="content", condition="has:retry",
                                 max_traversals=1, on_exhausted="default")])

    def test_registry_key_must_match_worker_name(self):
        with pytest.raises(GraphSpecError, match="does not match the worker's name"):
            make_agent([EdgeSpec(source="validate", destination="end")],
                       extra_workers={"other": KGraphFunctionWorker(name="renamed", fn=dict)})

    def test_all_errors_listed_at_once(self):
        with pytest.raises(GraphSpecError) as exc:
            make_agent([
                EdgeSpec(source="validate", destination="end", condition="eq:x"),
                EdgeSpec(source="validate", destination="content"),
            ])
        assert len(exc.value.errors) >= 2

    def test_missing_default_warns(self, caplog):
        caplog.set_level(logging.WARNING)
        make_agent([EdgeSpec(source="validate", destination="end", condition="has:ok")])
        assert "no __default__ edge" in caplog.text

    def test_example_specs_still_validate(self):
        from kgraphplanner.sample.general.general_agent import build_graph_spec
        assert validate_graph_spec(build_graph_spec()).is_valid


# ---------------------------------------------------------------------------
# R6: input reset on re-entry
# ---------------------------------------------------------------------------

class TestReentry:
    def agent(self, reentry):
        workers = {
            "content": KGraphFunctionWorker(name="content", fn=lambda a: {"items": a.get("items", [])}),
            "validate": KGraphFunctionWorker(name="validate", fn=always_retry),
        }
        spec = GraphSpec(graph_id="m", nodes=[
            StartNodeSpec(initial_data={"args": {"x": 1}}), EndNodeSpec(),
            WorkerNodeSpec(id="content", worker_name="content", reentry=reentry),
            WorkerNodeSpec(id="validate", worker_name="validate"),
        ], edges=[
            EdgeSpec(source="start", destination="content",
                     bindings={"items": [Binding(from_node="start", path="$.x", reduce="append_list")]}),
            EdgeSpec(source="content", destination="validate"),
            EdgeSpec(source="validate", destination="content", condition="has:retry",
                     max_traversals=1, on_exhausted="default",
                     bindings={"items": [Binding(from_node="start", path="$.x", reduce="append_list")]}),
            EdgeSpec(source="validate", destination="end", condition="__default__"),
        ])
        return KGraphExecGraphAgent(name="m", graph_spec=spec, worker_registry=workers)

    async def test_keep_keeps_previous_inputs(self):
        result = await self.agent("keep").run({}, recursion_limit=30)
        assert result.results["content"]["items"] == [1, 1]

    async def test_reset_starts_fresh(self):
        result = await self.agent("reset").run({}, recursion_limit=30)
        assert result.results["content"]["items"] == [1]


# ---------------------------------------------------------------------------
# State reducer and parallel branches
# ---------------------------------------------------------------------------

class TestAgentDataReducer:
    def test_plain_dicts_merge_one_level(self):
        assert merge_agent_data({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"x": 1, "y": 2}}

    def test_replace_clears_a_key(self):
        assert merge_agent_data({"a": {"x": 1}}, {"a": replace_value({})}) == {"a": {}}

    def test_add_and_append(self):
        left = {"t": {"calls": 1, "by_model": {"m": {"calls": 1}}}, "trace": ["a"]}
        right = {"t": add_values({"calls": 2, "by_model": {"m": {"calls": 1}, "n": {"calls": 1}}}),
                 "trace": append_values(["b"])}
        assert merge_agent_data(left, right) == {
            "t": {"calls": 3, "by_model": {"m": {"calls": 2}, "n": {"calls": 1}}},
            "trace": ["a", "b"],
        }

    def test_markers_apply_when_left_is_empty(self):
        assert merge_agent_data({}, {"trace": append_values(["a"])}) == {"trace": ["a"]}

    async def test_parallel_branches_keep_each_others_writes(self):
        llm = ScriptedLLM(default=["a", "b"])
        workers = {
            "left": KGraphChatWorker(name="left", llm=llm, include_conversation=False),
            "right": KGraphChatWorker(name="right", llm=llm, include_conversation=False),
            "join": KGraphFunctionWorker(name="join", fn=lambda a: {"n": len(a)}),
        }
        spec = GraphSpec(graph_id="p", nodes=[
            StartNodeSpec(initial_data={"args": {"q": "x"}}), EndNodeSpec(),
            WorkerNodeSpec(id="left", worker_name="left"),
            WorkerNodeSpec(id="right", worker_name="right"),
            WorkerNodeSpec(id="join", worker_name="join"),
        ], edges=[
            EdgeSpec(source="start", destination="left", bindings={"q": b("start", "$.q")}),
            EdgeSpec(source="start", destination="right", bindings={"q": b("start", "$.q")}),
            EdgeSpec(source="left", destination="join", bindings={"l": b("left", "$.result_text")}),
            EdgeSpec(source="right", destination="join", bindings={"r": b("right", "$.result_text")}),
            EdgeSpec(source="join", destination="end"),
        ])
        result = await KGraphExecGraphAgent(name="p", graph_spec=spec, worker_registry=workers).run({})
        assert result.usage["calls"] == 2
        assert sorted(result.trace[:2]) == ["left", "right"]
        assert result.results["join"]["n"] == 2


# ---------------------------------------------------------------------------
# The general_agent sample, end to end
# ---------------------------------------------------------------------------

class TestGeneralAgentSample:
    async def test_tool_round_trip_then_respond(self):
        import json
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.tools import tool

        from kgraphplanner.sample.general.general_agent import build_agent

        @tool
        async def weather_tool(query: str) -> str:
            """Current weather."""
            return f"{query}: 72F sunny"

        class Tools:
            def get_tool(self, tool_id):
                return type("T", (), {"get_tool_function": lambda self: weather_tool})() \
                    if tool_id == "weather_tool" else None

        llm = ScriptedLLM(script={
            "orchestrator": [
                {"action": "tool", "requests": [{"tool": "weather_tool", "query": "Austin"}],
                 "plan": "get weather, then answer"},
                {"action": "respond", "answer": "It's 72F and sunny in Austin."},
            ],
            "tool_executor": [
                AIMessage(content="", tool_calls=[
                    {"name": "weather_tool", "args": {"query": "Austin"}, "id": "c1"}]),
                "Austin: 72F sunny",              # per-result summary
                '{"include_summaries": ["c1"]}',  # final decision picks it
            ],
            "final_responder": ["Good news: 72F and sunny in Austin!"],
        })
        agent = build_agent(llm, Tools(), checkpointer=MemorySaver())
        state = await agent.arun(messages=[HumanMessage(content="Weather in Austin?")],
                                 config={"configurable": {"thread_id": "g1"}})

        assert_node_order(state, ["orchestrator", "tool_executor", "orchestrator",
                                  "final_responder"])
        results = state["agent_data"]["results"]
        assert results["final_responder"]["result_text"] == "Good news: 72F and sunny in Austin!"

        # The tool executor gets the decision without the orchestrator's own fields.
        request = llm.calls_for("tool_executor")[0]["messages"]
        sent = json.loads(next(m.content for m in request if isinstance(m, HumanMessage)))
        assert json.loads(sent["tool_request"]) == {
            "requests": [{"tool": "weather_tool", "query": "Austin"}]}

        # Second orchestrator turn sees the plan and the tool result, and the
        # user's message only once.
        second = llm.calls_for("orchestrator")[1]["messages"]
        humans = [m.content for m in second if isinstance(m, HumanMessage)]
        assert humans[0] == "Weather in Austin?" and len(humans) == 2
        assert "get weather, then answer" in humans[1] and "72F sunny" in humans[1]
