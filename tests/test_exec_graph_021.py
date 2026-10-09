"""
Exec graph changes in 0.2.1 (planning/v0_2_1_patch_plan.md, K2, K3, K4, K6).

K2 — the bugs:
  - concat_text raised TypeError when the previous value was None (an earlier
    overwrite binding whose path didn't resolve) or the value wasn't a string;
  - a caller-seeded results["start"] replaced the StartNodeSpec defaults, so
    every key the caller left out resolved to None;
  - a binding error surfaced from an anonymous hop node.
"""
import pytest

from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.graph.exec_graph import (
    Binding, EdgeSpec, EndNodeSpec, GraphSpec, StartNodeSpec, WorkerNodeSpec,
)
from kgraphplanner.graph.graph_hop import apply_reduce, merge_activation
from kgraphplanner.worker.kgraph_function_worker import KGraphFunctionWorker


def b(node, path, **kw):
    return [Binding(from_node=node, path=path, **kw)]


def echo_graph(start_args, bindings, **agent_kwargs):
    """start -> echo -> end; echo returns its inputs."""
    spec = GraphSpec(graph_id="e", nodes=[
        StartNodeSpec(initial_data={"args": start_args}), EndNodeSpec(),
        WorkerNodeSpec(id="echo", worker_name="echo"),
    ], edges=[
        EdgeSpec(source="start", destination="echo", bindings=bindings),
        EdgeSpec(source="echo", destination="end"),
    ])
    return KGraphExecGraphAgent(name="e", graph_spec=spec, worker_registry={
        "echo": KGraphFunctionWorker(name="echo", fn=lambda a: {"args": a})}, **agent_kwargs)


# ---------------------------------------------------------------------------
# K2: concat_text, start defaults, binding errors
# ---------------------------------------------------------------------------

class TestConcatText:
    def test_none_previous_value(self):
        assert apply_reduce({"t": None}, "t", "next", "concat_text") == {"t": "next"}

    def test_none_value(self):
        assert apply_reduce({"t": "kept"}, "t", None, "concat_text") == {"t": "kept"}

    def test_non_string_values_are_converted(self):
        assert apply_reduce({"t": 1}, "t", {"a": 2}, "concat_text") == {"t": "1\n{'a': 2}"}

    async def test_unresolved_overwrite_then_concat_in_a_graph(self):
        # The first binding resolves to None (no such start key); the second
        # concatenates onto it. This raised TypeError before 0.2.1.
        agent = echo_graph({"b": "second"}, {"t": [
            Binding(from_node="start", path="$.missing"),
            Binding(from_node="start", path="$.b", reduce="concat_text"),
        ]})
        result = await agent.run({})
        assert result.results["echo"]["args"]["t"] == "second"


class TestStartDefaults:
    async def test_seeded_start_keeps_undeclared_defaults(self):
        # A caller that seeds results["start"] itself (raw ainvoke) used to
        # lose every StartNodeSpec default it didn't repeat.
        agent = echo_graph({"mode": "default", "lang": "en"},
                           {"mode": b("start", "$.mode"), "lang": b("start", "$.lang")})
        state = await agent.get_compiled_graph().ainvoke({
            "messages": [], "work": {},
            "agent_data": {"results": {"start": {"mode": "seeded"}}},
        })
        assert state["agent_data"]["results"]["echo"]["args"] == {"mode": "seeded", "lang": "en"}

    async def test_start_data_still_wins(self):
        agent = echo_graph({"mode": "default"}, {"mode": b("start", "$.mode")})
        result = await agent.run({"mode": "run"})
        assert result.results["echo"]["args"] == {"mode": "run"}


def test_binding_error_names_the_edge():
    # A json transform of a value that can't be serialised fails inside the hop.
    edge = EdgeSpec(source="plan", destination="content",
                    bindings={"x": b("plan", "$.y", transform="json")})
    with pytest.raises(TypeError) as info:
        merge_activation({}, {}, edge, {"plan": {"y": object()}})
    # The original exception, with the edge named in a note.
    assert "not JSON serializable" in str(info.value)
    assert any("'plan' -> 'content'" in n and "['x']" in n for n in info.value.__notes__)


def test_binding_error_note_holds_no_bound_values():
    # json.dumps of a set raises TypeError; the note must not echo the value.
    secret = "SSN-123-45-6789"
    edge = EdgeSpec(source="plan", destination="content",
                    bindings={"x": b("plan", "$.y", transform="json")})
    with pytest.raises(TypeError) as info:
        merge_activation({}, {}, edge, {"plan": {"y": {secret}}})
    assert info.value.__notes__
    assert not any(secret in n for n in info.value.__notes__)


async def test_binding_from_a_node_that_has_not_run_is_skipped():
    # A skipped branch leaves the destination's default (or nothing) in place,
    # instead of overwriting it with None. A missing *path* on a node that did
    # run still resolves to None.
    edge = EdgeSpec(source="x", destination="y", bindings={
        "from_missing_node": b("never_ran", "$.v"),
        "from_missing_path": b("ran", "$.nope"),
    })
    act = merge_activation({}, {"args": {"from_missing_node": "default"}}, edge, {"ran": {}})
    assert act["args"] == {"from_missing_node": "default", "from_missing_path": None}


# ---------------------------------------------------------------------------
# K4: conditions on start edges are guards
# ---------------------------------------------------------------------------

from kgraphplanner.graph.exec_errors import ExecGraphRoutingError, GraphSpecError  # noqa: E402
from kgraphplanner.testing import assert_node_order  # noqa: E402

CLASSIFIERS = ["reactive", "blocker", "cadence"]


def nurture_graph(*, reactive_edge=None, extra_start_edges=(), agg_defaults=None, **agent_kwargs):
    """start -> reactive | blocker | cadence (parallel) -> resolver -> end.

    The reactive classifier's start edge is guarded by default: it runs only
    when there is an inbound message.
    """
    reactive_edge = reactive_edge or EdgeSpec(
        source="start", destination="reactive", condition="has:inbound_message",
        bindings={"input": b("start", "$.inbound_message")})
    workers = {n: KGraphFunctionWorker(name=n, fn=(lambda name: lambda a: {"signal": f"{name}:{a.get('input', '-')}"})(n))
               for n in CLASSIFIERS}
    workers["resolver"] = KGraphFunctionWorker(name="resolver", fn=lambda a: {"seen": dict(a)})
    nodes = [StartNodeSpec(initial_data={"args": {"inbound_message": "", "lead": "L1"}}), EndNodeSpec(),
             *[WorkerNodeSpec(id=n, worker_name=n) for n in CLASSIFIERS],
             WorkerNodeSpec(id="resolver", worker_name="resolver",
                            defaults={"args": agg_defaults} if agg_defaults else {})]
    edges = [reactive_edge,
             EdgeSpec(source="start", destination="blocker", bindings={"input": b("start", "$.lead")}),
             EdgeSpec(source="start", destination="cadence", bindings={"input": b("start", "$.lead")}),
             *extra_start_edges,
             *[EdgeSpec(source=n, destination="resolver", bindings={n: b(n, "$.signal")}) for n in CLASSIFIERS],
             EdgeSpec(source="resolver", destination="end")]
    return KGraphExecGraphAgent(name="n", graph_spec=GraphSpec(graph_id="n", nodes=nodes, edges=edges),
                                worker_registry=workers, **agent_kwargs)


class TestStartGuards:
    async def test_false_guard_skips_only_that_branch(self):
        result = await nurture_graph().run({"inbound_message": ""})
        assert "reactive" not in result.trace
        assert sorted(result.trace[:2]) == ["blocker", "cadence"]
        assert result.results["resolver"]["seen"] == {"blocker": "blocker:L1", "cadence": "cadence:L1"}

    async def test_true_guard_runs_it_in_parallel_with_the_others(self):
        result = await nurture_graph().run({"inbound_message": "stop texting me"})
        assert sorted(result.trace[:3]) == ["blocker", "cadence", "reactive"]
        assert result.results["resolver"]["seen"]["reactive"] == "reactive:stop texting me"

    async def test_skipped_branch_gets_the_downstream_default(self):
        result = await nurture_graph(agg_defaults={"reactive": "no_inbound"}).run({})
        assert result.results["resolver"]["seen"]["reactive"] == "no_inbound"

    async def test_default_fires_only_when_nothing_else_does(self):
        workers = {n: KGraphFunctionWorker(name=n, fn=lambda a: {}) for n in ("a", "b")}
        spec = GraphSpec(graph_id="d", nodes=[StartNodeSpec(), EndNodeSpec(),
                         WorkerNodeSpec(id="a", worker_name="a"), WorkerNodeSpec(id="b", worker_name="b")],
                         edges=[EdgeSpec(source="start", destination="a", condition="has:go"),
                                EdgeSpec(source="start", destination="b", condition="__default__"),
                                EdgeSpec(source="a", destination="end"), EdgeSpec(source="b", destination="end")])
        agent = KGraphExecGraphAgent(name="d", graph_spec=spec, worker_registry=workers)
        assert_node_order(await agent.run({"go": 1}), ["a"])
        assert_node_order(await agent.run({}), ["b"])

    async def test_nothing_fires_raises(self):
        workers = {"a": KGraphFunctionWorker(name="a", fn=lambda a: {})}
        spec = GraphSpec(graph_id="n", nodes=[StartNodeSpec(), EndNodeSpec(),
                         WorkerNodeSpec(id="a", worker_name="a")],
                         edges=[EdgeSpec(source="start", destination="a", condition="has:go"),
                                EdgeSpec(source="a", destination="end")])
        with pytest.raises(ExecGraphRoutingError, match="start"):
            await KGraphExecGraphAgent(name="n", graph_spec=spec, worker_registry=workers).run({})

    async def test_guards_are_evaluated_once_per_run(self, monkeypatch):
        # A guard on run.now or run.time_left can change between evaluations.
        # Simulate one that holds the first time and fails after: the router
        # and the gather must agree, so the reactive branch runs with its
        # binding rather than with defaults only.
        import kgraphplanner.agent.kgraph_exec_graph_agent as exec_mod
        real = exec_mod.evaluate_condition
        calls = []

        def flips(cond, *, source, context):
            if source != "start":
                return real(cond, source=source, context=context)
            calls.append(1)
            return len(calls) == 1

        monkeypatch.setattr(exec_mod, "evaluate_condition", flips)
        result = await nurture_graph().run({"inbound_message": "hi"})
        assert len(calls) == 1
        assert result.results["resolver"]["seen"]["reactive"] == "reactive:hi"

    async def test_several_start_edges_into_one_node_merge_only_fired_ones(self):
        agent = nurture_graph(extra_start_edges=[
            EdgeSpec(source="start", destination="cadence", condition="has:inbound_message",
                     bindings={"note": b("start", "$.inbound_message")})])
        quiet = await agent.run({"inbound_message": ""})
        assert "note" not in quiet.state["agent_data"]["activation"]["cadence"]["args"]
        loud = await agent.run({"inbound_message": "hi"})
        assert loud.state["agent_data"]["activation"]["cadence"]["args"]["note"] == "hi"
        assert loud.trace.count("cadence") == 1

    async def test_guards_see_start_data_and_injected_input(self):
        guard = EdgeSpec(source="start", destination="reactive", condition="has:input",
                         bindings={"input": b("start", "$.input")})
        from langchain_core.messages import HumanMessage
        state = await nurture_graph(reactive_edge=guard).arun(messages=[HumanMessage(content="typed")])
        assert "reactive" in state["agent_data"]["trace"]

    def test_expression_guards_respect_allow_expressions(self):
        guard = EdgeSpec(source="start", destination="reactive",
                         condition="len(result.get('inbound_message', '')) > 0")
        with pytest.raises(GraphSpecError, match="expression conditions are disabled"):
            nurture_graph(reactive_edge=guard, allow_expressions=False)

    @pytest.mark.parametrize("kwargs, match", [
        ({"condition": "has:x", "max_traversals": 1}, "start edges don't loop"),
        ({"condition": "has:x", "on_exhausted": "error"}, "start edges don't loop"),
        ({"condition": "on_error"}, "start has no node to fail"),
    ])
    def test_loop_and_error_options_are_rejected(self, kwargs, match):
        with pytest.raises(GraphSpecError, match=match):
            nurture_graph(reactive_edge=EdgeSpec(source="start", destination="reactive", **kwargs))

    def test_default_with_unconditional_edges_warns(self, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        nurture_graph(extra_start_edges=[
            EdgeSpec(source="start", destination="cadence", condition="__default__")])
        assert "can never fire" in caplog.text

    async def test_guarded_edge_to_a_node_reached_from_a_worker_is_pruned(self):
        agent = nurture_graph(extra_start_edges=[
            EdgeSpec(source="start", destination="resolver", condition="has:inbound_message")])
        result = await agent.run({"inbound_message": "x"})
        assert result.trace.count("resolver") == 1 and result.trace[-1] == "resolver"

    async def test_specs_without_guards_build_the_same_graph(self):
        unguarded = EdgeSpec(source="start", destination="reactive",
                             bindings={"input": b("start", "$.lead")})
        nodes = set(nurture_graph(reactive_edge=unguarded).build_graph().nodes)
        assert not any(n.startswith("__start_guard__") for n in nodes)


class TestPlannerStartConditionals:
    LINEAR = {
        "program_id": "p",
        "static_nodes": [{"id": "start", "worker": "start", "defaults": {"args": {"topic": "t"}}},
                         {"id": "r", "worker": "researcher"}, {"id": "end", "worker": "end"}],
        "static_edges": [{"source": "start", "destination": "r",
                          "bindings": {"topic": [{"from_node": "start", "path": "$.topic"}]}},
                         {"source": "r", "destination": "end"}],
        "exit_nodes": ["end"],
    }

    def planner(self, llm):
        from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
        return KGraphPlannerAgent(name="p", planner_llm=llm, worker_registry={
            "researcher": KGraphFunctionWorker(name="researcher", fn=lambda a: {"result_text": "found"})})

    def test_conditional_from_start_fails_program_validation(self):
        from kgraphplanner.program.program import ProgramSpec
        from kgraphplanner.program.program_expander import validate_program_spec
        program = ProgramSpec.model_validate({**self.LINEAR, "conditionals": [
            {"source_tpl": "start", "branches": [{"condition": "has:topic", "destination_tpl": "r"}]}]})
        ok, messages = validate_program_spec(program)
        assert not ok and any("cannot originate from 'start'" in m for m in messages)

    def test_prompt_states_the_rule(self):
        from kgraphplanner.agent.kgraph_planner_agent import PLANNER_SYSTEM_PROMPT
        assert 'Conditionals cannot originate from "start"' in PLANNER_SYSTEM_PROMPT

    async def test_replans_with_the_message_and_accepts_the_alternative(self):
        from langchain_core.messages import HumanMessage
        from kgraphplanner.testing import ScriptedLLM
        from_start = {**self.LINEAR, "conditionals": [
            {"source_tpl": "start", "branches": [{"condition": "has:topic", "destination_tpl": "r"}],
             "default_destination_tpl": "end"}]}
        llm = ScriptedLLM(default=[from_start, self.LINEAR])
        program = await self.planner(llm).plan([HumanMessage(content="research t")])
        assert program.conditionals == []
        assert "cannot originate from 'start'" in llm.calls[1]["messages"][-1].content


# ---------------------------------------------------------------------------
# K6: the presence gate, and worker descriptions in the planner prompt
# ---------------------------------------------------------------------------

from kgraphplanner.worker.gates import DEFAULT_GATE_DESCRIPTION, is_present, presence_gate  # noqa: E402


class TestPresenceGate:
    @pytest.mark.parametrize("value, present", [
        (None, False), ("", False), ("   ", False), ([], False), ({}, False),
        ("hi", True), (0, True), (False, True), ([0], True), ({"a": None}, True),
    ])
    def test_emptiness_rule(self, value, present):
        assert is_present(value) is present

    async def test_routes_on_present_inputs(self):
        workers = {"gate": presence_gate(),
                   "classifier": KGraphFunctionWorker(name="classifier", fn=lambda a: {"got": a["input"]})}
        spec = GraphSpec(graph_id="g", nodes=[StartNodeSpec(), EndNodeSpec(),
                         WorkerNodeSpec(id="gate", worker_name="gate"),
                         WorkerNodeSpec(id="classifier", worker_name="classifier")], edges=[
            EdgeSpec(source="start", destination="gate",
                     bindings={"msg": b("start", "$.msg"),
                               "system_directive_append": b("start", "$.note")}),
            EdgeSpec(source="gate", destination="classifier", condition="has:present.msg",
                     bindings={"input": b("gate", "$.inputs.msg")}),
            EdgeSpec(source="gate", destination="end", condition="__default__"),
            EdgeSpec(source="classifier", destination="end")])
        agent = KGraphExecGraphAgent(name="g", graph_spec=spec, worker_registry=workers)

        hit = await agent.run({"msg": "hello", "note": "x"})
        assert_node_order(hit, ["gate", "classifier"])
        assert hit.results["classifier"] == {"got": "hello"}
        assert hit.results["gate"]["present"] == {"msg": True}      # directive keys excluded
        assert_node_order(await agent.run({"msg": "  "}), ["gate"])

    def test_is_a_code_node_with_a_description(self):
        gate = presence_gate("g2")
        assert gate.name == "g2" and gate.uses_llm is False
        assert gate.description == DEFAULT_GATE_DESCRIPTION
        assert presence_gate(description="custom").description == "custom"


class TestWorkerDescriptions:
    def planner(self, registry):
        from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
        from kgraphplanner.testing import ScriptedLLM
        return KGraphPlannerAgent(name="p", planner_llm=ScriptedLLM(), worker_registry=registry)

    def test_no_descriptions_keeps_the_old_prompt_exactly(self):
        registry = {"researcher": KGraphFunctionWorker(name="researcher", fn=dict),
                    "summarizer": KGraphFunctionWorker(name="summarizer", fn=dict)}
        assert self.planner(registry).workers_prompt() == "Available workers: ['researcher', 'summarizer']"

    def test_descriptions_are_listed(self):
        registry = {"researcher": KGraphFunctionWorker(name="researcher", fn=dict,
                                                       description="Finds facts."),
                    "gate": presence_gate(),
                    "plain": KGraphFunctionWorker(name="plain", fn=dict)}
        assert self.planner(registry).workers_prompt() == (
            "Available workers:\n"
            "- researcher: Finds facts.\n"
            f"- gate: {DEFAULT_GATE_DESCRIPTION}\n"
            "- plain")

    async def test_the_planner_prompt_uses_them(self):
        from langchain_core.messages import HumanMessage
        from kgraphplanner.testing import ScriptedLLM
        from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
        llm = ScriptedLLM(default=[TestPlannerStartConditionals.LINEAR])
        planner = KGraphPlannerAgent(name="p", planner_llm=llm, worker_registry={
            "researcher": KGraphFunctionWorker(name="researcher", fn=dict, description="Finds facts.")})
        await planner.plan([HumanMessage(content="x")])
        assert "- researcher: Finds facts." in llm.calls[0]["messages"][0].content

    async def test_a_generated_plan_routes_on_input_through_the_gate(self):
        from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
        from kgraphplanner.program.program import ProgramSpec
        from kgraphplanner.testing import ScriptedLLM
        planner = KGraphPlannerAgent(name="p", planner_llm=ScriptedLLM(), worker_registry={
            "gate": presence_gate(),
            "classifier": KGraphFunctionWorker(name="classifier", fn=lambda a: {"got": a.get("input")})})
        program = ProgramSpec.model_validate({
            "program_id": "gated",
            "static_nodes": [{"id": "start", "worker": "start", "defaults": {"args": {"inbound_message": ""}}},
                             {"id": "gate", "worker": "gate"}, {"id": "classifier", "worker": "classifier"},
                             {"id": "end", "worker": "end"}],
            "static_edges": [
                {"source": "start", "destination": "gate",
                 "bindings": {"inbound_message": [{"from_node": "start", "path": "$.inbound_message"}]}},
                {"source": "classifier", "destination": "end"}],
            "conditionals": [{
                "source_tpl": "gate",
                "branches": [{"condition": "has:present.inbound_message", "destination_tpl": "classifier",
                              "bindings": {"input": [{"from_node_tpl": "gate",
                                                      "path": "$.inputs.inbound_message"}]}}],
                "default_destination_tpl": "end"}],
            "exit_nodes": ["end"],
        })
        spec = planner.expand(program)
        empty = await planner.execute(spec, {"inbound_message": ""})
        assert empty.trace == ["gate"]
        full = await planner.execute(spec, {"inbound_message": "stop"})
        assert full.trace == ["gate", "classifier"] and full.results["classifier"] == {"got": "stop"}


# ---------------------------------------------------------------------------
# K3: built-in workers use the first non-empty input key
# ---------------------------------------------------------------------------

from kgraphplanner.testing import ScriptedLLM  # noqa: E402
from kgraphplanner.worker.kgraph_case_worker import KGCase, KGraphCaseWorker  # noqa: E402
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker  # noqa: E402
from kgraphplanner.worker.worker_errors import WorkerError  # noqa: E402


def one_node(worker, args):
    spec = GraphSpec(graph_id="o", nodes=[StartNodeSpec(initial_data={"args": args}), EndNodeSpec(),
                     WorkerNodeSpec(id="w", worker_name=worker.name)], edges=[
        EdgeSpec(source="start", destination="w", bindings={k: b("start", f"$.{k}") for k in args}),
        EdgeSpec(source="w", destination="end")])
    return KGraphExecGraphAgent(name="o", graph_spec=spec, worker_registry={worker.name: worker})


class TestFirstNonEmptyInput:
    async def test_case_worker_skips_an_empty_query(self):
        llm = ScriptedLLM(default=["a"])
        worker = KGraphCaseWorker(name="case", llm=llm, cases=[KGCase("a", "A", "a")])
        result = await one_node(worker, {"query": "", "input": "hello"}).run({})
        assert result.results["w"]["input"] == "hello"
        assert "Categorize: hello" in llm.calls[0]["messages"][-1].content

    async def test_case_worker_with_all_inputs_empty_raises(self):
        worker = KGraphCaseWorker(name="case", llm=ScriptedLLM(default=["a"]), cases=[KGCase("a", "A", "a")])
        with pytest.raises(WorkerError, match="No input"):
            await one_node(worker, {"query": " ", "message": None, "context": "x"}).run({})

    async def test_chat_worker_skips_an_empty_query(self):
        llm = ScriptedLLM(default=["ok"])
        worker = KGraphChatWorker(name="chat", llm=llm, include_conversation=False)
        await one_node(worker, {"query": "", "message": "the real question"}).run({})
        assert llm.calls[0]["messages"][-1].content == "the real question"

    async def test_chat_worker_with_empty_direct_keys_sends_all_args(self):
        llm = ScriptedLLM(default=["ok"])
        worker = KGraphChatWorker(name="chat", llm=llm, include_conversation=False)
        await one_node(worker, {"query": "", "context": "useful"}).run({})
        assert "context: useful" in llm.calls[0]["messages"][-1].content
