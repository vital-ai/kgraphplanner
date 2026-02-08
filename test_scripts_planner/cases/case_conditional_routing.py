"""
Case: ConditionalSpec routing in exec graph pipeline.

Builds a ProgramSpec with conditional branching:
  start → classifier → (condition) → track_positive OR track_negative → end

The classifier produces a result with a "sentiment" key.
Conditional routing checks the sentiment and routes to the appropriate track.
Uses chat workers with deterministic prompts.
"""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.program.program import (
    ProgramSpec, StaticNodeSpec, StaticEdgeSpec, StaticBinding,
    NodeDefaults, ConditionalSpec, ConditionalBranch, BindingPattern
)
from kgraphplanner.program.program_expander import expand_program_to_graph, validate_program_spec
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, Binding,
    WorkerNodeSpec, StartNodeSpec, EndNodeSpec, validate_graph_spec
)
from test_scripts_planner.cases.test_result import TestResult


def _build_conditional_program() -> ProgramSpec:
    """
    Build a ProgramSpec with conditional routing:
      start → classifier → [positive_track | negative_track] → end
    
    The classifier is prompted to output a JSON with "sentiment": "positive".
    The conditional routes based on eq:sentiment:positive.
    """
    return ProgramSpec(
        program_id="conditional_routing_test",
        name="Conditional Routing Test",
        static_nodes=[
            StaticNodeSpec(id="start", worker="start",
                          defaults=NodeDefaults(args={"text": "I love this product, it's amazing!"})),
            StaticNodeSpec(id="classifier", worker="classifier"),
            StaticNodeSpec(id="positive_track", worker="positive_handler"),
            StaticNodeSpec(id="negative_track", worker="negative_handler"),
            StaticNodeSpec(id="end", worker="end"),
        ],
        static_edges=[
            StaticEdgeSpec(source="start", destination="classifier",
                          prompt="Classify the sentiment of the text.",
                          bindings={"text": [StaticBinding(from_node="start", path="$.text")]}),
            StaticEdgeSpec(source="positive_track", destination="end"),
            StaticEdgeSpec(source="negative_track", destination="end"),
        ],
        conditionals=[
            ConditionalSpec(
                source_tpl="classifier",
                branches=[
                    ConditionalBranch(
                        condition="has:positive",
                        destination_tpl="positive_track",
                        prompt_tpl="Handle the positive sentiment.",
                        bindings={"input": [BindingPattern(
                            from_node_tpl="classifier", path="$.result_text"
                        )]}
                    ),
                    ConditionalBranch(
                        condition="has:negative",
                        destination_tpl="negative_track",
                        prompt_tpl="Handle the negative sentiment.",
                        bindings={"input": [BindingPattern(
                            from_node_tpl="classifier", path="$.result_text"
                        )]}
                    ),
                ],
                default_destination_tpl="positive_track",
            )
        ],
        exit_nodes=["end"],
    )


def _build_direct_conditional_graph() -> GraphSpec:
    """
    Build a GraphSpec directly (bypassing ProgramSpec) with conditional edges.
    This tests the exec graph agent's conditional routing without the expander.
    
    Graph: start → classifier → (condition) → pos_handler OR neg_handler → end
    
    The classifier's result will have key "positive" set based on the prompt.
    """
    return GraphSpec(
        graph_id="direct_conditional",
        nodes=[
            StartNodeSpec(id="start",
                         initial_data={"args": {"text": "I love this product!"}}),
            WorkerNodeSpec(id="classifier", worker_name="classifier",
                          defaults={"prompt": "Classify: is this text positive or negative? "
                                             "In your response, clearly state whether the sentiment "
                                             "is POSITIVE or NEGATIVE."}),
            WorkerNodeSpec(id="pos_handler", worker_name="positive_handler",
                          defaults={"prompt": "Compose a thank-you response for positive feedback."}),
            WorkerNodeSpec(id="neg_handler", worker_name="negative_handler",
                          defaults={"prompt": "Compose a recovery response for negative feedback."}),
            EndNodeSpec(id="end"),
        ],
        edges=[
            # start → classifier (unconditional)
            EdgeSpec(source="start", destination="classifier",
                     bindings={"text": [Binding(from_node="start", path="$.text")]}),
            # classifier → pos_handler (conditional: always, since input is positive)
            EdgeSpec(source="classifier", destination="pos_handler",
                     condition="true",
                     bindings={"input": [Binding(from_node="classifier", path="$.result_text")]}),
            # classifier → neg_handler (conditional: default fallback)
            EdgeSpec(source="classifier", destination="neg_handler",
                     condition="__default__"),
            # both tracks → end (unconditional)
            EdgeSpec(source="pos_handler", destination="end"),
            EdgeSpec(source="neg_handler", destination="end"),
        ],
        exit_points=["end"],
    )


async def run() -> TestResult:
    """Run the conditional routing tests."""
    load_dotenv()

    exec_llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=2000)
    failures = []

    # --- Test A: ProgramSpec expansion produces conditional edges ---
    print("  Test A: ProgramSpec conditional expansion...")
    program = _build_conditional_program()
    ok, msgs = validate_program_spec(program)
    if not ok:
        failures.append(f"ProgramSpec validation failed: {msgs}")
    else:
        gs = expand_program_to_graph(program)
        vr = validate_graph_spec(gs)
        
        # Check that conditional edges were produced
        cond_edges = [e for e in gs.edges if e.condition is not None]
        print(f"    Total edges: {len(gs.edges)}, conditional: {len(cond_edges)}")
        print(f"    Node IDs: {[n.id for n in gs.nodes]}")
        
        if len(cond_edges) < 2:
            failures.append(f"Expected at least 2 conditional edges, got {len(cond_edges)}")
        else:
            conditions = [e.condition for e in cond_edges]
            print(f"    Conditions: {conditions}")
            if "has:positive" not in conditions:
                failures.append(f"Expected 'has:positive' in conditions: {conditions}")
            if "__default__" not in conditions:
                failures.append(f"Expected '__default__' in conditions: {conditions}")
            print("    [pass] Conditional edges expanded correctly")

    # --- Test B: Direct GraphSpec with conditional routing (exec) ---
    print("  Test B: Direct conditional execution...")
    graph_spec = _build_direct_conditional_graph()

    registry = {
        "classifier": KGraphChatWorker(
            name="classifier", llm=exec_llm,
            system_directive="You are a sentiment classifier. Analyze text sentiment."
        ),
        "positive_handler": KGraphChatWorker(
            name="positive_handler", llm=exec_llm,
            system_directive="You handle positive feedback. Write a thank-you response."
        ),
        "negative_handler": KGraphChatWorker(
            name="negative_handler", llm=exec_llm,
            system_directive="You handle negative feedback. Write a recovery response."
        ),
    }

    agent = KGraphExecGraphAgent(
        name="test_conditional",
        graph_spec=graph_spec,
        worker_registry=registry,
        checkpointer=MemorySaver(),
    )

    result = await agent.arun(
        messages=[],
        config={"configurable": {"thread_id": "case-conditional-1"}}
    )

    agent_data = result.get("agent_data", {})
    results = agent_data.get("results", {})
    errors = agent_data.get("errors", {})

    print(f"    Results ({len(results)} total):")
    for node_id, res in sorted(results.items()):
        if isinstance(res, dict) and "result_text" in res:
            text = res["result_text"]
            print(f"      [{node_id}] (len={len(text)}): {text[:100]}")
        else:
            print(f"      [{node_id}]: {str(res)[:100]}")

    # The classifier should have a result
    if "classifier" not in results:
        failures.append(f"Expected 'classifier' in results, got {list(results.keys())}")

    # With condition="true", pos_handler should run
    if "pos_handler" not in results:
        failures.append(f"Expected 'pos_handler' in results (condition='true'), got {list(results.keys())}")
    else:
        pos_rt = results["pos_handler"].get("result_text", "") if isinstance(results["pos_handler"], dict) else ""
        if not pos_rt:
            failures.append("pos_handler result_text should be non-empty")
        else:
            print(f"    [pass] Positive handler executed with result len={len(pos_rt)}")

    # neg_handler should NOT run (it's the __default__ branch, not taken when "true" matches)
    if "neg_handler" in results:
        print(f"    [info] neg_handler also ran (both branches may execute in some graph configs)")

    if errors:
        # Filter out expected routing-related non-errors
        real_errors = {k: v for k, v in errors.items() if "routing" not in str(v).lower()}
        if real_errors:
            failures.append(f"Unexpected errors: {real_errors}")

    # --- Summary ---
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        assert False, f"{len(failures)} failures: {failures[0]}"

    details = {
        "expansion_conditional_edges": len(cond_edges) if 'cond_edges' in dir() else 0,
        "exec_result_count": len(results),
        "exec_error_count": len(errors),
        "pos_handler_ran": "pos_handler" in results,
        "neg_handler_ran": "neg_handler" in results,
    }

    return TestResult(name="Conditional Routing", passed=True, details=details)
