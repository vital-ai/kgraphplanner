"""
Case: Unit tests for program_expander.py

Tests template expansion, validation, and edge cases without any LLM calls.
Fast, deterministic, no external dependencies.
"""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from kgraphplanner.program.program import (
    ProgramSpec, StaticNodeSpec, StaticEdgeSpec, StaticBinding,
    NodeDefaults, TemplateSpec, ForEachSpec, NodePattern, EdgePattern,
    BindingPattern, FanOutSpec, FanOutBranch, FanInSpec
)
from kgraphplanner.program.program_expander import (
    expand_program_to_graph, validate_program_spec
)
from kgraphplanner.graph.exec_graph import validate_graph_spec
from test_scripts_planner.cases.test_result import TestResult


def _minimal_program(**overrides) -> ProgramSpec:
    """Build a minimal valid ProgramSpec with optional overrides."""
    defaults = dict(
        program_id="test",
        static_nodes=[
            StaticNodeSpec(id="start", worker="start"),
            StaticNodeSpec(id="end", worker="end"),
        ],
        static_edges=[],
        exit_nodes=["end"],
    )
    defaults.update(overrides)
    return ProgramSpec(**defaults)


async def run() -> TestResult:
    """Run all program_expander unit tests."""
    failures = []

    # --- validate_program_spec ---

    # 1. Minimal valid program
    ok, msgs = validate_program_spec(_minimal_program())
    if not ok:
        failures.append(f"minimal valid program rejected: {msgs}")

    # 2. Missing program_id
    try:
        p = ProgramSpec(program_id="", static_nodes=[
            StaticNodeSpec(id="start", worker="start"),
            StaticNodeSpec(id="end", worker="end"),
        ], exit_nodes=["end"])
        ok, msgs = validate_program_spec(p)
        if ok:
            failures.append("empty program_id should fail validation")
    except Exception:
        pass  # Expected

    # 3. Missing start node
    try:
        p = ProgramSpec(program_id="test", static_nodes=[
            StaticNodeSpec(id="end", worker="end"),
        ], exit_nodes=["end"])
        ok, msgs = validate_program_spec(p)
        if ok:
            failures.append("missing start node should fail validation")
    except Exception:
        pass

    # 4. Missing end node
    try:
        p = ProgramSpec(program_id="test", static_nodes=[
            StaticNodeSpec(id="start", worker="start"),
        ], exit_nodes=[])
        ok, msgs = validate_program_spec(p)
        if ok:
            failures.append("missing end node should fail validation")
    except Exception:
        pass

    # 5. Worker→worker edge without bindings
    p = _minimal_program(
        static_nodes=[
            StaticNodeSpec(id="start", worker="start"),
            StaticNodeSpec(id="w1", worker="worker_a"),
            StaticNodeSpec(id="w2", worker="worker_b"),
            StaticNodeSpec(id="end", worker="end"),
        ],
        static_edges=[
            StaticEdgeSpec(source="start", destination="w1"),
            StaticEdgeSpec(source="w1", destination="w2"),  # no bindings!
            StaticEdgeSpec(source="w2", destination="end"),
        ],
    )
    ok, msgs = validate_program_spec(p)
    if ok:
        failures.append("worker→worker edge without bindings should fail")
    else:
        print(f"  [pass] missing bindings detected: {msgs[-1][:80]}")

    # 6. Worker→worker edge WITH bindings passes
    p = _minimal_program(
        static_nodes=[
            StaticNodeSpec(id="start", worker="start"),
            StaticNodeSpec(id="w1", worker="worker_a"),
            StaticNodeSpec(id="w2", worker="worker_b"),
            StaticNodeSpec(id="end", worker="end"),
        ],
        static_edges=[
            StaticEdgeSpec(source="start", destination="w1"),
            StaticEdgeSpec(source="w1", destination="w2", bindings={
                "input": [StaticBinding(from_node="w1", path="$.result_text")]
            }),
            StaticEdgeSpec(source="w2", destination="end"),
        ],
    )
    ok, msgs = validate_program_spec(p)
    if not ok:
        failures.append(f"edge with bindings should pass: {msgs}")
    else:
        print("  [pass] edge with bindings validates")

    # 7. Template with empty expandable content
    p = _minimal_program(
        templates=[
            TemplateSpec(
                name="empty_template",
                for_each=ForEachSpec(source_from="literal", literal_items=["a"]),
                loop_nodes=[], loop_edges=[], fan_out=[], fan_in=[]
            )
        ]
    )
    ok, msgs = validate_program_spec(p)
    if ok:
        failures.append("template with no expandable content should fail")
    else:
        print(f"  [pass] empty template caught: {msgs[-1][:80]}")

    # --- expand_program_to_graph ---

    # 8. Static-only expansion (no templates)
    p = _minimal_program(
        static_nodes=[
            StaticNodeSpec(id="start", worker="start",
                          defaults=NodeDefaults(args={"topic": "AI"})),
            StaticNodeSpec(id="w1", worker="worker_a"),
            StaticNodeSpec(id="end", worker="end"),
        ],
        static_edges=[
            StaticEdgeSpec(source="start", destination="w1"),
            StaticEdgeSpec(source="w1", destination="end"),
        ],
    )
    gs = expand_program_to_graph(p)
    if len(gs.nodes) != 3:
        failures.append(f"static-only: expected 3 nodes, got {len(gs.nodes)}")
    if len(gs.edges) != 2:
        failures.append(f"static-only: expected 2 edges, got {len(gs.edges)}")
    print(f"  [pass] static-only: {len(gs.nodes)} nodes, {len(gs.edges)} edges")

    # 9. Template expansion with for_each literal items
    p = _minimal_program(
        static_nodes=[
            StaticNodeSpec(id="start", worker="start"),
            StaticNodeSpec(id="end", worker="end"),
        ],
        templates=[
            TemplateSpec(
                name="items",
                for_each=ForEachSpec(
                    source_from="literal",
                    literal_items=["alpha", "beta", "gamma"],
                    item_var="item", idx_var="i"
                ),
                loop_nodes=[
                    NodePattern(id_tpl="worker_{i}", worker="my_worker",
                                defaults_tpl=NodeDefaults(prompt="Process {item}"))
                ],
                loop_edges=[
                    EdgePattern(source_tpl="start", destination_tpl="worker_{i}",
                                prompt_tpl="Handle {item}")
                ]
            )
        ]
    )
    gs = expand_program_to_graph(p)
    worker_nodes = [n for n in gs.nodes if n.id.startswith("worker_")]
    if len(worker_nodes) != 3:
        failures.append(f"template literal: expected 3 workers, got {len(worker_nodes)}")
    else:
        ids = sorted(n.id for n in worker_nodes)
        if ids != ["worker_0", "worker_1", "worker_2"]:
            failures.append(f"template literal: unexpected IDs {ids}")
        else:
            print(f"  [pass] literal for_each: {ids}")

    # 10. Template expansion with start_args
    p = _minimal_program(
        static_nodes=[
            StaticNodeSpec(id="start", worker="start",
                          defaults=NodeDefaults(args={"items": ["x", "y"]})),
            StaticNodeSpec(id="end", worker="end"),
        ],
        templates=[
            TemplateSpec(
                name="from_args",
                for_each=ForEachSpec(
                    source_from="start_args",
                    source_path="$.items",
                    item_var="val", idx_var="idx"
                ),
                loop_nodes=[
                    NodePattern(id_tpl="node_{idx}", worker="w")
                ],
                loop_edges=[
                    EdgePattern(source_tpl="start", destination_tpl="node_{idx}")
                ]
            )
        ]
    )
    gs = expand_program_to_graph(p)
    worker_nodes = [n for n in gs.nodes if n.id.startswith("node_")]
    if len(worker_nodes) != 2:
        failures.append(f"start_args for_each: expected 2 workers, got {len(worker_nodes)}")
    else:
        print(f"  [pass] start_args for_each: {sorted(n.id for n in worker_nodes)}")

    # 11. Empty items list produces no expanded nodes
    p = _minimal_program(
        static_nodes=[
            StaticNodeSpec(id="start", worker="start",
                          defaults=NodeDefaults(args={"items": []})),
            StaticNodeSpec(id="end", worker="end"),
        ],
        templates=[
            TemplateSpec(
                name="empty",
                for_each=ForEachSpec(source_from="start_args", source_path="$.items"),
                loop_nodes=[NodePattern(id_tpl="w_{idx}", worker="w")],
                loop_edges=[EdgePattern(source_tpl="start", destination_tpl="w_{idx}")]
            )
        ]
    )
    gs = expand_program_to_graph(p)
    if len(gs.nodes) != 2:  # just start + end
        failures.append(f"empty items: expected 2 nodes, got {len(gs.nodes)}")
    else:
        print("  [pass] empty items: no expanded nodes")

    # 12. Fan-out expansion
    p = _minimal_program(
        static_nodes=[
            StaticNodeSpec(id="start", worker="start",
                          defaults=NodeDefaults(args={"companies": ["A"]})),
            StaticNodeSpec(id="end", worker="end"),
        ],
        templates=[
            TemplateSpec(
                name="fan",
                for_each=ForEachSpec(
                    source_from="start_args", source_path="$.companies",
                    item_var="co", idx_var="i"
                ),
                loop_nodes=[
                    NodePattern(id_tpl="research_{i}", worker="researcher"),
                    NodePattern(id_tpl="track_a_{i}", worker="analyst_a"),
                    NodePattern(id_tpl="track_b_{i}", worker="analyst_b"),
                ],
                loop_edges=[
                    EdgePattern(source_tpl="start", destination_tpl="research_{i}")
                ],
                fan_out=[FanOutSpec(
                    source_tpl="research_{i}",
                    branches=[
                        FanOutBranch(destination_tpl="track_a_{i}",
                                     prompt_tpl="Track A for {co}",
                                     bindings={"input": [BindingPattern(
                                         from_node_tpl="research_{i}", path="$.result_text"
                                     )]}),
                        FanOutBranch(destination_tpl="track_b_{i}",
                                     prompt_tpl="Track B for {co}"),
                    ]
                )]
            )
        ]
    )
    gs = expand_program_to_graph(p)
    # 1 company: start + end + research_0 + track_a_0 + track_b_0 = 5
    if len(gs.nodes) != 5:
        failures.append(f"fan-out: expected 5 nodes, got {len(gs.nodes)}: {[n.id for n in gs.nodes]}")
    # Edges: start→research_0 + research_0→track_a_0 + research_0→track_b_0 = 3
    fan_edges = [e for e in gs.edges if e.source == "research_0"]
    if len(fan_edges) != 2:
        failures.append(f"fan-out: expected 2 fan edges from research_0, got {len(fan_edges)}")
    else:
        # Check that bindings were expanded
        a_edge = next(e for e in fan_edges if e.destination == "track_a_0")
        if "input" not in a_edge.bindings:
            failures.append("fan-out: track_a binding missing 'input' key")
        else:
            b = a_edge.bindings["input"][0]
            if b.from_node != "research_0":
                failures.append(f"fan-out: expected from_node='research_0', got '{b.from_node}'")
            else:
                print("  [pass] fan-out expansion with bindings")

    # 13. Fan-in expansion
    p = _minimal_program(
        static_nodes=[
            StaticNodeSpec(id="start", worker="start",
                          defaults=NodeDefaults(args={"items": ["x"]})),
            StaticNodeSpec(id="agg", worker="aggregator"),
            StaticNodeSpec(id="end", worker="end"),
        ],
        static_edges=[StaticEdgeSpec(source="agg", destination="end")],
        templates=[
            TemplateSpec(
                name="fi",
                for_each=ForEachSpec(
                    source_from="start_args", source_path="$.items",
                    item_var="v", idx_var="i"
                ),
                loop_nodes=[
                    NodePattern(id_tpl="src_a_{i}", worker="wa"),
                    NodePattern(id_tpl="src_b_{i}", worker="wb"),
                ],
                loop_edges=[
                    EdgePattern(source_tpl="start", destination_tpl="src_a_{i}"),
                    EdgePattern(source_tpl="start", destination_tpl="src_b_{i}"),
                ],
                fan_in=[FanInSpec(
                    sources_tpl=["src_a_{i}", "src_b_{i}"],
                    destination_tpl="agg",
                    param="results",
                    reduce="append_list",
                    prompt_tpl="Merge {v}"
                )]
            )
        ]
    )
    gs = expand_program_to_graph(p)
    fan_in_edges = [e for e in gs.edges if e.destination == "agg" and e.source != "agg"]
    if len(fan_in_edges) != 2:
        failures.append(f"fan-in: expected 2 edges into agg, got {len(fan_in_edges)}")
    else:
        # Check reduce is set on bindings
        for e in fan_in_edges:
            if "results" not in e.bindings:
                failures.append(f"fan-in: missing 'results' binding on {e.source}→agg")
                break
            b = e.bindings["results"][0]
            if b.reduce != "append_list":
                failures.append(f"fan-in: expected reduce='append_list', got '{b.reduce}'")
                break
        else:
            print("  [pass] fan-in expansion with append_list reducer")

    # 14. start_seed merges into start args
    p = _minimal_program(
        static_nodes=[
            StaticNodeSpec(id="start", worker="start",
                          defaults=NodeDefaults(args={"base": "val"})),
            StaticNodeSpec(id="end", worker="end"),
        ],
        templates=[
            TemplateSpec(
                name="seeded",
                for_each=ForEachSpec(source_from="start_args", source_path="$.extra"),
                loop_nodes=[NodePattern(id_tpl="w_{idx}", worker="w")],
                loop_edges=[EdgePattern(source_tpl="start", destination_tpl="w_{idx}")]
            )
        ]
    )
    gs = expand_program_to_graph(p, start_seed={"extra": ["one", "two"]})
    worker_nodes = [n for n in gs.nodes if n.id.startswith("w_")]
    if len(worker_nodes) != 2:
        failures.append(f"start_seed: expected 2 workers, got {len(worker_nodes)}")
    else:
        print("  [pass] start_seed merges into start args")

    # 15. GraphSpec validation on expanded graph
    p = _minimal_program(
        static_nodes=[
            StaticNodeSpec(id="start", worker="start"),
            StaticNodeSpec(id="w1", worker="my_worker"),
            StaticNodeSpec(id="end", worker="end"),
        ],
        static_edges=[
            StaticEdgeSpec(source="start", destination="w1"),
            StaticEdgeSpec(source="w1", destination="end"),
        ],
    )
    gs = expand_program_to_graph(p)
    vr = validate_graph_spec(gs)
    if vr.has_errors():
        failures.append(f"graph validation errors: {[e.message for e in vr.errors]}")
    else:
        print("  [pass] expanded graph passes validation")

    # --- Summary ---
    total_tests = 15
    passed = total_tests - len(failures)
    print(f"\n  Subtests: {passed}/{total_tests} passed")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        assert False, f"{len(failures)} subtests failed: {failures[0]}"

    return TestResult(
        name="Unit: program_expander",
        passed=True,
        details={"subtests_passed": passed, "subtests_total": total_tests}
    )
