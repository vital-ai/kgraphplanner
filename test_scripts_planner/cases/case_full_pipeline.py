"""
Case: Deterministic full pipeline (no LLM planning).
Pre-built ProgramSpec with templates, fan-out, fan-in, bindings.
Exercises the same graph structure as the original test_planner.py:

  start → [per company: research → fan-out(analyst_a, analyst_b) → fan-in(aggregator)] → end

Tests: for_each templates, loop_nodes, loop_edges, fan_out, fan_in,
       binding resolution, activation merging, multi-worker execution.
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
    ProgramSpec, StaticNodeSpec, StaticEdgeSpec,
    NodeDefaults, TemplateSpec, ForEachSpec, NodePattern, EdgePattern,
    BindingPattern, FanOutSpec, FanOutBranch, FanInSpec
)
from kgraphplanner.program.program_expander import expand_program_to_graph, validate_program_spec
from kgraphplanner.graph.exec_graph import validate_graph_spec
from test_scripts_planner.cases.test_result import TestResult


def build_company_analysis_program() -> ProgramSpec:
    """
    Build a ProgramSpec matching the original test_planner.py demo:
      - 3 companies (reduced from 5 for speed)
      - Per company: research → fan-out to analyst_a + analyst_b → fan-in to aggregator
      - aggregator → end
    """
    return ProgramSpec(
        program_id="company_analysis",
        name="Multi-Company Analysis Pipeline",
        static_nodes=[
            StaticNodeSpec(
                id="start", worker="start",
                defaults=NodeDefaults(args={"companies": ["OpenAI", "Anthropic", "Cohere"]})
            ),
            StaticNodeSpec(
                id="aggregator_node", worker="aggregator",
                defaults=NodeDefaults(prompt="Aggregate all analyses into a combined summary report.")
            ),
            StaticNodeSpec(id="end", worker="end")
        ],
        static_edges=[
            StaticEdgeSpec(source="aggregator_node", destination="end")
        ],
        templates=[
            TemplateSpec(
                name="per_company",
                for_each=ForEachSpec(
                    source_from="start_args",
                    source_path="$.companies",
                    item_var="company",
                    idx_var="idx"
                ),
                loop_nodes=[
                    NodePattern(
                        id_tpl="research_{idx}",
                        worker="research_worker",
                        defaults_tpl=NodeDefaults(
                            prompt="Research {company} thoroughly.",
                            args={"topic": "{company}"}
                        )
                    ),
                    NodePattern(id_tpl="analysis_a_{idx}", worker="analyst_a"),
                    NodePattern(id_tpl="analysis_b_{idx}", worker="analyst_b")
                ],
                loop_edges=[
                    EdgePattern(
                        source_tpl="start",
                        destination_tpl="research_{idx}",
                        prompt_tpl="Research the company {company}",
                        bindings={
                            "company": [BindingPattern(
                                from_node_tpl="start",
                                path="$.companies[{idx}]",
                                reduce="overwrite"
                            )]
                        }
                    )
                ],
                fan_out=[
                    FanOutSpec(
                        source_tpl="research_{idx}",
                        branches=[
                            FanOutBranch(
                                destination_tpl="analysis_a_{idx}",
                                prompt_tpl="Analyze track A for {company}: strengths and products",
                                bindings={
                                    "input": [BindingPattern(
                                        from_node_tpl="research_{idx}",
                                        path="$.result_text"
                                    )]
                                }
                            ),
                            FanOutBranch(
                                destination_tpl="analysis_b_{idx}",
                                prompt_tpl="Analyze track B for {company}: risks and competition",
                                bindings={
                                    "input": [BindingPattern(
                                        from_node_tpl="research_{idx}",
                                        path="$.result_text"
                                    )]
                                }
                            )
                        ]
                    )
                ],
                fan_in=[
                    FanInSpec(
                        sources_tpl=["analysis_a_{idx}", "analysis_b_{idx}"],
                        destination_tpl="aggregator_node",
                        param="analyses",
                        reduce="append_list",
                        prompt_tpl="Include analysis for {company}"
                    )
                ]
            )
        ],
        exit_nodes=["end"],
        max_parallel=3
    )


def make_full_registry(exec_llm):
    """Create a worker registry matching the original test_planner.py demo."""
    return {
        "research_worker": KGraphChatWorker(
            name="research_worker", llm=exec_llm,
            system_directive="Research the company using available knowledge; produce factual notes."
        ),
        "analyst_a": KGraphChatWorker(
            name="analyst_a", llm=exec_llm,
            system_directive="Analyze input A; produce concise structured notes on strengths and products."
        ),
        "analyst_b": KGraphChatWorker(
            name="analyst_b", llm=exec_llm,
            system_directive="Analyze input B; produce concise structured notes on risks and competition."
        ),
        "aggregator": KGraphChatWorker(
            name="aggregator", llm=exec_llm,
            system_directive="Aggregate a list of analyses into a single combined summary report."
        )
    }


EXPECTED_WORKERS = [
    "research_0", "research_1", "research_2",
    "analysis_a_0", "analysis_a_1", "analysis_a_2",
    "analysis_b_0", "analysis_b_1", "analysis_b_2",
    "aggregator_node"
]


async def run() -> TestResult:
    """Run the deterministic full pipeline test."""
    load_dotenv()
    exec_llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=2000)

    # Step 1: Build and validate ProgramSpec
    program = build_company_analysis_program()
    is_valid, messages = validate_program_spec(program)
    print(f"  ProgramSpec valid: {is_valid}")
    for m in messages:
        print(f"    {m}")
    assert is_valid, f"ProgramSpec validation failed: {messages}"

    # Step 2: Expand
    graph_spec = expand_program_to_graph(program)
    validation = validate_graph_spec(graph_spec)
    node_ids = [n.id for n in graph_spec.nodes]
    print(f"  GraphSpec: {len(graph_spec.nodes)} nodes, {len(graph_spec.edges)} edges")
    print(f"  Node IDs: {node_ids}")
    if validation.has_errors():
        for e in validation.errors:
            print(f"    ERROR: {e.message}")
    if validation.has_warnings():
        for w in validation.warnings:
            print(f"    WARN: {w.message}")

    expected_node_count = 3 + 3 * 3  # start+end+aggregator + 3 companies * 3 workers
    assert len(graph_spec.nodes) == expected_node_count, \
        f"Expected {expected_node_count} nodes, got {len(graph_spec.nodes)}: {node_ids}"

    # Step 3: Execute
    registry = make_full_registry(exec_llm)
    agent = KGraphExecGraphAgent(
        name="test_exec_full",
        graph_spec=graph_spec,
        worker_registry=registry,
        checkpointer=MemorySaver()
    )

    result = await agent.arun(
        messages=[],
        config={"configurable": {"thread_id": "case-full-pipeline-1"}}
    )

    agent_data = result.get("agent_data", {})
    results = agent_data.get("results", {})
    errors = agent_data.get("errors", {})

    print(f"  Results ({len(results)} total):")
    for node_id, res in sorted(results.items()):
        if isinstance(res, dict) and "result_text" in res:
            text = res["result_text"]
            print(f"    [{node_id}] (len={len(text)}): {text[:100]}")
        else:
            print(f"    [{node_id}]: {str(res)[:100]}")

    if errors:
        print(f"  Errors:")
        for node_id, err in errors.items():
            print(f"    [{node_id}]: {err}")

    # Verify all expected workers have non-empty results
    for w in EXPECTED_WORKERS:
        assert w in results, f"Expected '{w}' in results, got keys: {list(results.keys())}"
        if isinstance(results[w], dict):
            rt = results[w].get("result_text", "")
            assert len(rt) > 0, f"Expected non-empty result_text for '{w}', got len={len(rt)}"

    assert len(errors) == 0, f"Expected no errors, got {errors}"

    # Fan-in verification: check the aggregator's activation received data
    # from all analyst workers.  The gather node should have accumulated
    # all fan-in bindings (reduce=append_list) into the activation args.
    agg_activation = agent_data.get("activation", {}).get("aggregator_node", {})
    agg_args = agg_activation.get("args", {})
    analyses_list = agg_args.get("analyses", [])
    # 3 companies × 2 analysts = 6 fan-in edges
    print(f"  Fan-in check: aggregator received {len(analyses_list)} analyses (expected 6)")
    assert len(analyses_list) == 6, \
        f"Fan-in data loss: aggregator received {len(analyses_list)} analyses, expected 6"

    details = {
        "node_count": len(graph_spec.nodes),
        "edge_count": len(graph_spec.edges),
        "result_count": len(results),
        "error_count": len(errors),
        "verified_workers": EXPECTED_WORKERS,
        "fan_in_analyses_count": len(analyses_list),
    }

    return TestResult(
        name="Full Pipeline (templates + fan-out/fan-in)",
        passed=True, details=details
    )
