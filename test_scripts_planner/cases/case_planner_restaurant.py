"""
Case: LLM planner with tool-based restaurant research.

Same worker architecture as the AI-company research case, but with a
completely different domain prompt:

  "Recommend restaurants in 4 cities with different cuisines.
   For each city, research top restaurants, then analyze from
   two tracks (cuisine quality & ambiance/value), and produce
   a combined dining guide."

Workers are domain-agnostic — the planner LLM generates
ProgramSpec prompts that provide the restaurant/city context.

Outputs:
  - test_output/restaurant_program_spec.json
  - test_output/restaurant_graph_spec.json
  - test_output/restaurant_planner_graph.png
  - test_output/restaurant_execution_graph.png
  - test_output/restaurant_run.log

Requires: KGPLAN__ env vars, Keycloak credentials, tool server running.
If tools/auth unavailable, test is skipped.
"""
from __future__ import annotations

import os
import sys
import io
import json
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
from test_scripts_planner.cases.test_result import TestResult
from test_scripts_planner.cases.case_helpers import (
    OUTPUT_DIR,
    create_tool_manager,
    make_generic_registry,
    log as _log,
    write_log as _write_log,
    save_png as _save_png,
    execute_with_timing,
)


async def run() -> TestResult:
    """Run the LLM planner + tool restaurant recommendation end-to-end test."""
    load_dotenv()
    buf = io.StringIO()

    _log(buf, f"  Run started: {datetime.now().isoformat()}")

    # Check tool infrastructure first
    try:
        tm = await create_tool_manager()
        available = tm.list_available_tools()
        _log(buf, f"  Available tools: {available}")
    except Exception as e:
        return TestResult(
            name="Planner + Restaurant Recommendations",
            passed=True,
            details={"skipped": True, "reason": f"ToolManager init failed: {e}"}
        )

    if not available:
        return TestResult(
            name="Planner + Restaurant Recommendations",
            passed=True,
            details={"skipped": True, "reason": "No tools available (tool server not running?)"}
        )

    planner_llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    exec_llm = ChatOpenAI(model="gpt-5.2", temperature=0)

    registry = make_generic_registry(exec_llm, tm)

    agent = KGraphPlannerAgent(
        name="test_planner_restaurant",
        planner_llm=planner_llm,
        worker_registry=registry,
        execution_llm=exec_llm,
        checkpointer=MemorySaver()
    )

    # Worker-agnostic user message — different domain, same structure
    user_message = HumanMessage(
        content=(
            "Recommend restaurants in 4 cities (Tokyo, Paris, Mexico City, New York) "
            "with different cuisines for each. For each city, research top-rated "
            "restaurants, then analyze from two tracks: (A) cuisine quality and "
            "signature dishes, (B) ambiance, value and practical tips. "
            "Produce a combined dining guide."
        )
    )

    _log(buf, f"  User prompt: {user_message.content}")
    _log(buf, f"  Available workers: {list(registry.keys())}")

    # ---- Step 1: Plan ----
    _log(buf, "\n  === Step 1: Plan ===")
    program = await agent.plan([user_message])
    program_spec = program.model_dump()

    _log(buf, f"  program_id: {program_spec.get('program_id')}")
    _log(buf, f"  static_nodes: {len(program_spec.get('static_nodes', []))}")
    templates = program_spec.get("templates", [])
    _log(buf, f"  templates: {len(templates)}")
    for t in templates:
        _log(buf, f"    template '{t.get('name')}': "
              f"{len(t.get('loop_nodes', []))} loop_nodes, "
              f"{len(t.get('fan_out', []))} fan_out, "
              f"{len(t.get('fan_in', []))} fan_in")

    # Save ProgramSpec
    _log(buf, "\n  --- ProgramSpec (full JSON) ---")
    program_json = json.dumps(program_spec, indent=2, default=str)
    _log(buf, program_json)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "restaurant_program_spec.json"), "w") as f:
        f.write(program_json)
    _log(buf, f"  Saved: {OUTPUT_DIR}/restaurant_program_spec.json")

    # ---- Step 2: Expand ----
    _log(buf, "\n  === Step 2: Expand ===")
    graph_spec = agent.expand(program)
    graph_spec_dict = graph_spec.model_dump()

    node_ids = [n.get("id") for n in graph_spec_dict.get("nodes", [])]
    _log(buf, f"  GraphSpec: {len(graph_spec_dict.get('nodes', []))} nodes, "
          f"{len(graph_spec_dict.get('edges', []))} edges")
    _log(buf, f"  Node IDs: {node_ids}")

    # Derive expected result keys from the plan
    worker_node_ids = [
        n["id"] for n in graph_spec_dict.get("nodes", [])
        if n.get("node_type") == "worker"
    ]
    end_edges = [e for e in graph_spec_dict.get("edges", []) if e["destination"] == "end"]
    aggregator_ids = [e["source"] for e in end_edges]

    _log(buf, f"  Expected worker nodes: {worker_node_ids}")
    _log(buf, f"  Aggregator node(s) (→ end): {aggregator_ids}")

    # Dump edges
    _log(buf, "\n  --- Edges ---")
    for e in graph_spec_dict.get("edges", []):
        bindings_summary = ""
        if e.get("bindings"):
            bkeys = list(e["bindings"].keys())
            bindings_summary = f"  bindings={bkeys}"
        _log(buf, f"    {e['source']} → {e['destination']}{bindings_summary}")

    # Save GraphSpec
    _log(buf, "\n  --- GraphSpec (full JSON) ---")
    graph_json = json.dumps(graph_spec_dict, indent=2, default=str)
    _log(buf, graph_json)

    with open(os.path.join(OUTPUT_DIR, "restaurant_graph_spec.json"), "w") as f:
        f.write(graph_json)
    _log(buf, f"  Saved: {OUTPUT_DIR}/restaurant_graph_spec.json")

    # ---- Step 3: Generate diagrams ----
    _log(buf, "\n  === Step 3: Generate Diagrams ===")

    planner_graph = agent.build_graph()
    planner_compiled = planner_graph.compile(checkpointer=agent.checkpointer)
    await _save_png(planner_compiled, "restaurant_planner_graph.png", buf)

    exec_state_graph = agent._build_exec_graph(graph_spec)
    compiled = exec_state_graph.compile(checkpointer=agent.checkpointer)
    await _save_png(compiled, "restaurant_execution_graph.png", buf)

    # ---- Step 4: Execute ----
    _log(buf, "\n  === Step 4: Execute ===")
    initial_state = {
        "messages": [user_message],
        "agent_data": {},
        "work": {}
    }

    result = await execute_with_timing(
        compiled, initial_state, "case-planner-restaurant-1", buf
    )

    agent_data = result.get("agent_data", {})
    results = agent_data.get("results", {})
    errors = agent_data.get("errors", {})

    # Log results
    _log(buf, f"\n  === Results ({len(results)} total) ===")
    for node_id, res in sorted(results.items()):
        if isinstance(res, dict) and "result_text" in res:
            text = res["result_text"]
            _log(buf, f"    [{node_id}] (len={len(text)}):")
            _log(buf, f"      {text[:300]}")
        else:
            _log(buf, f"    [{node_id}]: {str(res)[:200]}")

    if errors:
        _log(buf, f"\n  === Errors ({len(errors)}) ===")
        for node_id, err in errors.items():
            _log(buf, f"    [{node_id}]: {err}")

    # Dump full results
    _log(buf, "\n  --- Full results JSON ---")
    _log(buf, json.dumps(results, indent=2, default=str))

    # Write log
    _log(buf, f"\n  Run finished: {datetime.now().isoformat()}")
    _write_log(buf, "restaurant_run.log")

    # --- Assertions (plan-derived) ---
    assert program_spec is not None, "Expected program_spec"
    assert graph_spec_dict is not None, "Expected graph_spec"
    assert len(templates) >= 1, f"Expected at least 1 template, got {len(templates)}"

    has_fan_out = any(len(t.get("fan_out", [])) > 0 for t in templates)
    has_fan_in = any(len(t.get("fan_in", [])) > 0 for t in templates)

    # Check every worker node from the plan produced a result
    missing = [nid for nid in worker_node_ids if nid not in results]
    assert not missing, \
        f"Plan expected results for {worker_node_ids}, missing: {missing} (got: {list(results.keys())})"

    # Check aggregator produced a result
    agg_missing = [nid for nid in aggregator_ids if nid not in results]
    assert not agg_missing, \
        f"Aggregator node(s) {aggregator_ids} missing from results (got: {list(results.keys())})"

    _log(buf, f"  Worker results: {len([n for n in worker_node_ids if n in results])}")

    details = {
        "program_id": program_spec.get("program_id"),
        "template_count": len(templates),
        "has_fan_out": has_fan_out,
        "has_fan_in": has_fan_in,
        "node_count": len(graph_spec_dict.get("nodes", [])),
        "edge_count": len(graph_spec_dict.get("edges", [])),
        "expected_worker_nodes": len(worker_node_ids),
        "result_count": len(results),
        "worker_result_count": len([n for n in worker_node_ids if n in results]),
        "error_count": len(errors),
        "tools_used": available,
        "output_dir": OUTPUT_DIR,
    }

    return TestResult(
        name="Planner + Restaurant Recommendations",
        passed=True,
        details=details
    )
