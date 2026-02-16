"""
Case: Diagnostic plan-only — stops before execution to inspect bindings.

Uses the same worker registry and user prompt as case_planner_tool_research,
but only runs plan() and expand(). Dumps the full ProgramSpec and GraphSpec
with detailed binding inspection so we can diagnose data-flow issues:
  1. Are start → research_{i} edges passing the company name?
  2. Are fan-out edges (research → analyst) passing result_text?
  3. Are fan-in edges (analyst → aggregator) using append_list correctly?
"""

from __future__ import annotations

import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.program.program_expander import expand_program_to_graph
from test_scripts_planner.cases.test_result import TestResult
from test_scripts_planner.cases.case_helpers import create_tool_manager


def _make_registry(exec_llm, tool_manager):
    available_tools = tool_manager.list_available_tools()
    web_tool_ids = [t for t in available_tools if "web_search" in t]
    tool_ids = web_tool_ids if web_tool_ids else available_tools[:1]

    return {
        "research_worker": KGraphToolWorker(
            name="research_worker", llm=exec_llm,
            system_directive="Research the company using tools if needed; then finalize.",
            tool_manager=tool_manager, available_tool_ids=tool_ids,
        ),
        "analyst_a": KGraphChatWorker(
            name="analyst_a", llm=exec_llm,
            system_directive="Analyze input A; produce concise, structured notes; then finalize.",
        ),
        "analyst_b": KGraphChatWorker(
            name="analyst_b", llm=exec_llm,
            system_directive="Analyze input B; produce concise, structured notes; then finalize.",
        ),
        "aggregator": KGraphChatWorker(
            name="aggregator", llm=exec_llm,
            system_directive="Aggregate a list of analyses into a single summary; then finalize.",
        ),
    }


def _dump_program_spec(program):
    """Pretty-print the ProgramSpec focusing on bindings."""
    d = program.model_dump()

    print("\n  === PROGRAM SPEC ===")
    print(f"  program_id: {d['program_id']}")

    print(f"\n  --- Static Nodes ({len(d['static_nodes'])}) ---")
    for n in d['static_nodes']:
        defaults_str = ""
        if n.get('defaults'):
            defaults_str = f"  defaults={json.dumps(n['defaults'], default=str)[:200]}"
        print(f"    {n['id']} (worker={n['worker']}){defaults_str}")

    print(f"\n  --- Static Edges ({len(d['static_edges'])}) ---")
    for e in d['static_edges']:
        bindings_str = ""
        if e.get('bindings'):
            bindings_str = f"\n      bindings: {json.dumps(e['bindings'], default=str, indent=8)}"
        print(f"    {e['source']} → {e['destination']}"
              f"  prompt={(e.get('prompt') or '(none)')[:80]}{bindings_str}")

    print(f"\n  --- Templates ({len(d['templates'])}) ---")
    for t in d['templates']:
        fe = t.get('for_each', {})
        print(f"    Template: '{t['name']}'")
        print(f"      for_each: source_from={fe.get('source_from')}, "
              f"source_path={fe.get('source_path')}, "
              f"item_var={fe.get('item_var')}, idx_var={fe.get('idx_var')}")

        print(f"      loop_nodes ({len(t.get('loop_nodes', []))}):")
        for ln in t.get('loop_nodes', []):
            defaults_str = ""
            if ln.get('defaults_tpl'):
                defaults_str = f"  defaults_tpl={json.dumps(ln['defaults_tpl'], default=str)[:150]}"
            print(f"        {ln['id_tpl']} (worker={ln['worker']}){defaults_str}")

        print(f"      loop_edges ({len(t.get('loop_edges', []))}):")
        for le in t.get('loop_edges', []):
            bindings_str = ""
            if le.get('bindings'):
                bindings_str = f"\n          bindings: {json.dumps(le['bindings'], default=str, indent=12)}"
            print(f"        {le['source_tpl']} → {le['destination_tpl']}"
                  f"  prompt={(le.get('prompt_tpl') or '(none)')[:100]}{bindings_str}")

        print(f"      fan_out ({len(t.get('fan_out', []))}):")
        for fo in t.get('fan_out', []):
            print(f"        source: {fo['source_tpl']}")
            for bi, br in enumerate(fo.get('branches', [])):
                bindings_str = ""
                if br.get('bindings'):
                    bindings_str = f"\n            bindings: {json.dumps(br['bindings'], default=str, indent=14)}"
                print(f"          branch[{bi}]: → {br['destination_tpl']}"
                      f"  prompt={(br.get('prompt_tpl') or '(none)')[:80]}{bindings_str}")

        print(f"      fan_in ({len(t.get('fan_in', []))}):")
        for fi in t.get('fan_in', []):
            print(f"        sources: {fi['sources_tpl']} → {fi['destination_tpl']}")
            print(f"        param={fi.get('param')}, reduce={fi.get('reduce')}")
            print(f"        prompt={(fi.get('prompt_tpl') or '(none)')[:80]}")


def _dump_graph_spec(graph_spec):
    """Pretty-print the expanded GraphSpec focusing on edge bindings."""
    print("\n  === GRAPH SPEC (expanded) ===")
    print(f"  graph_id: {graph_spec.graph_id}")
    print(f"  exit_points: {graph_spec.exit_points}")

    print(f"\n  --- Nodes ({len(graph_spec.nodes)}) ---")
    for n in graph_spec.nodes:
        nd = n.model_dump()
        node_type = nd.get('node_type', '?')
        extras = ""
        if node_type == "worker":
            extras = f" worker_name={nd.get('worker_name')}"
            if nd.get('defaults'):
                extras += f" defaults={json.dumps(nd['defaults'], default=str)[:150]}"
        elif node_type == "start":
            extras = f" initial_data={json.dumps(nd.get('initial_data', {}), default=str)[:150]}"
        print(f"    [{nd['id']}] type={node_type}{extras}")

    print(f"\n  --- Edges ({len(graph_spec.edges)}) ---")
    issues = []
    for e in graph_spec.edges:
        cond_str = f" condition={e.condition}" if e.condition else ""
        print(f"    {e.source} → {e.destination}{cond_str}")
        if e.prompt:
            print(f"      prompt: {e.prompt[:100]}")
        if e.bindings:
            for param, binding_list in e.bindings.items():
                for b in binding_list:
                    bd = b.model_dump()
                    parts = []
                    if bd.get('from_node'):
                        parts.append(f"from_node={bd['from_node']}")
                    parts.append(f"path={bd.get('path', '$')}")
                    if bd.get('literal') is not None:
                        parts.append(f"literal={bd['literal']}")
                    if bd.get('reduce'):
                        parts.append(f"reduce={bd['reduce']}")
                    if bd.get('alias'):
                        parts.append(f"alias={bd['alias']}")
                    if bd.get('transform') and bd['transform'] != 'as_is':
                        parts.append(f"transform={bd['transform']}")
                    print(f"      binding[{param}]: {', '.join(parts)}")
        else:
            # Flag worker→worker edges without bindings
            if e.source != "start" and e.destination != "end":
                issues.append(f"Edge {e.source} → {e.destination} has NO bindings")

    if issues:
        print(f"\n  ⚠️  BINDING ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")

    return issues


async def run() -> TestResult:
    """Run plan-only diagnostic for the 5-company tool research scenario."""
    load_dotenv()

    planner_llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    exec_llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=2000)

    # Set up tools (needed for registry)
    try:
        tm = await create_tool_manager()
        available = tm.list_available_tools()
        print(f"  Available tools: {available}")
    except Exception as e:
        return TestResult(
            name="Diagnostic: Plan Only",
            passed=True,
            details={"skipped": True, "reason": f"ToolManager init failed: {e}"}
        )

    if not available:
        return TestResult(
            name="Diagnostic: Plan Only",
            passed=True,
            details={"skipped": True, "reason": "No tools available"}
        )

    registry = _make_registry(exec_llm, tm)

    agent = KGraphPlannerAgent(
        name="diagnostic_plan_only",
        planner_llm=planner_llm,
        worker_registry=registry,
        execution_llm=exec_llm,
    )

    # Same worker-agnostic prompt as the research case
    user_message = HumanMessage(
        content=(
            "Research 5 AI companies, analyze via two tracks per company, "
            "and produce a combined report."
        )
    )

    print(f"\n  User prompt: {user_message.content}")
    print(f"  Available workers: {list(registry.keys())}")

    # Step 1: Plan (LLM generates ProgramSpec)
    print("\n  --- Step 1: plan() ---")
    program = await agent.plan([user_message])
    _dump_program_spec(program)

    # Step 2: Expand (ProgramSpec → GraphSpec)
    print("\n  --- Step 2: expand() ---")
    graph_spec = agent.expand(program)
    binding_issues = _dump_graph_spec(graph_spec)

    # --- Targeted checks ---
    issues = list(binding_issues)

    # Check 1: start → research edges should pass company name
    print("\n  === DIAGNOSTIC CHECKS ===")
    research_edges = [e for e in graph_spec.edges
                      if e.source == "start" and "research" in e.destination]
    print(f"\n  Check 1: start → research edges ({len(research_edges)}):")
    for e in research_edges:
        has_company_binding = False
        for param, bindings in e.bindings.items():
            for b in bindings:
                if b.from_node == "start" and "compan" in (b.path or "").lower():
                    has_company_binding = True
                if b.from_node == "start":
                    print(f"    {e.destination}: param={param}, from_node={b.from_node}, path={b.path}")
        if not has_company_binding:
            issues.append(f"start → {e.destination}: no binding passes company name")
            print(f"    ⚠️  {e.destination}: NO company name binding found!")

    # Check 2: fan-out edges (research → analyst) should pass result_text
    print(f"\n  Check 2: research → analyst fan-out edges:")
    fanout_edges = [e for e in graph_spec.edges
                    if "research" in e.source and ("analyst" in e.destination or "analysis" in e.destination)]
    for e in fanout_edges:
        has_result_binding = False
        for param, bindings in e.bindings.items():
            for b in bindings:
                if "result" in (b.path or "").lower():
                    has_result_binding = True
                print(f"    {e.source} → {e.destination}: param={param}, "
                      f"from_node={b.from_node}, path={b.path}")
        if not e.bindings:
            issues.append(f"{e.source} → {e.destination}: NO bindings at all")
            print(f"    ⚠️  {e.source} → {e.destination}: NO bindings!")
        elif not has_result_binding:
            issues.append(f"{e.source} → {e.destination}: no binding for result_text")
            print(f"    ⚠️  {e.source} → {e.destination}: no result_text binding!")

    # Check 3: fan-in edges (analyst → aggregator) should use append_list
    print(f"\n  Check 3: analyst → aggregator fan-in edges:")
    agg_nodes = [n.id for n in graph_spec.nodes
                 if hasattr(n, 'worker_name') and n.worker_name and "aggregat" in n.worker_name]
    fanin_edges = [e for e in graph_spec.edges if e.destination in agg_nodes]
    for e in fanin_edges:
        for param, bindings in e.bindings.items():
            for b in bindings:
                reduce_str = b.reduce or "overwrite"
                print(f"    {e.source} → {e.destination}: param={param}, "
                      f"from_node={b.from_node}, path={b.path}, reduce={reduce_str}")
                if reduce_str != "append_list":
                    issues.append(f"{e.source} → {e.destination}: reduce={reduce_str} (expected append_list)")
    if not fanin_edges:
        issues.append("No fan-in edges found targeting aggregator")
        print(f"    ⚠️  No fan-in edges to aggregator!")

    # Summary
    print(f"\n  === SUMMARY ===")
    print(f"  ProgramSpec: {program.program_id}")
    print(f"  Templates: {len(program.templates)}")
    print(f"  GraphSpec: {len(graph_spec.nodes)} nodes, {len(graph_spec.edges)} edges")
    print(f"  Binding issues found: {len(issues)}")
    for i in issues:
        print(f"    ⚠️  {i}")

    details = {
        "program_id": program.program_id,
        "template_count": len(program.templates),
        "node_count": len(graph_spec.nodes),
        "edge_count": len(graph_spec.edges),
        "start_to_research_edges": len(research_edges),
        "fanout_edges": len(fanout_edges),
        "fanin_edges": len(fanin_edges),
        "binding_issues": len(issues),
        "issue_details": issues,
    }

    return TestResult(
        name="Diagnostic: Plan Only",
        passed=True,  # Always passes — it's diagnostic
        details=details
    )
