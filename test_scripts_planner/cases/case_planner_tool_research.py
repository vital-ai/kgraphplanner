"""
Case: LLM planner with tool-based research worker (mirrors original demo).

Closely follows the test_planner.py demo():
  - research_worker: KGraphToolWorker with web search (tool-based research)
  - analyst_a: KGraphChatWorker (strengths/products track)
  - analyst_b: KGraphChatWorker (risks/competition track)
  - aggregator: KGraphChatWorker (combines all analyses)

The user prompt is worker-agnostic — it just says what it wants:
  "Research 5 AI companies, analyze via two tracks per company,
   and produce a combined report."

The planner LLM decides which workers to use and creates the ProgramSpec
(templates, fan-out, fan-in) without the user knowing the worker names.

Outputs:
  - test_output/research_planner_graph.png   (planner graph diagram)
  - test_output/research_execution_graph.png (execution graph diagram)
  - test_output/research_run.log             (full run log)

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
from langchain_core.runnables.graph import MermaidDrawMethod

from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.tool_manager.tool_manager import ToolManager
from test_scripts_planner.cases.test_result import TestResult
from test_scripts_planner.cases.case_helpers import create_tool_manager

OUTPUT_DIR = os.path.join(project_root, "test_output")


def _make_registry(exec_llm, tool_manager):
    """
    Build worker registry matching the original demo:
      - research_worker: tool worker with web search
      - analyst_a: chat worker (strengths/products)
      - analyst_b: chat worker (risks/competition)
      - aggregator: chat worker (combine analyses)
    """
    available_tools = tool_manager.list_available_tools()
    # Use web search if available, otherwise all tools
    web_tool_ids = [t for t in available_tools if "web_search" in t]
    tool_ids = web_tool_ids if web_tool_ids else available_tools[:1]

    return {
        "research_worker": KGraphToolWorker(
            name="research_worker",
            llm=exec_llm,
            system_directive=(
                "Research the company using tools if needed; "
                "then finalize with factual notes about the company."
            ),
            tool_manager=tool_manager,
            available_tool_ids=tool_ids,
        ),
        "analyst_a": KGraphChatWorker(
            name="analyst_a",
            llm=exec_llm,
            system_directive=(
                "Analyze input A; produce concise, structured notes "
                "on strengths and products; then finalize."
            ),
        ),
        "analyst_b": KGraphChatWorker(
            name="analyst_b",
            llm=exec_llm,
            system_directive=(
                "Analyze input B; produce concise, structured notes "
                "on risks and competition; then finalize."
            ),
        ),
        "aggregator": KGraphChatWorker(
            name="aggregator",
            llm=exec_llm,
            system_directive=(
                "Aggregate a list of analyses into a single combined "
                "summary report; then finalize."
            ),
        ),
    }


def _log(log: io.StringIO, msg: str):
    """Print and write to log buffer."""
    print(msg)
    log.write(msg + "\n")


def _write_log(log: io.StringIO):
    """Flush log buffer to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "research_run.log")
    with open(path, "w") as f:
        f.write(log.getvalue())
    print(f"  Log written to {path}")


async def _save_png(compiled_graph, filename: str, log: io.StringIO):
    """Generate and save a Mermaid PNG diagram."""
    import asyncio
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    try:
        graph_drawable = compiled_graph.get_graph()
        # Use API method to avoid pyppeteer Chrome atexit cleanup errors
        image_bytes = await asyncio.to_thread(
            graph_drawable.draw_mermaid_png,
            draw_method=MermaidDrawMethod.API
        )
        with open(path, "wb") as f:
            f.write(image_bytes)
        _log(log, f"  Diagram saved: {path} ({len(image_bytes)} bytes)")
    except Exception as e:
        _log(log, f"  Diagram generation failed ({filename}): {e}")


async def run() -> TestResult:
    """Run the LLM planner + tool research worker end-to-end test."""
    load_dotenv()
    log = io.StringIO()

    _log(log, f"  Run started: {datetime.now().isoformat()}")

    # Check tool infrastructure first
    try:
        tm = await create_tool_manager()
        available = tm.list_available_tools()
        _log(log, f"  Available tools: {available}")
    except Exception as e:
        return TestResult(
            name="Planner + Tool Research (5 companies)",
            passed=True,
            details={"skipped": True, "reason": f"ToolManager init failed: {e}"}
        )

    if not available:
        return TestResult(
            name="Planner + Tool Research (5 companies)",
            passed=True,
            details={"skipped": True, "reason": "No tools available (tool server not running?)"}
        )

    planner_llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    exec_llm = ChatOpenAI(model="gpt-5.2", temperature=0)

    registry = _make_registry(exec_llm, tm)

    agent = KGraphPlannerAgent(
        name="test_planner_tool_research",
        planner_llm=planner_llm,
        worker_registry=registry,
        execution_llm=exec_llm,
        checkpointer=MemorySaver()
    )

    # Worker-agnostic user message — just says what it wants
    user_message = HumanMessage(
        content=(
            "Research 5 AI companies, analyze via two tracks per company, "
            "and produce a combined report."
        )
    )

    _log(log, f"  User prompt: {user_message.content}")
    _log(log, f"  Available workers: {list(registry.keys())}")

    # ---- Step 1: Plan ----
    _log(log, "\n  === Step 1: Plan ===")
    program = await agent.plan([user_message])
    program_spec = program.model_dump()

    _log(log, f"  program_id: {program_spec.get('program_id')}")
    _log(log, f"  static_nodes: {len(program_spec.get('static_nodes', []))}")
    templates = program_spec.get("templates", [])
    _log(log, f"  templates: {len(templates)}")
    for t in templates:
        _log(log, f"    template '{t.get('name')}': "
              f"{len(t.get('loop_nodes', []))} loop_nodes, "
              f"{len(t.get('fan_out', []))} fan_out, "
              f"{len(t.get('fan_in', []))} fan_in")

    # Dump full ProgramSpec to log and save as file
    _log(log, "\n  --- ProgramSpec (full JSON) ---")
    program_json = json.dumps(program_spec, indent=2, default=str)
    _log(log, program_json)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "program_spec.json"), "w") as f:
        f.write(program_json)
    _log(log, f"  Saved: {OUTPUT_DIR}/program_spec.json")

    # ---- Step 2: Expand ----
    _log(log, "\n  === Step 2: Expand ===")
    graph_spec = agent.expand(program)
    graph_spec_dict = graph_spec.model_dump()

    node_ids = [n.get("id") for n in graph_spec_dict.get("nodes", [])]
    _log(log, f"  GraphSpec: {len(graph_spec_dict.get('nodes', []))} nodes, "
          f"{len(graph_spec_dict.get('edges', []))} edges")
    _log(log, f"  Node IDs: {node_ids}")

    # Derive expected result keys from the plan
    worker_node_ids = [
        n["id"] for n in graph_spec_dict.get("nodes", [])
        if n.get("node_type") == "worker"
    ]
    exit_point_ids = graph_spec_dict.get("exit_points", [])
    # The aggregator is the exit point (or the node right before "end")
    end_edges = [e for e in graph_spec_dict.get("edges", []) if e["destination"] == "end"]
    aggregator_ids = [e["source"] for e in end_edges]

    _log(log, f"  Expected worker nodes: {worker_node_ids}")
    _log(log, f"  Aggregator node(s) (→ end): {aggregator_ids}")

    # Dump edges with bindings to log
    _log(log, "\n  --- Edges ---")
    for e in graph_spec_dict.get("edges", []):
        bindings_summary = ""
        if e.get("bindings"):
            bkeys = list(e["bindings"].keys())
            bindings_summary = f"  bindings={bkeys}"
        _log(log, f"    {e['source']} → {e['destination']}{bindings_summary}")

    # Save full GraphSpec as file
    _log(log, "\n  --- GraphSpec (full JSON) ---")
    graph_json = json.dumps(graph_spec_dict, indent=2, default=str)
    _log(log, graph_json)

    with open(os.path.join(OUTPUT_DIR, "graph_spec.json"), "w") as f:
        f.write(graph_json)
    _log(log, f"  Saved: {OUTPUT_DIR}/graph_spec.json")

    # ---- Step 3: Generate diagrams BEFORE execution ----
    _log(log, "\n  === Step 3: Generate Diagrams ===")

    # Program-level graph (from planner agent)
    planner_graph = agent.build_graph()
    planner_compiled = planner_graph.compile(checkpointer=agent.checkpointer)
    await _save_png(planner_compiled, "research_planner_graph.png", log)

    # Execution graph (from exec graph agent)
    exec_state_graph = agent._build_exec_graph(graph_spec)
    compiled = exec_state_graph.compile(checkpointer=agent.checkpointer)
    await _save_png(compiled, "research_execution_graph.png", log)

    # ---- Step 4: Execute (with stdout capture for timing) ----
    _log(log, "\n  === Step 4: Execute ===")
    initial_state = {
        "messages": [user_message],
        "agent_data": {},
        "work": {}
    }

    # Capture both stdout (print) and logging (logger.info) into the log buffer
    import sys
    import logging as _logging

    class _Tee:
        def __init__(self, original, buf):
            self._orig = original
            self._buf = buf
        def write(self, data):
            self._orig.write(data)
            self._buf.write(data)
        def flush(self):
            self._orig.flush()

    _old_stdout = sys.stdout
    sys.stdout = _Tee(_old_stdout, log)

    # Add a logging handler that writes to the log buffer
    _log_handler = _logging.StreamHandler(log)
    _log_handler.setLevel(_logging.INFO)
    _log_handler.setFormatter(_logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    _logging.getLogger().addHandler(_log_handler)

    try:
        result = await compiled.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": "case-planner-tool-research-1"}}
        )
    finally:
        sys.stdout = _old_stdout
        _logging.getLogger().removeHandler(_log_handler)

    agent_data = result.get("agent_data", {})
    results = agent_data.get("results", {})
    errors = agent_data.get("errors", {})

    # Log results
    _log(log, f"\n  === Results ({len(results)} total) ===")
    for node_id, res in sorted(results.items()):
        if isinstance(res, dict) and "result_text" in res:
            text = res["result_text"]
            _log(log, f"    [{node_id}] (len={len(text)}):")
            _log(log, f"      {text[:300]}")
        else:
            _log(log, f"    [{node_id}]: {str(res)[:200]}")

    if errors:
        _log(log, f"\n  === Errors ({len(errors)}) ===")
        for node_id, err in errors.items():
            _log(log, f"    [{node_id}]: {err}")

    # Dump full results to log
    _log(log, "\n  --- Full results JSON ---")
    _log(log, json.dumps(results, indent=2, default=str))

    # Write log file
    _log(log, f"\n  Run finished: {datetime.now().isoformat()}")
    _write_log(log)

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

    # Check aggregator (→ end) produced a result
    agg_missing = [nid for nid in aggregator_ids if nid not in results]
    assert not agg_missing, \
        f"Aggregator node(s) {aggregator_ids} missing from results (got: {list(results.keys())})"

    _log(log, f"  Research worker results: {len([n for n in worker_node_ids if n in results])}")

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
        name="Planner + Tool Research (5 companies)",
        passed=True,
        details=details
    )
