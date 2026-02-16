"""
Case: parse_json_response — verify KGraphChatWorker JSON parsing + conditional routing.

Tests the parse_json_response flag added in Gap #2:
  1. A chat worker with parse_json_response=True is instructed to output JSON.
  2. Parsed fields (action, event_code, etc.) become top-level keys in the
     result dict alongside result_text.
  3. A downstream conditional edge routes based on a parsed field.
  4. Two different target workers confirm the correct branch was taken.

Pipeline:
  classifier (parse_json_response=True)
      ├── action=="alpha" → handler_alpha
      └── action=="beta"  → handler_beta
"""

from __future__ import annotations

import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, Binding,
    WorkerNodeSpec, StartNodeSpec, EndNodeSpec,
)
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import log, write_log, save_png, OUTPUT_DIR

# ── Test requests ──────────────────────────────────────────────

TEST_REQUESTS = [
    {
        "message": "The customer has a pending application and wants a status update.",
        "expected_action": "alpha",
        "label": "Should route to alpha (application status)",
    },
    {
        "message": "The customer is asking about new product rates and promotions.",
        "expected_action": "beta",
        "label": "Should route to beta (product inquiry)",
    },
]


# ── Workers ────────────────────────────────────────────────────

def _build_workers(llm):
    classifier = KGraphChatWorker(
        name="classifier",
        llm=llm,
        parse_json_response=True,
        system_directive=(
            "You are a request classifier.  Given a customer message, decide "
            "whether it is about an existing application/loan (action=alpha) or "
            "about new products/general questions (action=beta).\n\n"
            "You MUST respond with ONLY a JSON object, no other text:\n"
            '{"action": "alpha", "reason": "..."}\n'
            "or\n"
            '{"action": "beta", "reason": "..."}\n'
        ),
    )

    handler_alpha = KGraphChatWorker(
        name="handler_alpha",
        llm=llm,
        system_directive="You handle application/loan status requests. Respond briefly.",
    )

    handler_beta = KGraphChatWorker(
        name="handler_beta",
        llm=llm,
        system_directive="You handle product and general inquiry requests. Respond briefly.",
    )

    return {"classifier": classifier, "handler_alpha": handler_alpha, "handler_beta": handler_beta}


# ── Graph spec ─────────────────────────────────────────────────

def _build_graph_spec():
    nodes = [
        StartNodeSpec(
            id="start",
            initial_data={"args": {"input": ""}},
        ),
        WorkerNodeSpec(
            id="classifier",
            worker_name="classifier",
            defaults={"prompt": "Classify this customer request.", "args": {}},
        ),
        WorkerNodeSpec(
            id="handler_alpha",
            worker_name="handler_alpha",
            defaults={"prompt": "Handle this application/loan status request.", "args": {}},
        ),
        WorkerNodeSpec(
            id="handler_beta",
            worker_name="handler_beta",
            defaults={"prompt": "Handle this product/general inquiry.", "args": {}},
        ),
        EndNodeSpec(id="end"),
    ]

    edges = [
        # start → classifier
        EdgeSpec(
            source="start",
            destination="classifier",
            bindings={
                "input": [Binding(from_node="start", path="$.input")],
            },
        ),
        # classifier → handler_alpha (conditional: action == "alpha")
        EdgeSpec(
            source="classifier",
            destination="handler_alpha",
            prompt="Handle this application/loan status request.",
            condition="result.get('action') == 'alpha'",
            bindings={
                "classification": [Binding(from_node="classifier", path="$.result_text", transform="text")],
                "reason": [Binding(from_node="classifier", path="$.reason", transform="text")],
            },
        ),
        # classifier → handler_beta (conditional: action == "beta")
        EdgeSpec(
            source="classifier",
            destination="handler_beta",
            prompt="Handle this product/general inquiry.",
            condition="result.get('action') == 'beta'",
            bindings={
                "classification": [Binding(from_node="classifier", path="$.result_text", transform="text")],
                "reason": [Binding(from_node="classifier", path="$.reason", transform="text")],
            },
        ),
        # both handlers → end
        EdgeSpec(source="handler_alpha", destination="end"),
        EdgeSpec(source="handler_beta", destination="end"),
    ]

    return GraphSpec(
        graph_id="parse_json_test",
        nodes=nodes,
        edges=edges,
        exit_points=["handler_alpha", "handler_beta"],
    )


# ── Build + run ────────────────────────────────────────────────

def _build_agent(workers, graph_spec, checkpointer):
    return KGraphExecGraphAgent(
        name="parse_json_test",
        graph_spec=graph_spec,
        worker_registry=workers,
        checkpointer=checkpointer,
    )


async def run() -> TestResult:
    load_dotenv()
    buf = io.StringIO()
    log(buf, "  === parse_json_response test ===")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    workers = _build_workers(llm)
    graph_spec = _build_graph_spec()
    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)

    agent = _build_agent(workers, graph_spec, checkpointer)
    compiled = agent.get_compiled_graph()
    await save_png(compiled, "parse_json_test_graph.png", buf)

    passed_all = True
    details = {}

    for i, req in enumerate(TEST_REQUESTS, 1):
        log(buf, f"\n  --- Test {i}: {req['label']} ---")

        config = {"configurable": {"thread_id": f"parse-json-test-{i}"}}
        initial_state = {
            "messages": [HumanMessage(content=req["message"])],
            "agent_data": {
                "results": {
                    "start": {"input": req["message"]},
                },
            },
            "work": {},
        }

        result = await compiled.ainvoke(initial_state, config=config)

        # Inspect classifier results — parsed JSON fields should be present
        classifier_result = result.get("agent_data", {}).get("results", {}).get("classifier", {})
        action = classifier_result.get("action", "<missing>")
        reason = classifier_result.get("reason", "<missing>")
        result_text = classifier_result.get("result_text", "")

        log(buf, f"  Classifier result_text: {result_text[:200]}")
        log(buf, f"  Parsed action: {action}")
        log(buf, f"  Parsed reason: {reason}")

        # Check routing
        expected = req["expected_action"]
        handler_key = f"handler_{expected}"
        handler_result = result.get("agent_data", {}).get("results", {}).get(handler_key, {})
        handler_text = handler_result.get("result_text", "")

        routed_correctly = bool(handler_text) and action == expected
        log(buf, f"  Handler '{handler_key}' responded: {bool(handler_text)}")
        log(buf, f"  Route correct: {routed_correctly}")

        details[f"test_{i}"] = {
            "action": action,
            "expected": expected,
            "reason": reason,
            "routed_correctly": routed_correctly,
        }
        if not routed_correctly:
            passed_all = False

    write_log(buf, "parse_json_response_run.log")

    return TestResult(
        name="Parse JSON Response",
        passed=passed_all,
        details=details,
    )
