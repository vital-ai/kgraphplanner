"""
Case: the planner routes on its inputs through a gate worker (0.2.1, K6).

The registry holds presence_gate plus two described LLM workers. The request
needs a branch on whether an input is empty. Planner conditionals can't start
at `start`, so a correct plan is start -> gate -> (classifier | responder).

Checks:
  - the generated plan uses the gate and branches on has:present.<arg>;
  - executing it with an empty inbound message skips the classifier;
  - executing it with a message classifies first.
"""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from kgraphplanner.agent.kgraph_planner_agent import KGraphPlannerAgent
from kgraphplanner.worker.gates import presence_gate
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from test_scripts_planner.cases.test_result import TestResult

REQUEST = (
    "Prepare an outbound message to a customer. The start data has two args: "
    "'inbound_message' (the customer's latest message, which may be empty) and "
    "'customer_name'. If there is an inbound message, classify it first and write "
    "a reply that uses the classification. If there is no inbound message, skip "
    "classification entirely and write a short proactive check-in instead."
)


async def run() -> TestResult:
    load_dotenv()
    planner_llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    exec_llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=1000)

    registry = {
        "gate": presence_gate(),
        "classifier": KGraphChatWorker(
            name="classifier", llm=exec_llm, include_conversation=False,
            description="Classifies a customer's inbound message into a short category label.",
            system_directive="Classify the customer's message as one of: complaint, question, "
                             "request, other. Reply with the label only."),
        "responder": KGraphChatWorker(
            name="responder", llm=exec_llm, include_conversation=False,
            description="Writes the outbound message to the customer from whatever inputs it gets.",
            system_directive="Write a brief, friendly outbound message to the customer."),
    }
    planner = KGraphPlannerAgent(name="test_planner_gate", planner_llm=planner_llm,
                                 worker_registry=registry)

    program = await planner.plan([HumanMessage(content=REQUEST)])
    spec = planner.expand(program)

    gate_nodes = [n.id for n in spec.nodes if getattr(n, "worker_name", None) == "gate"]
    gate_conditions = [e.condition for e in spec.edges
                       if e.source in gate_nodes and e.condition is not None]
    print(f"  program_id: {program.program_id}")
    for e in spec.edges:
        print(f"    {e.source} -> {e.destination}" + (f"  [{e.condition}]" if e.condition else ""))

    assert gate_nodes, f"Expected the plan to use the gate worker; nodes: {[n.id for n in spec.nodes]}"
    assert any("present.inbound_message" in c for c in gate_conditions), \
        f"Expected a branch on has:present.inbound_message; got {gate_conditions}"

    classifier_nodes = {n.id for n in spec.nodes if getattr(n, "worker_name", None) == "classifier"}
    responder_nodes = {n.id for n in spec.nodes if getattr(n, "worker_name", None) == "responder"}

    quiet = await planner.execute(spec, {"inbound_message": "", "customer_name": "Dana"}, timeout=120)
    print(f"  empty message trace: {quiet.trace}")
    assert not classifier_nodes & set(quiet.trace), "Classifier ran for an empty inbound message"
    assert responder_nodes & set(quiet.trace), "Responder didn't run for an empty inbound message"

    loud = await planner.execute(
        spec, {"inbound_message": "My last invoice charged me twice, please fix it.",
               "customer_name": "Dana"}, timeout=120)
    print(f"  with message trace: {loud.trace}")
    classified_at = min(loud.trace.index(n) for n in classifier_nodes if n in loud.trace)
    replied_at = min(loud.trace.index(n) for n in responder_nodes if n in loud.trace)
    assert classified_at < replied_at, "Expected the classifier to run before the responder"

    return TestResult(name="Planner routes through a gate worker", passed=True, details={
        "program_id": program.program_id,
        "gate_conditions": gate_conditions,
        "empty_trace": quiet.trace,
        "message_trace": loud.trace,
    })
