"""
Case: Sales Agent — multi-step business-loan assistant using GraphSpec + ExecGraph.

Demonstrates a multi-step pipeline built declaratively via GraphSpec and
executed by KGraphExecGraphAgent.  The flow models a business-loan sales
agent that:

  1. **Topic classifier** – categorises the inbound message into one of:
        business_qa, product_qa, active_application, active_loan, other

  2. **Initial lookup** (conditional) – for active-application / active-loan
     cases, automatically extracts identifiers (email, phone, application ID)
     from the message and performs a one-shot account lookup.  May return
     nothing if no identifiers are present — that is fine, the pipeline
     continues either way.

  3. **Interactive chat + tool loop** – a chat worker (orchestrator) and a
     KGraphToolWorker (executor) connected in a cycle.  The tool worker
     uses a mock loan_lookup_tool that returns deterministic fake account
     data.  The chat worker checks the tool's result: if found, it exits
     the loop; if not found, it uses the LLM to draft a question, calls
     interrupt() to pause the graph and ask the customer, then on resume
     routes back to the tool worker with the new information.

  4. **Assistant router** – a second case-classification step that assigns
     a specialist assistant based on the first-round topic and whatever
     account data was gathered in steps 2–3.

  5. **Specialist** – the selected specialist handles the request, possibly
     calling tools (web search, etc.) to gather supporting data.

  6. **Final responder** – polishes the specialist output into a friendly,
     customer-facing response.

Workers:
  - topic_classifier   : KGraphCaseWorker  (5 cases)
  - initial_lookup      : KGraphChatWorker  (auto extraction + one-shot lookup)
  - interactive_chat     : KGraphChatWorker  (orchestrator, enable_interrupt=True)
  - interactive_tool     : KGraphToolWorker  (mock loan_lookup_tool — exercises
        the decision → tool → finalize subgraph inside an exec-graph loop)
  - assistant_router     : KGraphCaseWorker  (3 specialist types)
  - specialist           : KGraphChatWorker  (domain specialist)
  - final_responder      : KGraphChatWorker  (response polish)

In production initial_lookup would also be a KGraphToolWorker calling a real
loan-management API, and the specialist could be a KGraphToolWorker with
access to product databases, underwriting rules, etc.
"""

from __future__ import annotations

import io
import os
import sys
import asyncio

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from langgraph.types import Command

from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, Binding,
    WorkerNodeSpec, StartNodeSpec, EndNodeSpec,
)
from kgraphplanner.worker.kgraph_case_worker import KGraphCaseWorker, KGCase
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.tools.mock_loan_lookup.loan_lookup_tool import LoanLookupTool, TOOL_NAME as LOAN_TOOL_NAME
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer
from kgraphplanner.checkpointer.kgraphredis_checkpointer import KGraphRedisCheckpointer

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import (
    log, write_log, save_png, execute_with_logging, OUTPUT_DIR,
)


# ============================================================
# Cases
# ============================================================

TOPIC_CASES = [
    KGCase(
        id="business_qa",
        name="Business Q&A",
        description="general questions about the business, contact information, "
                    "company background, or how to get in touch",
    ),
    KGCase(
        id="product_qa",
        name="Product Q&A",
        description="questions about loan product offerings, interest rates, "
                    "terms, eligibility requirements, or how to apply",
    ),
    KGCase(
        id="active_application",
        name="Active Application",
        description="questions or follow-ups about an in-progress loan "
                    "application that the customer has already submitted",
    ),
    KGCase(
        id="active_loan",
        name="Active Loan / New Applicant",
        description="questions about an existing, funded loan such as "
                    "payment schedule, payoff, or account servicing, "
                    "or a new applicant inquiring about starting a loan application",
    ),
    KGCase(
        id="other",
        name="Other",
        description="anything that does not fit the categories above",
    ),
]

ASSISTANT_CASES = [
    KGCase(
        id="application_specialist",
        name="Application Specialist",
        description="handles questions about in-progress loan applications, "
                    "status checks, document requirements, and next steps",
    ),
    KGCase(
        id="account_specialist",
        name="Account Specialist",
        description="handles questions about funded loans, payments, "
                    "payoff amounts, account changes, and servicing",
    ),
    KGCase(
        id="sales_advisor",
        name="Sales Advisor",
        description="handles general business questions, product information, "
                    "eligibility guidance, and new-loan inquiries",
    ),
]


# ============================================================
# Test requests
# ============================================================

TEST_REQUESTS = [
    {
        "input": "What interest rates do you offer on a 5-year business loan?",
        "expected_topic": "Product Q&A",
        "description": "Product rate inquiry",
        "simulated_reply": None,  # no interrupt expected
    },
    {
        "input": "I submitted my application last week from john@acme.com. "
                 "What's the status?",
        "expected_topic": "Active Application",
        "description": "Application status check (with email)",
        "simulated_reply": None,
    },
    {
        "input": "How do I contact your underwriting team?",
        "expected_topic": "Business Q&A",
        "description": "Business contact inquiry",
        "simulated_reply": None,
    },
    {
        "input": "I have an active loan and need to know my remaining balance. "
                 "My email is jane@widgets.co.",
        "expected_topic": "Active Loan",
        "description": "Existing-loan balance inquiry",
        "simulated_reply": None,
    },
    {
        "input": "Can you tell me a joke?",
        "expected_topic": "Other",
        "description": "Off-topic request",
        "simulated_reply": None,
    },
    {
        "input": "I already have a funded business loan with you and I need to "
                 "check my remaining balance and next payment date.",
        "expected_topic": "Active Loan",
        "description": "Active loan inquiry (no identifiers — triggers interrupt)",
        "simulated_reply": "My loan ID is LN-98765",
    },
]


# ============================================================
# Worker factory
# ============================================================

def bind_for_role(llm: BaseChatModel, role: str) -> BaseChatModel:
    """Return a bound LLM with provider-appropriate caps for the given role.

    OpenAI reasoning models (gpt-5-mini etc.) consume reasoning tokens inside
    max_tokens, so caps must be higher.  Anthropic's max_tokens counts only
    visible output — thinking budget is separate — so caps can be lower.
    """
    is_openai = isinstance(llm, ChatOpenAI)

    ROLE_CAPS = {
        "classify": {
            "openai":    {"max_tokens": 100,  "reasoning_effort": "low"},
            "anthropic": {"max_tokens": 30},
        },
        "short": {
            "openai":    {"max_tokens": 400,  "reasoning_effort": "low"},
            "anthropic": {"max_tokens": 200},
        },
        "answer": {
            "openai":    {"max_tokens": 1024, "reasoning_effort": "low"},
            "anthropic": {"max_tokens": 500},
        },
    }
    caps = ROLE_CAPS.get(role, {})
    kwargs = caps.get("openai" if is_openai else "anthropic", {})
    return llm.bind(**kwargs) if kwargs else llm


def _build_workers(llm: BaseChatModel):
    """Instantiate all workers and return a name→worker registry dict."""

    # Per-worker LLM instances with tailored max_tokens to reduce latency.
    llm_classify = bind_for_role(llm, "classify")
    llm_short    = bind_for_role(llm, "short")
    llm_answer   = bind_for_role(llm, "answer")

    topic_classifier = KGraphCaseWorker(
        name="topic_classifier",
        llm=llm_classify,
        system_directive=(
            "You are a front-desk classifier for a business-loan company. "
            "Categorise the customer's message into exactly one topic."
        ),
        cases=TOPIC_CASES,
    )

    initial_lookup = KGraphChatWorker(
        name="initial_lookup",
        llm=llm_short,
        system_directive=(
            "You are an automatic account-lookup service.  Given a customer "
            "message, extract any identifying information (email address, "
            "phone number, application ID, full name, etc.) and simulate "
            "looking up their account in one shot.\n\n"
            "If you find identifiers and can locate an account, respond with "
            "a JSON block:\n"
            '  {"found": true, "email": "...", "loan_id": "LN-12345", '
            '"status": "active", "details": "..."}\n\n'
            "If identifiers are present but no account matches, respond with:\n"
            '  {"found": false, "identifiers": {"email": "..."}, '
            '"reason": "no matching account"}\n\n'
            "If no identifiers are present at all, respond with:\n"
            '  {"found": false, "identifiers": {}, '
            '"reason": "no identifying information provided"}\n\n'
            "Always respond with ONLY the JSON block, no other text."
        ),
    )

    interactive_chat = KGraphChatWorker(
        name="interactive_chat",
        llm=llm_short,
        enable_interrupt=True,
        system_directive=(
            "You are an interactive loan-account lookup orchestrator.  "
            "When a lookup fails, draft a short, friendly question asking "
            "the customer for identifying information so the lookup can "
            "be retried."
        ),
    )

    # Create a ToolManager with the mock loan-lookup tool so
    # interactive_tool can be a real KGraphToolWorker exercising the
    # decision → tool → finalize subgraph inside an exec-graph loop.
    tm = ToolManager()
    LoanLookupTool(config={}, tool_manager=tm)

    interactive_tool = KGraphToolWorker(
        name="interactive_tool",
        llm=llm_short,
        system_directive=(
            "You are a loan-account lookup assistant.  Use the "
            "loan_lookup_tool to search for the customer's account.\n\n"
            "Extract identifiers (email, phone, loan_id, application_id, "
            "or name) from the context provided, then call the tool.\n\n"
            "After receiving the tool result you MUST respond with ONLY "
            "a JSON block, no other text.  Copy the tool's JSON output "
            "exactly — do NOT add commentary.\n\n"
            "If the context has no usable identifiers, respond:\n"
            '  {"found": false, "reason": "no identifying information provided"}\n'
        ),
        tool_manager=tm,
        available_tool_ids=[LOAN_TOOL_NAME],
    )

    assistant_router = KGraphCaseWorker(
        name="assistant_router",
        llm=llm_classify,
        system_directive=(
            "You are an internal routing assistant.  Based on the topic "
            "classification and any account lookup results provided, select "
            "the most appropriate specialist to handle this customer request."
        ),
        cases=ASSISTANT_CASES,
    )

    specialist = KGraphChatWorker(
        name="specialist",
        llm=llm_answer,
        system_directive=(
            "You are a knowledgeable business-loan specialist.  Use the "
            "context provided (topic, account details, specialist role) to "
            "draft a helpful answer to the customer's question.  "
            "If account information was not found, politely ask the customer "
            "to provide identifying details (email, application ID, etc.) so "
            "you can assist them.\n\n"
            "Be professional and concise — aim for 2-4 short paragraphs max."
        ),
    )

    final_responder = KGraphChatWorker(
        name="final_responder",
        llm=llm_answer,
        system_directive=(
            "You are the final quality-assurance step.  Take the specialist's "
            "draft response and produce a polished, customer-ready message.  "
            "Maintain all factual content but ensure the tone is warm, "
            "professional, and on-brand for a business-lending company.  "
            "Do NOT add information that was not in the draft.  "
            "Keep the response concise — no longer than the draft."
        ),
    )

    return {
        "topic_classifier": topic_classifier,
        "initial_lookup": initial_lookup,
        "interactive_chat": interactive_chat,
        "interactive_tool": interactive_tool,
        "assistant_router": assistant_router,
        "specialist": specialist,
        "final_responder": final_responder,
    }


# ============================================================
# GraphSpec
# ============================================================

def _build_graph_spec() -> GraphSpec:
    """
    Build the declarative graph for the sales-agent pipeline.

    Flow:
        start ──► topic_classifier
                       │
                       ├──[active_application | active_loan]──► initial_lookup ──► interactive_chat ◄──► interactive_tool
                       │                                                               │ (loop)
                       │                                                               ├──[action=continue]──► assistant_router
                       │                                                               │
                       └──[default: other topics]──────────────────────────────────────► assistant_router
                                                                                             │
                                                                                         specialist
                                                                                             │
                                                                                      final_responder
                                                                                             │
                                                                                            END

    The interactive_chat ↔ interactive_tool loop:
    - interactive_chat routes to interactive_tool when action="lookup"
    - interactive_tool always routes back to interactive_chat with results
    - interactive_chat routes to assistant_router when action="continue"
    - interactive_chat may call interrupt() to ask the user for info
    """

    nodes = [
        StartNodeSpec(
            id="start",
            initial_data={
                "args": {"input": ""},  # overridden at runtime via start_seed
            },
        ),
        WorkerNodeSpec(
            id="topic_classifier",
            worker_name="topic_classifier",
            defaults={
                "prompt": "Classify the customer message into one of the available categories.",
                "args": {},
            },
        ),
        WorkerNodeSpec(
            id="initial_lookup",
            worker_name="initial_lookup",
            defaults={
                "prompt": "Extract identifiers from the message and attempt a one-shot account lookup.",
                "args": {},
            },
        ),
        WorkerNodeSpec(
            id="interactive_chat",
            worker_name="interactive_chat",
            defaults={
                "prompt": "Orchestrate the interactive lookup loop.",
                "args": {},
            },
        ),
        WorkerNodeSpec(
            id="interactive_tool",
            worker_name="interactive_tool",
            defaults={
                "prompt": "Look up the customer's account using the available identifiers.",
                "args": {},
            },
        ),
        WorkerNodeSpec(
            id="assistant_router",
            worker_name="assistant_router",
            defaults={
                "prompt": "Select the best specialist for this request based on the topic and any account details.",
                "args": {},
            },
        ),
        WorkerNodeSpec(
            id="specialist",
            worker_name="specialist",
            defaults={
                "prompt": "Draft a helpful response to the customer's question.",
                "args": {},
            },
        ),
        WorkerNodeSpec(
            id="final_responder",
            worker_name="final_responder",
            defaults={
                "prompt": "Polish the specialist draft into a customer-ready response.",
                "args": {},
            },
        ),
        EndNodeSpec(id="end"),
    ]

    edges = [
        # ── start → topic_classifier ──
        EdgeSpec(
            source="start",
            destination="topic_classifier",
            bindings={
                "input": [Binding(from_node="start", path="$.input")],
            },
        ),

        # ── topic_classifier → initial_lookup  (active application or loan) ──
        EdgeSpec(
            source="topic_classifier",
            destination="initial_lookup",
            condition="result.get('selected_case_id') in ('active_application', 'active_loan')",
            prompt="Extract identifiers and attempt a one-shot account lookup.",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "topic": [Binding(from_node="topic_classifier", path="$.result_text", transform="text")],
            },
        ),

        # ── initial_lookup → interactive_chat  (enter the loop) ──
        EdgeSpec(
            source="initial_lookup",
            destination="interactive_chat",
            prompt="Orchestrate the interactive lookup loop.",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "topic": [Binding(from_node="topic_classifier", path="$.result_text", transform="text")],
                "initial_lookup_result": [
                    Binding(from_node="initial_lookup", path="$.result_text", transform="text"),
                ],
            },
        ),

        # ── interactive_chat → interactive_tool  (action="lookup") ──
        EdgeSpec(
            source="interactive_chat",
            destination="interactive_tool",
            condition="result.get('action') == 'lookup'",
            prompt="Look up the customer's account using the available identifiers.",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "lookup_request": [
                    Binding(from_node="interactive_chat", path="$", transform="text"),
                ],
            },
        ),

        # ── interactive_tool → interactive_chat  (always: report results back) ──
        EdgeSpec(
            source="interactive_tool",
            destination="interactive_chat",
            bindings={
                "lookup_result": [
                    Binding(from_node="interactive_tool", path="$.result_text"),
                ],
            },
        ),

        # ── interactive_chat → assistant_router  (action="continue") ──
        EdgeSpec(
            source="interactive_chat",
            destination="assistant_router",
            condition="result.get('action') == 'continue'",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "topic": [Binding(from_node="topic_classifier", path="$.result_text", transform="text")],
                "account_info": [
                    Binding(from_node="interactive_chat", path="$", transform="text"),
                ],
            },
        ),

        # ── topic_classifier → assistant_router  (all other topics) ──
        EdgeSpec(
            source="topic_classifier",
            destination="assistant_router",
            condition="__default__",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "topic": [Binding(from_node="topic_classifier", path="$.result_text", transform="text")],
            },
        ),

        # ── assistant_router → specialist ──
        EdgeSpec(
            source="assistant_router",
            destination="specialist",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "topic": [Binding(from_node="topic_classifier", path="$.result_text", transform="text")],
                "assigned_specialist": [
                    Binding(from_node="assistant_router", path="$.result_text", transform="text"),
                ],
                "account_info": [
                    Binding(from_node="interactive_chat", path="$.result_text", transform="text"),
                ],
            },
        ),

        # ── specialist → final_responder ──
        EdgeSpec(
            source="specialist",
            destination="final_responder",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "draft_response": [
                    Binding(from_node="specialist", path="$.result_text", transform="text"),
                ],
            },
        ),
    ]

    return GraphSpec(
        graph_id="sales_agent",
        name="Business Loan Sales Agent",
        description="Multi-step sales agent: classify → lookup → chat↔tool loop → route → respond → polish",
        nodes=nodes,
        edges=edges,
        exit_points=["final_responder"],
    )


# ============================================================
# Agent builder
# ============================================================

def _build_agent(llm: BaseChatModel):
    """Build the KGraphExecGraphAgent with all workers wired up."""
    serializer = KGraphSerializer()
    agent_config = AgentConfig.from_env()
    backend = agent_config.checkpointing.backend.lower()
    if backend == "redis":
        checkpointer = KGraphRedisCheckpointer(
            serde=serializer, checkpoint_config=agent_config.checkpointing
        )
    else:
        checkpointer = KGraphMemoryCheckpointer(serde=serializer)

    worker_registry = _build_workers(llm)
    graph_spec = _build_graph_spec()

    return KGraphExecGraphAgent(
        name="sales_agent",
        graph_spec=graph_spec,
        worker_registry=worker_registry,
        checkpointer=checkpointer,
    )


# ============================================================
# Main entry point
# ============================================================

async def run() -> TestResult:
    """Run the sales-agent test."""
    load_dotenv()
    buf = io.StringIO()

    # --- Setup ---
    log(buf, "  === Step 1: Setup ===")
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "anthropic":
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3, max_tokens=1024)
        log(buf, f"  LLM provider: Anthropic (claude-haiku-4-5-20251001)")
    else:
        llm = ChatOpenAI(model="gpt-5-mini", temperature=0.3)
        log(buf, f"  LLM provider: OpenAI (gpt-5-mini)")
    agent = _build_agent(llm)
    compiled = agent.get_compiled_graph()

    # --- Graph PNG ---
    log(buf, "\n  === Step 2: Graph Diagram ===")
    await save_png(compiled, "sales_agent_graph.png", buf)

    # --- Agent info ---
    agent_info = agent.get_agent_info()
    log(buf, f"\n  Agent: {agent_info['name']}  type={agent_info['agent_type']}  "
            f"nodes={agent_info['node_count']}  edges={agent_info['edge_count']}")
    log(buf, f"  Worker nodes: {agent_info['worker_nodes']}")

    # --- Execute test requests ---
    # Anthropic has strict per-minute token limits (50K input tokens for Haiku).
    # Run sequentially to avoid 429 rate-limit errors.
    # OpenAI has much higher limits, so requests run in parallel.
    run_parallel = True
    log(buf, f"\n  === Step 3: Execute {len(TEST_REQUESTS)} Requests "
            f"({'parallel' if run_parallel else 'sequential'}) ===")

    async def _run_one_request(i, tc):
        import time as _time
        rbuf = io.StringIO()
        user_input = tc["input"]
        expected = tc["expected_topic"]
        desc = tc["description"]
        simulated_reply = tc.get("simulated_reply")
        log(rbuf, f"\n  --- Request {i}: {desc} ---")
        log(rbuf, f"  Expected topic: {expected}")
        log(rbuf, f"  Input: {user_input}")
        if simulated_reply:
            log(rbuf, f"  Simulated reply (for interrupt): {simulated_reply}")

        try:
            _req_t0 = _time.time()
            config = {"configurable": {"thread_id": f"sales-agent-test-{i}"}}
            messages = [HumanMessage(content=user_input)]

            # --- Initial invocation ---
            result = await execute_with_logging(
                agent.arun(messages=messages, config=config),
                rbuf,
            )

            # --- Interrupt detection + resume loop ---
            max_resumes = 3
            resume_count = 0
            while resume_count < max_resumes:
                interrupt_list = result.get("__interrupt__")
                if not interrupt_list:
                    break

                intr_obj = interrupt_list[0]
                payload_val = getattr(intr_obj, 'value', intr_obj)
                question = (
                    payload_val.get("question", str(payload_val))
                    if isinstance(payload_val, dict)
                    else str(payload_val)
                )
                worker_name = (
                    payload_val.get("worker", "?")
                    if isinstance(payload_val, dict)
                    else "?"
                )
                log(rbuf, f"\n  🛑 INTERRUPT detected!")
                log(rbuf, f"     Worker: {worker_name}")
                log(rbuf, f"     Question: {question}")

                if not simulated_reply:
                    log(rbuf, f"     ⚠️  No simulated_reply configured — cannot resume")
                    break

                log(rbuf, f"     Resuming with: {simulated_reply}")
                resume_count += 1

                result = await execute_with_logging(
                    compiled.ainvoke(
                        Command(resume=simulated_reply),
                        config=config,
                    ),
                    rbuf,
                )
                simulated_reply = None

            if resume_count > 0:
                log(rbuf, f"  Resumed {resume_count} time(s)")

            # Extract final results from agent_data
            agent_data = result.get("agent_data", {})
            results_map = agent_data.get("results", {})

            final_result = results_map.get("final_responder", {})
            final_text = final_result.get("result_text", "")

            topic_result = results_map.get("topic_classifier", {})
            init_lookup_result = results_map.get("initial_lookup", {})
            chat_result = results_map.get("interactive_chat", {})
            tool_result = results_map.get("interactive_tool", {})
            router_result = results_map.get("assistant_router", {})

            topic_id = topic_result.get('selected_case_id') or str(topic_result.get('result_text', '?'))[:80]
            router_id = router_result.get('selected_case_id') or str(router_result.get('result_text', '?'))[:80]

            log(rbuf, f"  Topic: {topic_id}")
            if init_lookup_result:
                log(rbuf, f"  Initial lookup: {str(init_lookup_result.get('result_text', ''))[:120]}")
            if chat_result:
                chat_action = chat_result.get('action', '?')
                log(rbuf, f"  Interactive chat: action={chat_action}")
            if tool_result:
                log(rbuf, f"  Interactive tool: {str(tool_result.get('result_text', ''))[:120]}")
            log(rbuf, f"  Router: {router_id}")
            log(rbuf, f"  Final response: {final_text[:300]}")

            _req_elapsed = _time.time() - _req_t0
            log(rbuf, f"  ⏱️  Request {i} took {_req_elapsed:.1f}s")

            ok = bool(final_text and len(final_text) > 20)
            return {
                "test": i, "desc": desc, "expected": expected,
                "ok": ok, "resumes": resume_count, "elapsed": _req_elapsed,
            }, rbuf.getvalue()

        except Exception as e:
            _req_elapsed = _time.time() - _req_t0
            log(rbuf, f"  ❌ Error after {_req_elapsed:.1f}s: {e}")
            import traceback
            log(rbuf, traceback.format_exc())
            return {
                "test": i, "desc": desc, "expected": expected,
                "ok": False, "error": str(e), "elapsed": _req_elapsed,
            }, rbuf.getvalue()

    # Launch requests — parallel (OpenAI) or sequential (Anthropic)
    sub_results = []
    if run_parallel:
        tasks = [
            _run_one_request(i, tc)
            for i, tc in enumerate(TEST_REQUESTS, 1)
        ]
        completed = await asyncio.gather(*tasks)
        for sub_result, req_log in completed:
            buf.write(req_log)
            sub_results.append(sub_result)
    else:
        for i, tc in enumerate(TEST_REQUESTS, 1):
            sub_result, req_log = await _run_one_request(i, tc)
            buf.write(req_log)
            sub_results.append(sub_result)

    # --- Summary ---
    passed_count = sum(1 for r in sub_results if r["ok"])
    total_count = len(sub_results)

    log(buf, f"\n  === Results: {passed_count}/{total_count} requests succeeded ===")
    for r in sub_results:
        status = "✅" if r["ok"] else "❌"
        resumes = r.get("resumes", 0)
        elapsed = r.get("elapsed", 0)
        resume_note = f" [resumed {resumes}x]" if resumes else ""
        time_note = f" ({elapsed:.1f}s)"
        log(buf, f"    {status} Test {r['test']}: {r['desc']} (expected: {r['expected']}){resume_note}{time_note}")

    # --- Write log ---
    write_log(buf, "sales_agent_run.log")

    return TestResult(
        name="Sales Agent Pipeline",
        passed=passed_count == total_count,
        details={
            "requests_total": total_count,
            "requests_passed": passed_count,
            "graph_id": "sales_agent",
            "worker_nodes": agent_info.get("worker_nodes", []),
            "output_dir": OUTPUT_DIR,
        },
    )
