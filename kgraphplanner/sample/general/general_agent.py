"""
General Conversational Agent — chat worker ↔ tool worker loop.

A general-purpose conversational agent built with KGraphExecGraphAgent.
A structured worker (orchestrator) decides whether to use a tool or
respond directly.  It returns an ``OrchestratorDecision``: with
``action = "tool"`` and a list of tool requests, it routes to a tool
worker that executes them and returns results.  The orchestrator
inspects the tool results and either requests more tool calls or
finishes with ``action = "respond"``.

Flow::

    start ──► orchestrator ◄──► tool_executor
                   │ (loop via action="tool" / action="respond")
                   └──[action=respond]──► final_responder ──► END

Tools available to the tool worker:
  - google_web_search_tool  (web search)
  - weather_tool            (weather lookups)
  - place_search_tool       (place / location search)

Workers:
  - orchestrator      : KGraphStructuredWorker  (OrchestratorDecision)
  - tool_executor     : KGraphToolWorker  (web search, weather, place search)
  - final_responder   : KGraphChatWorker  (response polish)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from langgraph.checkpoint.base import BaseCheckpointSaver

from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, Binding,
    WorkerNodeSpec, StartNodeSpec, EndNodeSpec,
)
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_structured_worker import KGraphStructuredWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.worker.kgraph_worker import KGraphWorker
from kgraphplanner.tool_manager.tool_manager import ToolManager

from kgraphplanner.sample.general.prompt_loader import load_prompt


# ============================================================
# Orchestrator output
# ============================================================

ToolId = Literal["google_web_search_tool", "weather_tool", "place_search_tool"]
TOOL_IDS: List[str] = list(ToolId.__args__)


class ToolRequest(BaseModel):
    tool: ToolId = Field(description="Which tool to call.")
    query: str = Field(description="What to ask the tool, e.g. a city or a search query.")


class OrchestratorDecision(BaseModel):
    """The orchestrator's decision for one turn."""

    action: Literal["tool", "respond"] = Field(
        description="'tool' to run tool calls first; 'respond' to answer the user now.")
    requests: List[ToolRequest] = Field(
        default_factory=list,
        description="When action is 'tool': every tool call needed now. They run in parallel.")
    answer: Optional[str] = Field(
        None, description="When action is 'respond': the answer to the user.")
    plan: Optional[str] = Field(
        None, description="Optional note to self, fed back on the next turn.")


# ============================================================
# Worker factory
# ============================================================

def build_workers(
    llm: BaseChatModel,
    tool_manager: ToolManager,
    summarization_llm: BaseChatModel | None = None,
) -> Dict[str, KGraphWorker]:
    """Instantiate all workers and return a name → worker registry dict.
    
    Args:
        llm: Primary LLM for orchestration and tool-call generation.
        tool_manager: Tool manager instance.
        summarization_llm: Optional lighter LLM for summarization / final
            response (e.g. reasoning_effort="low").  Falls back to *llm*.
    """
    sum_llm = summarization_llm or llm

    orchestrator = KGraphStructuredWorker(
        name="orchestrator",
        llm=llm,
        system_directive=load_prompt("orchestrator"),
        schema=OrchestratorDecision,
        include_conversation=True,
    )

    tool_executor = KGraphToolWorker(
        name="tool_executor",
        llm=llm,
        summarization_llm=sum_llm,
        system_directive=load_prompt("tool_executor"),
        tool_manager=tool_manager,
        available_tool_ids=TOOL_IDS,
        max_iters=10,
    )

    final_responder = KGraphChatWorker(
        name="final_responder",
        llm=sum_llm,
        system_directive=load_prompt("final_responder"),
    )

    return {
        "orchestrator": orchestrator,
        "tool_executor": tool_executor,
        "final_responder": final_responder,
    }


# ============================================================
# GraphSpec
# ============================================================

def build_graph_spec() -> GraphSpec:
    """
    Build the declarative graph for the general conversational agent.

    Flow::

        start ──► orchestrator ◄──► tool_executor
                       │ (loop)
                       └──[action=respond]──► final_responder ──► END

    The orchestrator ↔ tool_executor loop:
    - orchestrator emits action="tool"  → routes to tool_executor
    - tool_executor always routes back to orchestrator with results
    - orchestrator emits action="respond" → routes to final_responder
    """

    nodes = [
        StartNodeSpec(
            id="start",
            initial_data={"args": {"input": ""}},
        ),
        WorkerNodeSpec(
            id="orchestrator",
            worker_name="orchestrator",
            defaults={
                "prompt": "Decide whether to use a tool or respond directly.",
                "args": {},
            },
        ),
        WorkerNodeSpec(
            id="tool_executor",
            worker_name="tool_executor",
            defaults={
                "prompt": "Execute the requested tool and return results.",
                "args": {},
            },
        ),
        WorkerNodeSpec(
            id="final_responder",
            worker_name="final_responder",
            defaults={
                "prompt": "Polish the draft into a friendly, helpful response.",
                "args": {},
            },
        ),
        EndNodeSpec(id="end"),
    ]

    edges = [
        # ── start → orchestrator ──
        EdgeSpec(
            source="start",
            destination="orchestrator",
            bindings={
                "input": [Binding(from_node="start", path="$.input")],
            },
        ),

        # ── orchestrator → tool_executor  (action="tool") ──
        EdgeSpec(
            source="orchestrator",
            destination="tool_executor",
            condition="eq:data.action:tool",
            max_traversals=10,
            prompt="Execute the requested tool.",
            bindings={
                "tool_request": [
                    Binding(from_node="orchestrator", path="$.data", transform="json"),
                ],
            },
        ),

        # ── tool_executor → orchestrator  (always: return results) ──
        # NOTE: condition="always" is required so the exec-graph engine
        # classifies this as a conditional edge rather than an
        # unconditional worker edge.  Without it the engine prunes the
        # start → orchestrator edge (thinking orchestrator is only
        # reachable via tool_executor) and the first invocation has no
        # path to the orchestrator.
        EdgeSpec(
            source="tool_executor",
            destination="orchestrator",
            condition="always",
            prompt="Incorporate the tool results and decide next step.",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "tool_result": [
                    Binding(from_node="tool_executor", path="$.result_text"),
                ],
                "previous_plan": [
                    Binding(from_node="orchestrator", path="$.data.plan", transform="text"),
                ],
            },
        ),

        # ── orchestrator → final_responder  (action="respond") ──
        EdgeSpec(
            source="orchestrator",
            destination="final_responder",
            condition="eq:data.action:respond",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "draft_response": [
                    Binding(from_node="orchestrator", path="$.data.answer", transform="text"),
                ],
            },
        ),

        # ── orchestrator → final_responder  (fallback) ──
        # The schema only allows "tool" or "respond", so this is a safety
        # net: anything else goes to the responder instead of failing the
        # run with ExecGraphRoutingError.
        EdgeSpec(
            source="orchestrator",
            destination="final_responder",
            condition="__default__",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "draft_response": [
                    Binding(from_node="orchestrator", path="$.result_text", transform="text"),
                ],
            },
        ),
    ]

    return GraphSpec(
        graph_id="general_agent",
        name="General Conversational Agent",
        description="Chat ↔ tool loop: orchestrator decides tool calls, "
                    "tool_executor runs them, final_responder polishes output",
        nodes=nodes,
        edges=edges,
        exit_points=["final_responder"],
    )


# ============================================================
# Agent builder
# ============================================================

def build_agent(
    llm: BaseChatModel,
    tool_manager: ToolManager,
    checkpointer: BaseCheckpointSaver,
    summarization_llm: BaseChatModel | None = None,
) -> KGraphExecGraphAgent:
    """Build a ready-to-compile KGraphExecGraphAgent for general conversation."""

    worker_registry = build_workers(llm, tool_manager, summarization_llm=summarization_llm)
    graph_spec = build_graph_spec()

    return KGraphExecGraphAgent(
        name="general_agent",
        graph_spec=graph_spec,
        worker_registry=worker_registry,
        checkpointer=checkpointer,
    )
