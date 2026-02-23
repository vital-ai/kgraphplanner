"""
General Conversational Agent — chat worker ↔ tool worker loop.

A general-purpose conversational agent built with KGraphExecGraphAgent.
A chat worker (orchestrator) decides whether to use a tool or respond
directly.  When it needs a tool it emits a JSON message with
``action = "tool"`` and tool details; this routes to a tool worker that
executes the request and returns results.  The orchestrator inspects
the tool result and either requests another tool call or finishes with
``action = "respond"``.

Flow::

    start ──► orchestrator ◄──► tool_executor
                   │ (loop via action="tool" / action="respond")
                   └──[action=respond]──► final_responder ──► END

Tools available to the tool worker:
  - google_web_search_tool  (web search)
  - weather_tool            (weather lookups)
  - place_search_tool       (place / location search)

Workers:
  - orchestrator      : KGraphChatWorker  (parse_json_response=True)
  - tool_executor     : KGraphToolWorker  (web search, weather, place search)
  - final_responder   : KGraphChatWorker  (response polish)
"""

from __future__ import annotations

from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, Binding,
    WorkerNodeSpec, StartNodeSpec, EndNodeSpec,
)
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.worker.kgraph_worker import KGraphWorker
from kgraphplanner.tool_manager.tool_manager import ToolManager
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum

from kgraphplanner.sample.general.prompt_loader import load_prompt


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

    orchestrator = KGraphChatWorker(
        name="orchestrator",
        llm=llm,
        system_directive=load_prompt("orchestrator"),
        parse_json_response=True,
    )

    tool_executor = KGraphToolWorker(
        name="tool_executor",
        llm=llm,
        summarization_llm=sum_llm,
        system_directive=load_prompt("tool_executor"),
        tool_manager=tool_manager,
        available_tool_ids=[
            ToolNameEnum.google_web_search_tool.value,
            ToolNameEnum.weather_tool.value,
            ToolNameEnum.place_search_tool.value,
        ],
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
            condition="result.get('action') == 'tool'",
            max_traversals=10,
            prompt="Execute the requested tool.",
            bindings={
                "tool_request": [
                    Binding(from_node="orchestrator", path="$", transform="text"),
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
                    Binding(from_node="orchestrator", path="$.plan", transform="text"),
                ],
            },
        ),

        # ── orchestrator → final_responder  (action="respond") ──
        EdgeSpec(
            source="orchestrator",
            destination="final_responder",
            condition="result.get('action') == 'respond'",
            bindings={
                "input": [Binding(from_node="start", path="$.input", transform="text")],
                "draft_response": [
                    Binding(from_node="orchestrator", path="$.answer", transform="text"),
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
