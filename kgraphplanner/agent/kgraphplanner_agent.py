
import asyncio
import logging
from typing import Optional, Dict, Any, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from kgraphplanner.agent.kgraph_base_agent import KGraphBaseAgent, AgentState

logger = logging.getLogger(__name__)

VALID_CATEGORIES = [
    "greeting", "help_request", "question",
    "creation_request", "planning_request", "general",
]

CLASSIFICATION_PROMPT = """Classify the following user request into exactly one of these categories:
- greeting: Simple greetings, hellos, or social pleasantries (e.g. "Hi!", "Good morning!")
- help_request: Explicitly asking for help or assistance (must contain words like "help", "assist", "support")
- question: Questions seeking information or explanations (e.g. "What is...", "How does...")
- creation_request: Requests to create, generate, write, or build something specific (e.g. "Create a meal plan", "Write a poem")
- planning_request: Requests to plan, schedule, or organize a timeline or project (e.g. "Plan a project timeline", "Schedule my week")
- general: Casual conversation, sharing feelings, opinions, or anything that doesn't clearly fit the above categories

Important: If someone asks you to "create" or "make" something, that is a creation_request, not a planning_request. Only classify as planning_request if the request is specifically about scheduling, timelines, or project planning. If someone shares feelings or personal statements without explicitly asking for help, classify as general.

User request: "{user_input}"

Respond with only the category name."""

CANNED_RESPONSES = {
    "greeting": "Hello! I'm KGraphPlanner, ready to help you with planning and task management. How can I assist you today?",
    "help_request": "I'm here to help! I can assist with planning, organizing tasks, answering questions, and more. What specifically would you like help with?",
}


class KGraphPlannerAgent(KGraphBaseAgent):
    """Agent that classifies user requests and routes to category-specific handlers.

    Flow::

        START → classify_request → (conditional) → handle_{category} → END

    Categories with canned responses (greeting, help_request) don't require
    an LLM call.  All other categories invoke ``self.model``.

    Classification is performed by ``classification_model`` (defaults to
    ``model`` when not provided).

    An optional ``event_queue`` receives structured async events for
    observability (node_start, llm_call_start, classification_complete, etc.).
    """

    def __init__(
        self,
        *,
        name: str = "kgraphplanner_agent",
        model: LanguageModelLike,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        classification_model: Optional[LanguageModelLike] = None,
        event_queue: Optional[asyncio.Queue] = None,
    ):
        super().__init__(name=name, checkpointer=checkpointer)
        self.model = model
        self.classification_model = classification_model or model
        self.event_queue = event_queue or asyncio.Queue()

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    async def _emit(self, event_type: str, data: Dict[str, Any]):
        event = {
            "type": event_type,
            "timestamp": asyncio.get_event_loop().time(),
            "data": data,
        }
        await self.event_queue.put(event)
        logger.debug(f"Event: {event_type} — {data}")

    # ------------------------------------------------------------------
    # Graph construction (standard KGraphBaseAgent lifecycle)
    # ------------------------------------------------------------------

    def build_graph(self) -> StateGraph:
        """Build the classification→routing→handler graph."""
        graph = StateGraph(AgentState)

        # --- classify node ---------------------------------------------------
        async def classify_request(state: AgentState) -> Dict[str, Any]:
            await self._emit("node_start", {"node": "classify_request"})

            messages = state.get("messages", [])
            last_message = messages[-1] if messages else None

            if not last_message or not isinstance(last_message, HumanMessage):
                category = "general"
            else:
                await self._emit("llm_call_start", {"purpose": "classification"})
                prompt = CLASSIFICATION_PROMPT.format(user_input=last_message.content)
                response = await self.classification_model.ainvoke(
                    [HumanMessage(content=prompt)]
                )
                category = response.content.strip().lower()
                if category not in VALID_CATEGORIES:
                    category = "general"
                await self._emit("llm_call_end", {"category": category})

            await self._emit("classification_complete", {
                "category": category,
                "message": last_message.content if last_message else "No message",
            })
            await self._emit("node_end", {"node": "classify_request", "category": category})

            return {
                "agent_data": {
                    "request_category": category,
                    "processing_step": "classified",
                },
            }

        # --- handler factory --------------------------------------------------
        def _make_handler(category: str):
            """Return an async handler node for *category*."""
            canned = CANNED_RESPONSES.get(category)

            async def handler(state: AgentState) -> Dict[str, Any]:
                await self._emit("node_start", {"node": f"handle_{category}"})

                if canned:
                    response_content = canned
                else:
                    messages = state.get("messages", [])
                    await self._emit("llm_call_start", {"category": category})
                    response = await self.model.ainvoke(messages)
                    response_content = response.content
                    await self._emit("llm_call_end", {"response_length": len(response_content)})

                await self._emit("response_generated", {
                    "category": category,
                    "response_length": len(response_content),
                })
                await self._emit("node_end", {"node": f"handle_{category}"})

                return {
                    "messages": [AIMessage(content=response_content)],
                    "agent_data": {"processing_step": "completed"},
                }

            handler.__name__ = f"handle_{category}"
            return handler

        # --- routing function -------------------------------------------------
        def route_to_handler(state: AgentState) -> str:
            category = state.get("agent_data", {}).get("request_category", "general")
            return f"handle_{category}"

        # --- wire up nodes and edges ------------------------------------------
        graph.add_node("classify_request", classify_request)

        handler_names = {}
        for cat in VALID_CATEGORIES:
            node_name = f"handle_{cat}"
            graph.add_node(node_name, _make_handler(cat))
            handler_names[node_name] = node_name

        graph.add_edge(START, "classify_request")
        graph.add_conditional_edges("classify_request", route_to_handler, handler_names)

        for node_name in handler_names:
            graph.add_edge(node_name, END)

        return graph

    # ------------------------------------------------------------------
    # Execution (standard KGraphBaseAgent lifecycle)
    # ------------------------------------------------------------------

    async def arun(self, messages: List[BaseMessage], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Run the agent: classify the request and route to the appropriate handler."""
        compiled_graph = self.get_compiled_graph()

        await self._emit("agent_start", {"message_count": len(messages)})

        initial_state = {
            "messages": messages,
            "agent_data": {},
            "work": {},
        }

        try:
            result = await compiled_graph.ainvoke(initial_state, config=config)
            await self._emit("agent_complete", {
                "final_step": result.get("agent_data", {}).get("processing_step"),
            })
            return result
        except Exception as e:
            await self._emit("agent_error", {"error": str(e)})
            raise

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_agent_info(self) -> Dict[str, Any]:
        base_info = super().get_agent_info()
        base_info.update({
            "agent_type": "classifier_router",
            "categories": VALID_CATEGORIES,
        })
        return base_info
