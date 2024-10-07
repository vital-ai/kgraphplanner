from abc import ABC, abstractmethod
from typing import (
    Annotated,
    Callable,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import TypedDict

from langchain_core.language_models import LanguageModelLike

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage, HumanMessage, ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep

from kgraphplanner.agent.kg_planning_base_agent import KGPlanningBaseAgent, StateSchemaType, StateModifier, \
    MessagesModifier, AgentState
from kgraphplanner.agent.tool_executor import ToolExecutor
from kgraphplanner.agent.tool_node import ToolNode
from kgraphplanner.structured_response.agent_status_response import AgentStatusResponse
from kgraphplanner.structured_response.weather_response import WeatherResponse
from kgraphplanner.tools_internal.capture.capture_response_tool import CaptureResponseTool


class KGPlanningAgent(KGPlanningBaseAgent):
    def __init__(self, *, model: LanguageModelLike, tools: Union[ToolExecutor, Sequence[BaseTool]],
                 state_schema: Optional[StateSchemaType] = None, messages_modifier: Optional[MessagesModifier] = None,
                 state_modifier: Optional[StateModifier] = None, checkpointer: Optional[BaseCheckpointSaver] = None,
                 interrupt_before: Optional[Sequence[str]] = None, interrupt_after: Optional[Sequence[str]] = None,
                 debug: bool = False):

        super().__init__()
        self.initialized = False

        self.model = model
        self.tools = tools
        self.state_schema = state_schema
        self.messages_modifier = messages_modifier
        self.state_modifier = state_modifier
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before
        self.interrupt_after = interrupt_after
        self.debug = debug

        if self.state_schema is not None:
            if missing_keys := {"messages", "is_last_step"} - set(
                    self.state_schema.__annotations__
            ):
                raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

        if isinstance(self.tools, ToolExecutor):
            self.tool_classes = self.tools.tools
        else:
            self.tool_classes = self.tools

        self.model = self.model.bind_tools(self.tool_classes)

        self.preprocessor = self._get_model_preprocessing_runnable(self.state_modifier, self.messages_modifier)
        self.model_runnable = self.preprocessor | self.model

        self.workflow = StateGraph(self.state_schema or AgentState)

        # Define the two nodes we will cycle between
        self.workflow.add_node("agent", RunnableLambda(self.call_model, self.acall_model))
        self.workflow.add_node("tools", ToolNode(self.tool_classes))

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.workflow.set_entry_point("agent")

        # We now add a conditional edge
        self.workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "tools",
                # Otherwise we finish.
                "end": END,
            },
        )

        self.workflow.add_edge("tools", "agent")

        self.compiled_state_graph: CompiledGraph = None

    def compile(self):

        self.compiled_state_graph = self.workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=self.interrupt_before,
            interrupt_after=self.interrupt_after,
            debug=self.debug,
        )

        self.initialized = True

        return self.compiled_state_graph

    def should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"
