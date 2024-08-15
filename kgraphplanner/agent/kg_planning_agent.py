from typing import (
    Annotated,
    Callable,
    Optional,
    Sequence,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

from langchain_core.language_models import LanguageModelLike

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep

from kgraphplanner.agent.tool_executor import ToolExecutor
from kgraphplanner.agent.tool_node import ToolNode


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep


StateSchema = TypeVar("StateSchema", bound=AgentState)
StateSchemaType = Type[StateSchema]

STATE_MODIFIER_RUNNABLE_NAME = "StateModifier"

MessagesModifier = Union[
    SystemMessage,
    str,
    Callable[[Sequence[BaseMessage]], Sequence[BaseMessage]],
    Runnable[Sequence[BaseMessage], Sequence[BaseMessage]],
]

StateModifier = Union[
    SystemMessage,
    str,
    Callable[[StateSchema], Sequence[BaseMessage]],
    Runnable[StateSchema, Sequence[BaseMessage]],
]


class KGPlanningAgent:
    def __init__(self,
                 model: LanguageModelLike,
                 tools: Union[ToolExecutor, Sequence[BaseTool]],
                 *,
                 state_schema: Optional[StateSchemaType] = None,
                 messages_modifier: Optional[MessagesModifier] = None,
                 state_modifier: Optional[StateModifier] = None,
                 checkpointer: Optional[BaseCheckpointSaver] = None,
                 interrupt_before: Optional[Sequence[str]] = None,
                 interrupt_after: Optional[Sequence[str]] = None,
                 debug: bool = False,
                 ):

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

    def _convert_messages_modifier_to_state_modifier(self,
                                                     messages_modifier: MessagesModifier
                                                     ) -> StateModifier:
        state_modifier: StateModifier
        if isinstance(messages_modifier, (str, SystemMessage)):
            return messages_modifier
        elif callable(messages_modifier):

            def state_modifier(state: AgentState) -> Sequence[BaseMessage]:
                return messages_modifier(state["messages"])

            return state_modifier
        elif isinstance(messages_modifier, Runnable):
            state_modifier = (lambda state: state["messages"]) | messages_modifier
            return state_modifier
        raise ValueError(
            f"Got unexpected type for `messages_modifier`: {type(messages_modifier)}"
        )

    def _get_model_preprocessing_runnable(self,
                                          state_modifier: Optional[StateModifier],
                                          messages_modifier: Optional[MessagesModifier]
                                          ) -> Runnable:
        # Add the state or message modifier, if exists
        if state_modifier is not None and messages_modifier is not None:
            raise ValueError(
                "Expected value for either state_modifier or messages_modifier, got values for both"
            )

        if state_modifier is None and messages_modifier is not None:
            state_modifier = self._convert_messages_modifier_to_state_modifier(messages_modifier)

        return self._get_state_modifier_runnable(state_modifier)

    def _get_state_modifier_runnable(self, state_modifier: Optional[StateModifier]) -> Runnable:
        state_modifier_runnable: Runnable
        if state_modifier is None:
            state_modifier_runnable = RunnableLambda(
                lambda state: state["messages"], name=STATE_MODIFIER_RUNNABLE_NAME
            )
        elif isinstance(state_modifier, str):
            _system_message: BaseMessage = SystemMessage(content=state_modifier)
            state_modifier_runnable = RunnableLambda(
                lambda state: [_system_message] + state["messages"],
                name=STATE_MODIFIER_RUNNABLE_NAME,
            )
        elif isinstance(state_modifier, SystemMessage):
            state_modifier_runnable = RunnableLambda(
                lambda state: [state_modifier] + state["messages"],
                name=STATE_MODIFIER_RUNNABLE_NAME,
            )
        elif callable(state_modifier):
            state_modifier_runnable = RunnableLambda(
                state_modifier, name=STATE_MODIFIER_RUNNABLE_NAME
            )
        elif isinstance(state_modifier, Runnable):
            state_modifier_runnable = state_modifier
        else:
            raise ValueError(
                f"Got unexpected type for `state_modifier`: {type(state_modifier)}"
            )

        return state_modifier_runnable

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

    def call_model(self, state: AgentState, config: RunnableConfig):
        response = self.model_runnable.invoke(state, config)
        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def acall_model(self, state: AgentState, config: RunnableConfig):
        response = await self.model_runnable.ainvoke(state, config)
        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}


