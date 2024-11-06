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
from kgraphplanner.agent.tool_executor import ToolExecutor
from kgraphplanner.agent.tool_node import ToolNode
from kgraphplanner.structured_response.agent_status_response import AgentStatusResponse
from kgraphplanner.structured_response.weather_response import WeatherResponse
from kgraphplanner.tools_internal.capture.capture_response_tool import CaptureResponseTool


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep

    final_response: TypedDict


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


class KGPlanningBaseAgent(ABC):

    def __init__(self):
        self.initialized = False
        self.model_runnable = None
        self.reasoning_queue = None


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

    @abstractmethod
    def compile(self):
        pass

    def call_model(self, state: AgentState, config: RunnableConfig):

        print(f"call config: {config}")

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

        print(f"async call config: {config}")

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
