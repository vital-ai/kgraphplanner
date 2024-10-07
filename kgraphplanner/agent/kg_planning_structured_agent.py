import json
import pprint
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
from langgraph.managed import IsLastStep
from kgraphplanner.agent.tool_executor import ToolExecutor
from kgraphplanner.agent.tool_node import ToolNode
from kgraphplanner.structured_response.agent_payload_response import AgentPayloadResponse
from kgraphplanner.structured_response.agent_status_response import AgentStatusResponse
from kgraphplanner.structured_response.weather_response import WeatherResponse
from kgraphplanner.tools_internal.capture.capture_response_tool import CaptureResponseTool
from kgraphplanner.agent.kg_planning_base_agent import KGPlanningBaseAgent, StateSchemaType, StateModifier, MessagesModifier, \
    AgentState


class KGPlanningStructuredAgent(KGPlanningBaseAgent):
    def __init__(self, *, model_tools: LanguageModelLike, model_structured: LanguageModelLike,
                 tools: Union[ToolExecutor, Sequence[BaseTool]],
                 state_schema: Optional[StateSchemaType] = None, messages_modifier: Optional[MessagesModifier] = None,
                 state_modifier: Optional[StateModifier] = None, checkpointer: Optional[BaseCheckpointSaver] = None,
                 interrupt_before: Optional[Sequence[str]] = None, interrupt_after: Optional[Sequence[str]] = None,
                 debug: bool = False):

        super().__init__()
        self.initialized = False

        self.model_tools = model_tools
        self.model_structured = model_structured

        self.tools = tools
        self.state_schema = state_schema
        self.messages_modifier = messages_modifier
        self.state_modifier = state_modifier
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before
        self.interrupt_after = interrupt_after
        self.debug = debug

        # setting , strict=True slows things down a lot
        # potentially convert schema?

        self.model_structured_bound = self.model_structured.with_structured_output(
            AgentPayloadResponse)

        if self.state_schema is not None:
            if missing_keys := {"messages", "is_last_step"} - set(
                    self.state_schema.__annotations__
            ):
                raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

        if isinstance(self.tools, ToolExecutor):
            self.tool_classes = self.tools.tools
        else:
            self.tool_classes = self.tools

        self.model_tools = self.model_tools.bind_tools(self.tool_classes, parallel_tool_calls=True)

        self.preprocessor = self._get_model_preprocessing_runnable(self.state_modifier, self.messages_modifier)

        self.model_runnable = self.preprocessor | self.model_tools

        self.workflow = StateGraph(self.state_schema or AgentState)

        self.workflow.add_node("agent", RunnableLambda(self.call_model, self.acall_model))

        self.workflow.add_node("tools", ToolNode(self.tool_classes))

        self.workflow.add_node("end_state", RunnableLambda(self.end_state))

        self.workflow.set_entry_point("agent")

        self.workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end_state": "end_state"
            },
        )

        self.workflow.add_edge("tools", "agent")

        self.workflow.add_edge("end_state", END)

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
        if not last_message.tool_calls:
            return "end_state"
        else:
            return "continue"

    def format_dict(self, d, indent=0):
        """Recursively formats a dictionary for pretty printing."""
        text = ""
        for key, value in d.items():
            if isinstance(value, dict):
                text += "  " * indent + f"{key}:\n" + self.format_dict(value, indent + 1)
            elif isinstance(value, list):
                text += "  " * indent + f"{key}:\n"
                for item in value:
                    if isinstance(item, dict):
                        text += self.format_dict(item, indent + 1)
                    else:
                        text += "  " * (indent + 1) + f"- {item}\n"
            else:
                text += "  " * indent + f"{key}: {value}\n"
        return text

    def end_state(self, state: AgentState, config: RunnableConfig):

        print(f"State: {state}")
        print(f"Config: {config}")

        messages = state["messages"]

        message_content = ""

        for message in messages:
            if isinstance(message, HumanMessage):
                content = message.content
                message_content += (content + "\n")
            if isinstance(message, AIMessage):
                content = message.content
                # skip the tool calls that are empty content
                if content:
                    message_content += (content + "\n")
            if isinstance(message, ToolMessage):
                content = message.content
                if content:
                    data: dict = json.loads(content)

                    tool_data_class = data.get('tool_data_class', None)

                    if tool_data_class and tool_data_class == 'WeatherData':
                        # reduce data not needed for the payload selection
                        data.pop('daily_predictions', None)

                    # pretty_string = pprint.pformat(data)
                    # use text format instead of markup
                    # potentially switch to markdown
                    pretty_string = self.format_dict(data, indent=2)
                    message_content += (pretty_string + "\n")

        available_structured_responses = [
            "WeatherData: Use for Weather Reports"
        ]

        structured_response_bullet_list = "\n".join(
            f"\t\t\t\t\t* {item}" for item in available_structured_responses)


        internal_instructions = f"""
        (internal instructions)
        
            The available payload response classes are:
{structured_response_bullet_list}

            You decide if one or more of the above payload responses should be included in the response to the request
            that was specified in the previous messages.  You may include multiple instances of a payload response.
            The data record of a payload response must be found in the output of a previous tool call.
            You include a payload response only if it is relevant, and no payload response (empty list) if none are relevant.

            You fill out the fields of AgentPayloadResponse based on the previous messages.

            human_text_request: This should be the initial request from the human  
            agent_text_response: This should be the text of the response to the initial request. 
            agent_request_status: this status of the initial request
            agent_payload_list: this should be the list of payloads to include in the response, if any.
            agent_request_status_message: an optional message used if the status is not complete to explain why
            missing_input: an optional message for when the status is missing_input to list what is missing                   
  
        """


        instruction_text = f"""
        Previous Messages:
        {message_content}
        -----------------------------------------------------------
        {internal_instructions}
        """

        instructions = HumanMessage(content=instruction_text)

        # this is bound to AgentStatusResponse but could use a
        # different model with different structured output
        response = self.model_structured_bound.invoke(
            # messages
            [instructions]
        )

        # print(f"Final_Response:\n{response}")
        # Response is structured AgentStatusResponse

        # append the insructions but not the full summary of the history
        # as that would be redundant
        instructions_message = HumanMessage(content=internal_instructions)

        messages.append(instructions_message)

        state["messages"] = messages

        return {"final_response": response}
