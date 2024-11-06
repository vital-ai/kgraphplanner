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
from langgraph.graph import END, StateGraph, add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.managed import IsLastStep
from kgraphplanner.agent.tool_executor import ToolExecutor
from kgraphplanner.agent.tool_node import ToolNode
from kgraphplanner.structured_response.agent_status_response import AgentStatusResponse
from kgraphplanner.structured_response.weather_response import WeatherResponse
from kgraphplanner.tools_internal.capture.capture_response_tool import CaptureResponseTool
from kgraphplanner.agent.kg_planning_base_agent import KGPlanningBaseAgent, StateSchemaType, StateModifier, \
    MessagesModifier, AgentState


class AgentInternalState(AgentState):
    """The state of the agent with internal step."""

    # is there a better way to track this?
    # this is for the internal case to remember if instructions
    # were put into the messages
    respond_instructions: bool


class KGPlanningInternalStructuredAgent(KGPlanningBaseAgent):
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

        capture_response_tool = CaptureResponseTool({})

        self.internal_tools = [
            capture_response_tool.get_tool_function()
        ]

        self.model_internal_tools = self.model_tools.bind_tools(self.internal_tools)

        self.model_structured_bound = self.model_structured.with_structured_output(AgentStatusResponse)

        if self.state_schema is not None:
            if missing_keys := {"messages", "is_last_step"} - set(
                    self.state_schema.__annotations__
            ):
                raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

        if isinstance(self.tools, ToolExecutor):
            self.tool_classes = self.tools.tools
        else:
            self.tool_classes = self.tools

        self.model_tools = self.model_tools.bind_tools(self.tool_classes)

        self.preprocessor = self._get_model_preprocessing_runnable(self.state_modifier, self.messages_modifier)

        self.model_runnable = self.preprocessor | self.model_tools

        self.model_internal_runnable = self.preprocessor | self.model_internal_tools

        self.workflow = StateGraph(self.state_schema or AgentInternalState)

        self.workflow.add_node("agent", RunnableLambda(self.call_model, self.acall_model))

        self.workflow.add_node("respond", RunnableLambda(self.respond))

        self.workflow.add_node("tools", ToolNode(self.tool_classes))

        self.workflow.add_node("internal_tools", ToolNode(self.internal_tools))

        self.workflow.add_node("end_state", RunnableLambda(self.end_state))

        self.workflow.set_entry_point("agent")

        self.workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "respond": "respond"
            },
        )

        self.workflow.add_edge("tools", "agent")

        self.workflow.add_conditional_edges(
            "respond",
            self.should_respond_continue,
            {
                "continue": "internal_tools",
                "end_state": "end_state"
            },
        )

        self.workflow.add_edge("internal_tools", "respond")

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

    def should_continue(self, state: AgentInternalState):
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "respond"
        else:
            return "continue"

    def should_respond_continue(self, state: AgentInternalState):
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end_state"
        else:
            return "continue"

    def respond(self, state: AgentInternalState, config: RunnableConfig):

        print(f"State: {state}")
        print(f"Config: {config}")

        # list of available structured responses to be passed in
        # when creating the graph
        # this could be a choice of one for cases like "weather"
        # if the exec agent has already decided to call the weather agent
        # then it could have already decided it wants a weather report back

        # however even if it is a choice of one its possible that no
        # response will be produced if the underlying request could not be completed
        # due to missing input, error, or some other reason

        # use tools to generate structured output
        # the input can be guids of the tool output so we don't have to run it through the LLM again
        # tools can simply capture the existing tool output to use downstream

        # or potentially transform the tool output into a new structured form,
        # such as rendering for a UI which could just be captured and not sent through the LLM
        # the transform could be implemented within a tool programmatically or use a separate LLM call

        # add instructions the first time entering this
        if not state.get("respond_instructions"):
            messages = state["messages"]

            # this gets passed in
            available_structured_responses = [
                "WeatherData: Use for Weather Reports"
            ]

            structured_response_bullet_list = "\n".join(
                f"\t\t\t\t\t* {item}" for item in available_structured_responses)

            instruction_text = f"""
            (internal instructions)
                The available structured responses are:
{structured_response_bullet_list}

                You decide if one or more of the above structured responses should be included in the response to the request
                that was specified in the previous messages.  You may include multiple instances of a structured response.
                The data record of a structured response must be found in the output of a previous tool call.
                You include a structured response only if it is relevant, and no structured response if none are relevant.

                For structured responses you wish to include, you must call the capture_response tool.
                Group the structured responses by class and call the tool with a list of guids of that class.
                Once capture_response is called for the structured responses, if any, the task is complete.
            """

            instructions = HumanMessage(content=instruction_text)

            messages.append(instructions)

            state["messages"] = messages

            state["respond_instructions"] = True

        response = self.model_internal_runnable.invoke(state, config)

        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }

        return {"messages": [response]}

    def end_state(self, state: AgentInternalState, config: RunnableConfig):

        # this one does not re-enter so will only add instructions once

        messages = state["messages"]

        instruction_text = """
        (internal instructions)

            You fill out the fields of AgentStatusResponse based on the previous messages.

            human_text_request: This should be the initial request from the human  
            agent_text_response: This should be the text of the response to the initial request. 
            agent_request_status: this status of the initial request
            agent_include_payload: this should be True if you captured any structured output.
            agent_payload_class_list: this should be the list of the class names used in the calls to the capture_tool, if any.
            agent_payload_guid_list: this should be the list of guids that were used in the captured responses calls to the capture_tool, if any.
            agent_request_status_message: an optional message used if the status is not complete to explain why
            missing_input: an optional message for when the status is missing_input to list what is missing                   
        """

        instructions = HumanMessage(content=instruction_text)

        messages.append(instructions)

        # this is bound to AgentStatusResponse but could use a
        # different model with different structured output
        response = self.model_structured_bound.invoke(
            messages
        )

        # print(f"Final_Response:\n{response}")
        # Response is structured AgentStatusResponse

        state["messages"] = messages

        return {"final_response": response}
