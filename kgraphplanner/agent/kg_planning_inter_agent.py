import json
import logging
import queue
from typing import (
    Optional,
    Sequence,
    Union,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    HumanMessage, ToolMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, Tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from kgraphplanner.agent.tool_executor import ToolExecutor
from kgraphplanner.agent.tool_node import ToolNode
from kgraphplanner.inter.base_agent_schema import BaseAgentRequest
from kgraphplanner.structured_response.agent_call_response import AgentCallResponse, AgentCallCapture
from kgraphplanner.structured_response.agent_payload_response import AgentPayloadResponse
from kgraphplanner.agent.kg_planning_base_agent import KGPlanningBaseAgent, StateSchemaType, StateModifier, \
    MessagesModifier, AgentState
from kgraphplanner.tools_internal.agent.call_agent_tool import CallAgentTool


class KGPlanningInterAgent(KGPlanningBaseAgent):
    def __init__(self, *,
                 model_tools: LanguageModelLike,
                 model_structured: LanguageModelLike,
                 tools: Union[ToolExecutor, Sequence[BaseTool]],
                 reasoning_queue: queue.Queue = None,
                 state_schema: Optional[StateSchemaType] = None, messages_modifier: Optional[MessagesModifier] = None,
                 state_modifier: Optional[StateModifier] = None, checkpointer: Optional[BaseCheckpointSaver] = None,
                 interrupt_before: Optional[Sequence[str]] = None, interrupt_after: Optional[Sequence[str]] = None,
                 debug: bool = False):

        super().__init__()
        self.initialized = False

        self.model_tools = model_tools
        self.model_structured = model_structured

        self.tools = tools

        self.call_agent_tool = CallAgentTool({})

        call_agent_tool_func = self.call_agent_tool.get_tool_function()

        self.internal_tools = [
            call_agent_tool_func
        ]

        self.reasoning_queue = reasoning_queue

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

        self.merged_tool_classes = []

        if isinstance(self.tools, ToolExecutor):
            self.tool_classes = self.tools.tools
            self.merged_tool_classes.extend(self.tool_classes)
        else:
            self.tool_classes = self.tools
            self.merged_tool_classes.extend(self.tool_classes)

        self.merged_tool_classes.extend(self.internal_tools)

        self.model_tools = self.model_tools.bind_tools(self.merged_tool_classes, parallel_tool_calls=True)

        self.preprocessor = self._get_model_preprocessing_runnable(self.state_modifier, self.messages_modifier)

        self.model_runnable = self.preprocessor | self.model_tools

        self.workflow = StateGraph(self.state_schema or AgentState)

        # self.workflow.add_node("agent", RunnableLambda(self.call_model, self.acall_model))

        self.workflow.add_node("agent", RunnableLambda(self.agent, self.async_agent))

        self.workflow.add_node("tools", ToolNode(
            self.merged_tool_classes,
            reasoning_queue=self.reasoning_queue
        ))

        self.workflow.add_node("end_state", RunnableLambda(self.end_state))

        self.workflow.add_node("agent_call", RunnableLambda(self.agent_call))

        self.workflow.set_entry_point("agent")

        self.workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end_state": "end_state",
                "agent_call": "agent_call"
            },
        )

        self.workflow.add_edge("tools", "agent")

        self.workflow.add_edge("end_state", END)

        self.workflow.add_edge("agent_call", END)

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

    def post_to_reasoning_queue(self, message: dict):
        self.reasoning_queue.put(message)

    def agent(self, state: AgentState, config: RunnableConfig):

        reasoning_message = {
            "agent_thought": "calling model"
        }

        self.post_to_reasoning_queue(reasoning_message)

        return self.call_model(state, config)

    async def async_agent(self, state: AgentState, config: RunnableConfig):

        reasoning_message = {
            "agent_thought": "calling model"
        }

        self.post_to_reasoning_queue(reasoning_message)

        return await self.acall_model(state, config)

    def should_continue(self, state: AgentState):

        logger = logging.getLogger("HaleyAgentLogger")

        messages = state["messages"]

        for m in messages:
            t = type(m)
            logging.info(f"should_continue: History ({t}): {m}")

        last_message = messages[-1]

        if not last_message.tool_calls:

            reasoning_message = {
                "agent_thought": "completed calling tools."
            }

            self.post_to_reasoning_queue(reasoning_message)

            if len(self.call_agent_tool.get_agent_call_list()) > 0:

                reasoning_message = {
                    "agent_thought": "there are pending agent requests."
                }

                self.post_to_reasoning_queue(reasoning_message)

                return "agent_call"

            else:
                reasoning_message = {
                    "agent_thought": "there are no pending agent requests."
                }

                self.post_to_reasoning_queue(reasoning_message)

            return "end_state"
        else:
            reasoning_message = {
                "agent_thought": "deciding to continue calling tools."
            }

            self.post_to_reasoning_queue(reasoning_message)

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

        logger = logging.getLogger("HaleyAgentLogger")


        # print(f"State: {state}")
        # print(f"Config: {config}")

        messages = state["messages"]

        reasoning_message = {
            "agent_thought": "composing final response"
        }

        self.post_to_reasoning_queue(reasoning_message)

        determine_payloads = True  # False

        if not determine_payloads:

            for m in messages:
                t = type(m)
                logger.info(f"end_state: History ({t}): {m}")

            system_message = messages[0]

            human_message = messages[1]

            human_text = human_message.content

            last_message = messages[-1]

            ai_message = last_message

            ai_message_text = ai_message.content

            reasoning_message = {
                "agent_thought": "sending final response."
            }

            self.post_to_reasoning_queue(reasoning_message)

            agent_payload_response = AgentPayloadResponse(
                human_text_request=human_text,
                agent_text_response=ai_message_text,
                agent_request_status="complete",
                agent_payload_list=[],
                response_class_name="AgentPayloadResponse",
                missing_input=None,
                agent_request_status_message=None
            )

            return {
                "final_response": agent_payload_response
            }

        message_content = ""

        for message in messages:
            if isinstance(message, HumanMessage):
                content = message.content
                message_content += ('Human Message: ' + content + "\n")
            if isinstance(message, AIMessage):
                content = message.content
                # skip the tool calls that are empty content
                if content:
                    message_content += ('AI Message: ' + content + "\n")
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
                    message_content += ('Tool Result:\n' + pretty_string + "\n--------------------\n")

        available_structured_responses = [
            "NewsArticleData: Use for a News Article",
            "WeatherData: Use for Weather Reports",
            "WebSearchData: Use for Web Search Results",
            "ProductData: Use for information about a Shopping Store Product"
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

        # print("Instruction Text:")
        # print(instruction_text)

        instructions = HumanMessage(content=instruction_text)

        reasoning_message = {
            "agent_thought": "deciding final response payloads."
        }

        self.post_to_reasoning_queue(reasoning_message)

        response = self.model_structured_bound.invoke(
            [instructions]
        )

        instructions_message = HumanMessage(content=internal_instructions)

        new_messages = []

        for m in messages:
            new_messages.append(m)

        new_messages.append(instructions_message)

        state["messages"] = new_messages

        reasoning_message = {
            "agent_thought": "sending final response."
        }

        self.post_to_reasoning_queue(reasoning_message)

        return {"final_response": response}

    def agent_call(self, state: AgentState, config: RunnableConfig):

        messages = state["messages"]

        reasoning_message = {
            "agent_thought": "composing agent call final response"
        }

        self.post_to_reasoning_queue(reasoning_message)

        capture_list = self.call_agent_tool.get_agent_call_list()

        agent_call_list = []

        for capture in capture_list:
            agent_call_guid: str|None = capture.get('guid', None)
            agent_request: BaseAgentRequest|None = capture.get('agent_request', None)

            agent_call_capture = AgentCallCapture(
                agent_call_guid=agent_call_guid,
                agent_request=agent_request
            )

            agent_call_list.append(agent_call_capture)

        agent_call_response = AgentCallResponse(
            response_class_name="AgentCallResponse",
            agent_call_list=agent_call_list,
        )

        reasoning_message = {
            "agent_thought": "sending agent call final response."
        }

        self.post_to_reasoning_queue(reasoning_message)

        return {
            "final_response": agent_call_response
        }
