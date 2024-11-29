import json
import logging
import uuid
from typing import Callable, List, Dict, Annotated, Union, Type

# from kgraphplanner.inter.agents.agent_weather.agent_weather_schema import AgentWeatherRequest
# from kgraphplanner.inter.base_agent_schema import BaseAgentRequest
from kgraphplanner.tool_manager.abstract_tool import AbstractTool
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tool_manager.tool_response import ToolResponse
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class BaseAgentRequest(BaseModel):
    request_class_name: str = Field(..., description="The name of the request class.")
    agent_name: str = Field(..., description="The name of the agent.")
    agent_call_justification: str = Field(..., description="The justification for calling the agent.")


class AgentWeatherRequest(BaseAgentRequest):
    place_name_list: List[str] = Field(..., description="The list of place names of the weather locations")


class CallAgentCapture(BaseModel):
    """Arguments for capturing a request to an Agent."""
    agent_request: BaseAgentRequest = Field(..., description="The details of the Agent request which must be an instance of a subclass of BaseAgentRequest")

# AgentRequestType = Union[BaseAgentRequest, AgentWeatherRequest]

# AgentRequestType = Union[AgentWeatherRequest]


class CallAgentTool(AbstractTool):

    def __init__(self, config: dict,
                 tool_manager=None,
                 name: str = None,
                 description: str = None):

        super().__init__(config, tool_manager, name, description)

        self.call_list: List[dict] = []

    def get_agent_call_list(self) -> List[dict]:
        return self.call_list

    def clear_call_list(self):
        self.call_list = []

    def handle_request(self, request: ToolRequest) -> ToolResponse:
        pass

    def get_sample_text(self) -> str:
        pass

    def get_tool_schema(self) -> Type[BaseModel]:
        return CallAgentCapture

    def get_tool_function(self) -> Callable:

        # Note: this should be a dict but is being passed in as a string sometimes
        # the response back from the LLM is a quoted string and it's just not
        # converted into a dict before calling the function
        # potentially Tool Executor, Tool Node, Tool Validator needs updating
        # seems to handle base types but not dicts or more complex things for tool inputs
        # but structured outputs work ok, like the weather report

        # it seems to be confused when a dict is provided as it's looking for
        # a key called "agent_request"
        # but it lets through the base type of a string
        # so it must be confused when it's trying to unpack a map to
        # match the top level parameter names

        # specifying the base class means that the fields in
        # classes that extend it don't get set
        # even tho the tool call has them
        # def call_agent(agent_request: BaseAgentRequest) -> str:

        # @tool(args_schema=CallAgentCapture)
        # def call_agent(agent_request: BaseAgentRequest|str) -> str:
        @tool("call-agent-tool", args_schema=CallAgentCapture, return_direct=True)
        def call_agent( agent_request: Type[BaseAgentRequest]) -> str:
            """
            Use this to capture a requested call to an agent.

            Returns str: The GUID of the requested call to the agent
            """

            logger = logging.getLogger("HaleyAgentLogger")

            logger.info("call agent tool")

            # agent_request = {key: value for key, value in kwargs.items()}

            # agent_request = {
            #    "request_class_name": request_class_name,
            #    "agent_name": agent_name,
            #    "agent_call_justification": agent_call_justification,
            #    "place_name_list": place_name_list
            # }

            # need to validate against tool schemas
            #if type(agent_request) is str:
            #    agent_request = json.loads(agent_request)

            guid = str(uuid.uuid4())

            logger.info(f"CallAgentTool called with agent request: {agent_request}")
            logger.info(f"CallAgentTool assigned agent request: {guid}")

            agent_call = {
                "guid": guid,
                "agent_request": agent_request
            }

            self.call_list.append(agent_call)

            return guid

        return call_agent

# from pydantic import BaseModel

# class CallAgentCapture(BaseModel):
#   """Arguments for capturing a request to an Agent."""
    # agent_name: Annotated[str, ..., "The name of the agent"]
    # agent_call_justification: Annotated[str, ...,  "Provide a justification for calling the agent including the goal"]
#    agent_request: Annotated[BaseAgentRequest, ..., "The details of the Agent request which must be in a subclass of BaseAgentRequest"]
