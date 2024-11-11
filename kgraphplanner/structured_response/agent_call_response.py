from typing import Annotated, List
from typing_extensions import TypedDict
from kgraphplanner.inter.base_agent_schema import BaseAgentRequest


class AgentCallCapture(TypedDict):
    agent_call_guid: Annotated[str, ..., "The guid of the agent request being called"]
    agent_request: Annotated[BaseAgentRequest, ..., "The parameters of the agent request being called which must extend BaseAgentRequest"]


class AgentCallResponse(TypedDict):
    """Represents the response of a request to an A.I. Agent where the agent will call other agents to help service the request.."""

    agent_call_list: Annotated[List[AgentCallCapture], ..., "List of agent requests captured, or an empty list if none."]
    response_class_name: Annotated[str, ..., "The response class name, which is AgentCallResponse."]

