from typing import Annotated, Optional
from typing_extensions import TypedDict


class BaseAgentRequest(TypedDict):
    """Represents the BaseClass to collect the parameters for a request to an Agent"""
    request_class_name: Annotated[str, ..., "The name of the class of the request, which must be a subclass of BaseAgentRequest."]
    agent_name: Annotated[str, ..., "The name of the agent being called."]
    agent_call_justification: Annotated[str, ..., "A justification for why the agent being called to complete a goal."]


class BaseAgentResponse(TypedDict):
    """Represents the BaseClass for the response of a request to an Agent"""

    agent_name: Annotated[str, ..., "The name of the agent that was called and is responding."]
    request_guid: Annotated[str, ..., "The GUID of the request to the agent."]

    response_class_name: Annotated[str, ..., "The name of the class of the response, which must be a subclass of BaseAgentResponse."]
    status: Annotated[str, ..., "The status of producing a response to a request. One of: OK, ERROR"]
    status_message: Annotated[Optional[str], None, "A status message when there is an error to explain the problem."]
