from typing import Optional

from typing_extensions import TypedDict

from kgraphplanner.structured_response.structured_response import StructuredResponse


class AgentPayload(TypedDict):
    """
    Represents a payload to include in the response to the request.
    Attributes:
        payload_class_name (str): The class name, which is a subclass of TypedDict
        payload_guid (str): The identifier of the data record for this payload
    """
    payload_class_name: str
    payload_guid: str


class AgentPayloadResponse(StructuredResponse):
    """
    Represents the response and status of a request to an A.I. Agent.
        Attributes:
            human_text_request (str): Restate the exact initial request of the human
            agent_text_response (str): generated response to the human request
            agent_request_status (str): the request status
                One of:
                    complete: request is complete
                    incomplete: request is incomplete and should be continued
                    missing_input: request is pending due to missing input
                    error: error occurred, re-try is possible
                    failure: failure occurred, re-try is not possible
            agent_payload_list (list[AgentPayload]): list of payloads to include, or empty list
            agent_request_status_message (str): optional message for why a request is not complete
            missing_input (str): an optional message for what input is missing, if any
            response_class_name (str) is AgentPayloadResponse.
    """

    human_text_request: str
    agent_text_response: str
    agent_request_status: str
    agent_payload_list: list[AgentPayload]
    agent_request_status_message: Optional[str]
    missing_input: Optional[str]
