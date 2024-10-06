from typing import Optional

from kgraphplanner.structured_response.structured_response import StructuredResponse


class AgentStatusResponse(StructuredResponse):
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
            agent_include_payload (bool): true when the response should include
                one or more structured responses
            agent_payload_class_list (list[str]): list of payload class names
            agent_payload_guid_list (list[str]): the list payload guids
            agent_request_status_message (str): optional message for why a request is not complete
            missing_input (str): an optional message for what input is missing, if any
    """

    human_text_request: str
    agent_text_response: str
    agent_request_status: str
    agent_include_payload: bool
    agent_payload_class_list: list[str]
    agent_payload_guid_list: list[str]
    agent_request_status_message: Optional[str]
    missing_input: Optional[str]



