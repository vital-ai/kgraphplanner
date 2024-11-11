from typing import Optional, Annotated, List
from typing_extensions import TypedDict


class AgentStatusResponse(TypedDict):
    """The response and status of a request."""

    human_text_request: Annotated[str, ..., "Restate the exact initial request of the human."]
    agent_text_response: Annotated[str, ..., "Generated response to the human request."]
    agent_request_status: Annotated[str, ..., "The request status, one of: complete, incomplete, missing_input, error, or failure."]
    agent_include_payload: Annotated[bool, ..., "True when the response should include one or more structured responses."]
    agent_payload_class_list: Annotated[List[str], ..., "List of payload class names."]
    agent_payload_guid_list: Annotated[List[str], ..., "List of payload GUIDs."]
    agent_request_status_message: Annotated[Optional[str], None, "Optional message explaining why a request is not complete."]
    missing_input: Annotated[Optional[str], None, "Optional message specifying what input is missing, if any."]
    response_class_name: Annotated[str, ..., "The response class name, which is AgentStatusResponse."]



