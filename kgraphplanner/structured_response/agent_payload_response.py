from typing import Optional, Annotated, List

from typing import TypedDict

from kgraphplanner.structured_response.structured_response import StructuredResponse


class AgentPayload(TypedDict):
    """Represents a payload to include in the response to the request."""

    payload_class_name: Annotated[str, ..., "The class name, which is a subclass of TypedDict."]
    payload_guid: Annotated[str, ..., "The identifier of the data record for this payload."]


class AgentPayloadResponse(TypedDict):
    """Represents the response and status of a request to an A.I. Agent."""

    human_text_request: Annotated[str, ..., "Restate the exact initial request of the human."]
    agent_text_response: Annotated[str, ..., "Generated response to the human request."]
    agent_request_status: Annotated[
        str, ..., "The request status, one of: complete, incomplete, missing_input, error, or failure."]
    agent_payload_list: Annotated[List[AgentPayload], ..., "List of payloads to include, or an empty list if none."]
    agent_request_status_message: Annotated[
        Optional[str], None, "Optional message explaining why a request is not complete."]
    missing_input: Annotated[Optional[str], None, "Optional message specifying what input is missing, if any."]
    response_class_name: Annotated[str, ..., "The response class name, which is AgentPayloadResponse."]
