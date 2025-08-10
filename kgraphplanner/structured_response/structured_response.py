from typing import Annotated
from typing import TypedDict


class StructuredResponse(TypedDict):
    """Represents a response to a request that contains structured data."""

    response_class_name: Annotated[str, ..., "The name of the class of the structured response. This should refer to a subclass of StructuredResponse."]
