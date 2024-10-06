from typing_extensions import TypedDict


class StructuredResponse(TypedDict):
    """
           Represents a response to a request that contains structured data

           Attributes:
               response_class_name (str): the name of he class of the structured response this is
               The class name should refer to a subclass of StructuredResponse.
    """

    response_class_name: str
