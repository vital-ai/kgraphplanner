from typing import Optional
from pydantic import BaseModel, Field
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters


class ToolData(BaseModel):
    """
    Represents the data returned as the response to a tool request.

    Attributes:
        tool_request_guid: A globally unique identifier for the tool data response.
        tool_data_class: The class of the tool data response.
        tool_name: The name of the tool that produced the tool data response.
        tool_parameters: The parameters of calling the tool.
    """

    tool_request_guid: str = Field(..., description="A globally unique identifier for the tool data response")
    tool_data_class: str = Field(..., description="The class of the tool data response")
    tool_name: str = Field(..., description="The name of the tool that produced the tool data response")
    tool_parameters: ToolParameters = Field(..., description="The parameters of calling the tool")
    
    model_config = {"extra": "forbid"}  # Strict validation for this model
