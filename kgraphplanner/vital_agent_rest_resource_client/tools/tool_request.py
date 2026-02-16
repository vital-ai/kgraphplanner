from typing import Optional, Union, Any
from pydantic import BaseModel, Field
from .tool_name_enum import ToolName
from .google_address_validation.models import AddressValidationInput
from .place_search.models import PlaceSearchInput
from .weather.models import WeatherInput
from .web_search.models import WebSearchInput


class ToolRequest(BaseModel):
    """Base tool request model with non-tool-specific parameters"""
    tool: ToolName = Field(..., description="Tool name to execute")
    request_id: Optional[str] = Field(None, description="Optional request identifier")
    timeout: Optional[int] = Field(None, description="Request timeout in seconds")
    tool_input: Union[
        AddressValidationInput, 
        PlaceSearchInput, 
        WeatherInput,
        WebSearchInput
        ] = Field(..., description="Tool-specific input parameters")

    model_config = {"extra": "allow"}
    
    def to_dict(self) -> dict:
        """
        Converts ToolRequest instance to a dictionary.
        
        Returns:
            dict: The dictionary representation of the tool request.
        """
        return self.model_dump(exclude_none=True)
