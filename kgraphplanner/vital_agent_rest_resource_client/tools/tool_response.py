from typing import Optional, Union
from pydantic import BaseModel, Field
from .google_address_validation.models import AddressValidationOutput
from .place_search.models import PlaceSearchOutput
from .weather.models import WeatherOutput
from .web_search.models import WebSearchOutput


class ToolResponse(BaseModel):
    """Base tool response model with non-tool-specific fields"""
    duration: Optional[int] = Field(None, description="Response duration in milliseconds")
    success: bool = Field(..., description="Whether the tool execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    tool_output: Optional[Union[
        AddressValidationOutput, 
        PlaceSearchOutput, 
        WeatherOutput,
        WebSearchOutput
        ]] = Field(None, description="Tool-specific output data")

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def create_success(cls, tool_output, duration_ms: int):
        """Create a successful tool response"""
        return cls(
            duration=duration_ms,
            success=True,
            tool_output=tool_output
        )

    @classmethod
    def create_error(cls, error_message: str, duration_ms: int):
        """Create an error tool response"""
        return cls(
            duration=duration_ms,
            success=False,
            error_message=error_message,
            tool_output=None
        )



