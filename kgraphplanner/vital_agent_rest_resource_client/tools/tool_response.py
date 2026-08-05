from typing import Optional, Union
from pydantic import BaseModel, Field
from .google_address_validation.models import AddressValidationOutput
from .place_search.models import PlaceSearchOutput
from .weather.models import WeatherOutput
from .serper_web_search.models import SerperWebSearchOutput
from .web_search.models import WebSearchOutput
from .github.issue_models import GitHubIssueToolOutput
from .github.pr_models import GitHubPRToolOutput
from .github.actions_models import GitHubActionsToolOutput
from .github.code_models import GitHubCodeToolOutput
from .github.repo_models import GitHubRepoToolOutput


class ToolResponse(BaseModel):
    """Base tool response model with non-tool-specific fields"""
    duration: Optional[int] = Field(None, description="Response duration in milliseconds")
    success: bool = Field(..., description="Whether the tool execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    # Every new tool must be added here as well as to ToolRequest's input union --
    # this one is easy to miss, because a missing entry fails at response-parse
    # time with a confusing "not a valid instance of <unrelated model>" error.
    tool_output: Optional[Union[
        AddressValidationOutput,
        PlaceSearchOutput,
        SerperWebSearchOutput,
        WeatherOutput,
        WebSearchOutput,
        GitHubIssueToolOutput,
        GitHubPRToolOutput,
        GitHubActionsToolOutput,
        GitHubCodeToolOutput,
        GitHubRepoToolOutput
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



