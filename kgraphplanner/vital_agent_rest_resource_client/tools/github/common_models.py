"""
Shared request/response models for the github_* tools.

Mirrored from vital-agent-resource-rest at commit 9671939
(vital_agent_resource_app/tools/github/common_models.py). See
planning/kg_tools/github_tools_plan.md section 7 on keeping these in sync --
a field added upstream and missed here parses silently as None.
"""

from pydantic import BaseModel, Field
from typing import Optional


class GitHubRepoBase(BaseModel):
    """Base for every GitHub tool input.

    owner/repo are required on every request -- the service holds no defaults, and
    its repo allowlist check needs something to check.
    """
    owner: str = Field(..., description="Repository owner (user or organization)", min_length=1)
    repo: str = Field(..., description="Repository name", min_length=1)


class GitHubOutputBase(BaseModel):
    """Fields shared by every GitHub tool output."""
    repository: Optional[str] = Field(None, description="Repository the operation targeted, as 'owner/repo'")
    returned_count: Optional[int] = Field(
        None,
        description="How many records this response actually contains. Always trust this "
                    "over total_count when deciding what you received."
    )
    truncated: bool = Field(False, description="True if more results existed than were returned")
    next_page: Optional[int] = Field(
        None,
        description="Page to request for the next batch, or null if there is no more. "
                    "Always use this rather than incrementing `page` yourself."
    )
    api_error: Optional[str] = Field(
        None,
        description="Error message if the GitHub call failed. The service returns expected "
                    "failures (allowlist denial, write gate off, rate limiting, 404) here "
                    "with HTTP 200, so this being set is not a transport failure -- the text "
                    "is written to be actionable and should be surfaced to the caller."
    )
    api_status_code: Optional[int] = Field(None, description="GitHub API status code if the call failed")
    rate_limit_remaining: Optional[int] = Field(None, description="Requests remaining in the current rate limit window")
