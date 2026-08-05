"""
Request/response models for github_pr_tool.

Mirrored from vital-agent-resource-rest at commit `cf9f411`
(vital_agent_resource_app/tools/github/pr_models.py). Field sets verified against
test_data/github/service_schemas.json by tests/test_github_schema_parity.py.

All ten operations (merge_pr moved to github_code_tool at service commit 6543689). create_pr_review(APPROVE) is additionally
gated server-side by ALLOW_PR_MERGE.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from kgraphplanner.vital_agent_rest_resource_client.tools.github.common_models import (
    GitHubRepoBase, GitHubOutputBase
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class GitHubPRListInput(GitHubRepoBase):
    """List pull requests."""
    operation: Literal["list_prs"] = Field(..., description="Operation to perform")
    state: Optional[Literal["open", "closed", "all"]] = Field("open", description="PR state filter")
    head: Optional[str] = Field(None, description="Filter by head branch, as 'user:branch'")
    base: Optional[str] = Field(None, description="Filter by base branch name")
    sort: Optional[Literal["created", "updated", "popularity", "long-running"]] = Field(
        "created", description="Sort field"
    )
    direction: Optional[Literal["asc", "desc"]] = Field("desc", description="Sort direction")
    max_results: Optional[int] = Field(30, description="Maximum PRs to return", ge=1, le=100)
    page: Optional[int] = Field(None, description="Page number for pagination", ge=1)


class GitHubPRGetInput(GitHubRepoBase):
    """Get a single pull request."""
    operation: Literal["get_pr"] = Field(..., description="Operation to perform")
    pr_number: int = Field(..., description="Pull request number", ge=1)


class GitHubPRFilesInput(GitHubRepoBase):
    """List the files a pull request touches."""
    operation: Literal["list_pr_files"] = Field(..., description="Operation to perform")
    pr_number: int = Field(..., description="Pull request number", ge=1)
    include_patch: Optional[bool] = Field(
        False,
        description="Include the diff hunk per file. Off by default -- patches are large "
                    "and blow up an agent's context."
    )
    max_results: Optional[int] = Field(50, description="Maximum files to return", ge=1, le=100)
    page: Optional[int] = Field(None, description="Page number for pagination", ge=1)


class GitHubPRCommentListInput(GitHubRepoBase):
    """List conversation comments on a pull request.

    These are issue comments. Review comments anchored to a diff line are a
    different API, reached through the review endpoints.
    """
    operation: Literal["list_pr_comments"] = Field(..., description="Operation to perform")
    pr_number: int = Field(..., description="Pull request number", ge=1)
    max_results: Optional[int] = Field(30, description="Maximum comments to return", ge=1, le=100)
    page: Optional[int] = Field(None, description="Page number for pagination", ge=1)


class GitHubPRReviewListInput(GitHubRepoBase):
    """List reviews on a pull request."""
    operation: Literal["list_pr_reviews"] = Field(..., description="Operation to perform")
    pr_number: int = Field(..., description="Pull request number", ge=1)
    max_results: Optional[int] = Field(30, description="Maximum reviews to return", ge=1, le=100)
    page: Optional[int] = Field(
        None, description="Page to fetch; this operation reads exactly one page", ge=1
    )


# --- writes ----------------------------------------------------------------

class GitHubPRCreateInput(GitHubRepoBase):
    """Open a new pull request."""
    operation: Literal["create_pr"] = Field(..., description="Operation to perform")
    title: str = Field(..., description="Pull request title", min_length=1)
    head: str = Field(..., description="Branch containing the changes", min_length=1)
    base: str = Field(..., description="Branch to merge into", min_length=1)
    body: Optional[str] = Field(None, description="Pull request description in Markdown")
    draft: Optional[bool] = Field(False, description="Open as a draft")
    maintainer_can_modify: Optional[bool] = Field(None, description="Allow maintainer edits")


class GitHubPRUpdateInput(GitHubRepoBase):
    """Update a pull request. Only the fields provided are sent."""
    operation: Literal["update_pr"] = Field(..., description="Operation to perform")
    pr_number: int = Field(..., description="Pull request number", ge=1)
    title: Optional[str] = Field(None, description="New title", min_length=1)
    body: Optional[str] = Field(None, description="New description in Markdown")
    state: Optional[Literal["open", "closed"]] = Field(None, description="New state")
    base: Optional[str] = Field(None, description="Retarget to this base branch")


class GitHubPRRequestReviewersInput(GitHubRepoBase):
    """Request review from users or teams.

    Reports a reviewer GitHub would not accept -- a login that cannot review the
    repository -- rather than returning success having requested nobody.
    """
    operation: Literal["request_reviewers"] = Field(..., description="Operation to perform")
    pr_number: int = Field(..., description="Pull request number", ge=1)
    reviewers: Optional[List[str]] = Field(None, description="User logins to request review from")
    team_reviewers: Optional[List[str]] = Field(
        None, description="Team slugs to request review from")


class GitHubPRCommentCreateInput(GitHubRepoBase):
    """Add a conversation comment to a pull request."""
    operation: Literal["add_pr_comment"] = Field(..., description="Operation to perform")
    pr_number: int = Field(..., description="Pull request number", ge=1)
    body: str = Field(..., description="Comment body in Markdown", min_length=1)


class GitHubPRReviewCreateInput(GitHubRepoBase):
    """Submit a review on a pull request."""
    operation: Literal["create_pr_review"] = Field(..., description="Operation to perform")
    pr_number: int = Field(..., description="Pull request number", ge=1)
    event: Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"] = Field(
        ..., description="Review verdict"
    )
    body: Optional[str] = Field(None, description="Review body in Markdown")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class GitHubPullRequest(BaseModel):
    number: int = Field(..., description="Pull request number")
    title: str = Field(..., description="Pull request title")
    state: str = Field(..., description="open or closed")
    body: Optional[str] = Field(None, description="Description, truncated if very long")
    body_truncated: bool = Field(False, description="True if the body was truncated")
    html_url: str = Field(..., description="Browser URL for the pull request")
    user: Optional[str] = Field(None, description="Login of the author")
    head: Optional[str] = Field(None, description="Head branch")
    head_sha: Optional[str] = Field(None, description="Head commit sha")
    base: Optional[str] = Field(None, description="Base branch")
    draft: bool = Field(False, description="True if the PR is a draft")
    merged: bool = Field(False, description="True if the PR has been merged")
    mergeable: Optional[bool] = Field(None, description="GitHub's mergeability verdict, if computed")
    mergeable_state: Optional[str] = Field(None, description="Mergeability detail, e.g. clean, dirty")
    labels: List[str] = Field(default_factory=list, description="Label names")
    assignees: List[str] = Field(default_factory=list, description="Logins of assignees")
    requested_reviewers: List[str] = Field(default_factory=list, description="Logins of requested reviewers")
    comments: Optional[int] = Field(None, description="Number of conversation comments")
    review_comments: Optional[int] = Field(None, description="Number of review comments")
    commits: Optional[int] = Field(None, description="Number of commits")
    additions: Optional[int] = Field(None, description="Lines added")
    deletions: Optional[int] = Field(None, description="Lines removed")
    changed_files: Optional[int] = Field(None, description="Number of files changed")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    closed_at: Optional[str] = Field(None, description="Close timestamp if closed")
    merged_at: Optional[str] = Field(None, description="Merge timestamp if merged")


class GitHubPRFile(BaseModel):
    filename: str = Field(..., description="Path of the file")
    status: Optional[str] = Field(None, description="added, modified, removed, renamed")
    additions: int = Field(0, description="Lines added")
    deletions: int = Field(0, description="Lines removed")
    changes: int = Field(0, description="Total lines changed")
    patch: Optional[str] = Field(None, description="Diff hunk, only when include_patch is set")
    patch_truncated: bool = Field(False, description="True if the patch was truncated")
    previous_filename: Optional[str] = Field(None, description="Former path, for renames")


class GitHubPRComment(BaseModel):
    id: int = Field(..., description="Comment id")
    body: Optional[str] = Field(None, description="Comment body, truncated if very long")
    body_truncated: bool = Field(False, description="True if the body was truncated")
    user: Optional[str] = Field(None, description="Login of the author")
    html_url: Optional[str] = Field(None, description="Browser URL for the comment")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class GitHubPRReview(BaseModel):
    id: int = Field(..., description="Review id")
    state: Optional[str] = Field(None, description="APPROVED, CHANGES_REQUESTED, COMMENTED")
    body: Optional[str] = Field(None, description="Review body, truncated if very long")
    body_truncated: bool = Field(False, description="True if the body was truncated")
    user: Optional[str] = Field(None, description="Login of the reviewer")
    html_url: Optional[str] = Field(None, description="Browser URL for the review")
    submitted_at: Optional[str] = Field(None, description="Submission timestamp")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class GitHubPRToolOutput(GitHubOutputBase):
    """Output model for github_pr_tool, shared by all of its operations."""
    tool: Literal["github_pr_tool"] = Field("github_pr_tool", description="Tool identifier")
    operation: str = Field(..., description="Operation that was performed")
    pull_requests: List[GitHubPullRequest] = Field(default_factory=list, description="PRs from list operations")
    pull_request: Optional[GitHubPullRequest] = Field(None, description="PR from single-PR operations")
    files: List[GitHubPRFile] = Field(default_factory=list, description="Files touched by a PR")
    comments: List[GitHubPRComment] = Field(default_factory=list, description="Conversation comments")
    comment: Optional[GitHubPRComment] = Field(None, description="Newly created comment")
    reviews: List[GitHubPRReview] = Field(default_factory=list, description="Reviews on a PR")
    review: Optional[GitHubPRReview] = Field(None, description="Newly created review")
    total_count: Optional[int] = Field(
        None,
        description="Corpus total reported by GitHub, set only where GitHub supplies one."
    )


GITHUB_PR_OPERATION_MODELS = {
    "list_prs": GitHubPRListInput,
    "get_pr": GitHubPRGetInput,
    "list_pr_files": GitHubPRFilesInput,
    "list_pr_comments": GitHubPRCommentListInput,
    "list_pr_reviews": GitHubPRReviewListInput,
    "create_pr": GitHubPRCreateInput,
    "update_pr": GitHubPRUpdateInput,
    "request_reviewers": GitHubPRRequestReviewersInput,
    "add_pr_comment": GitHubPRCommentCreateInput,
    "create_pr_review": GitHubPRReviewCreateInput,
}
