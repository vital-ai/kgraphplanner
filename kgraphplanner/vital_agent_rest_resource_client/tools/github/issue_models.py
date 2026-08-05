"""
Request/response models for github_issue_tool.

Mirrored from vital-agent-resource-rest at commit `cf9f411`
(vital_agent_resource_app/tools/github/issue_models.py).

All nineteen operations. Field sets verified against
test_data/github/service_schemas.json by tests/test_github_schema_parity.py.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from kgraphplanner.vital_agent_rest_resource_client.tools.github.common_models import (
    GitHubRepoBase, GitHubOutputBase
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class GitHubIssueGetInput(GitHubRepoBase):
    """Fetch a single issue by number."""
    operation: Literal["get_issue"] = Field(..., description="Operation to perform")
    issue_number: int = Field(..., description="Issue number", ge=1)


class GitHubIssueListInput(GitHubRepoBase):
    """List issues in a repository."""
    operation: Literal["list_issues"] = Field(..., description="Operation to perform")
    state: Optional[Literal["open", "closed", "all"]] = Field("open", description="Issue state filter")
    labels: Optional[List[str]] = Field(None, description="Only issues carrying all of these labels")
    assignee: Optional[str] = Field(None, description="Login of the assignee, or '*' / 'none'")
    creator: Optional[str] = Field(None, description="Login of the issue author")
    milestone: Optional[str] = Field(None, description="Milestone number, '*' or 'none'")
    since: Optional[str] = Field(None, description="Only issues updated at or after this ISO 8601 timestamp")
    sort: Optional[Literal["created", "updated", "comments"]] = Field("created", description="Sort field")
    direction: Optional[Literal["asc", "desc"]] = Field("desc", description="Sort direction")
    include_pull_requests: Optional[bool] = Field(
        False,
        description="GitHub returns pull requests from the issues endpoint; set true to keep them"
    )
    max_results: Optional[int] = Field(30, description="Maximum issues to return", ge=1, le=100)
    page: Optional[int] = Field(
        None,
        description="Page to start from. This operation may consume several pages to fill "
                    "max_results after filtering out pull requests, so do not assume the "
                    "next batch is page+1 -- pass the next_page value from the response, "
                    "which may repeat a partly-consumed page and can therefore return "
                    "records you have already seen.",
        ge=1
    )


class GitHubIssueCommentListInput(GitHubRepoBase):
    """List comments on an issue."""
    operation: Literal["list_comments"] = Field(..., description="Operation to perform")
    issue_number: int = Field(..., description="Issue number", ge=1)
    since: Optional[str] = Field(None, description="Only comments updated at or after this ISO 8601 timestamp")
    max_results: Optional[int] = Field(30, description="Maximum comments to return", ge=1, le=100)
    page: Optional[int] = Field(
        None, description="Page to fetch; this operation reads exactly one page", ge=1
    )


class GitHubIssueSearchInput(GitHubRepoBase):
    """Search issues within one repository.

    The query is repo-relative: the service prepends the repo qualifier and
    rejects any query containing 'repo:', 'org:' or 'user:', since those would
    escape its allowlist.
    """
    operation: Literal["search_issues"] = Field(..., description="Operation to perform")
    query: str = Field(..., description="GitHub search syntax, repo-relative", min_length=1)
    sort: Optional[Literal["comments", "reactions", "created", "updated"]] = Field(None, description="Sort field")
    order: Optional[Literal["asc", "desc"]] = Field(None, description="Sort direction")
    include_pull_requests: Optional[bool] = Field(
        False, description="Search matches pull requests too; set true to keep them"
    )
    max_results: Optional[int] = Field(30, description="Maximum results to return", ge=1, le=100)
    page: Optional[int] = Field(
        None, description="Page to fetch; this operation reads exactly one page", ge=1
    )


# --- writes ----------------------------------------------------------------

class GitHubIssueCreateInput(GitHubRepoBase):
    """Open a new issue."""
    operation: Literal["create_issue"] = Field(..., description="Operation to perform")
    title: str = Field(..., description="Issue title", min_length=1)
    body: Optional[str] = Field(None, description="Issue body in Markdown")
    labels: Optional[List[str]] = Field(None, description="Labels to apply")
    assignees: Optional[List[str]] = Field(None, description="Logins to assign")
    milestone: Optional[int] = Field(None, description="Milestone number")
    # Supplied by calling code, never by a model -- an LLM inventing an
    # idempotency key defeats the point, which is that it derives
    # deterministically from the source event. Deliberately absent from the
    # agent-facing AGENT_FIELDS (plan section 15.3).
    idempotency_key: Optional[str] = Field(
        None,
        description="Deterministic identifier for the event this issue represents -- an "
                    "alert fingerprint, a message id. Supplying it makes the create "
                    "idempotent: a repeat returns the original issue instead of a duplicate."
    )


class GitHubIssueUpdateInput(GitHubRepoBase):
    """Update an existing issue. Only the fields provided are sent."""
    operation: Literal["update_issue"] = Field(..., description="Operation to perform")
    issue_number: int = Field(..., description="Issue number", ge=1)
    title: Optional[str] = Field(None, description="New title", min_length=1)
    body: Optional[str] = Field(None, description="New body in Markdown")
    state: Optional[Literal["open", "closed"]] = Field(None, description="New state")
    state_reason: Optional[Literal["completed", "not_planned", "reopened"]] = Field(
        None, description="Reason accompanying a state change"
    )
    labels: Optional[List[str]] = Field(None, description="Replace labels with this set")
    assignees: Optional[List[str]] = Field(None, description="Replace assignees with this set")
    milestone: Optional[int] = Field(None, description="Milestone number")


class GitHubIssueCloseInput(GitHubRepoBase):
    """Close an issue.

    GitHub's REST API has no delete-issue endpoint -- closing is the
    delete-equivalent.
    """
    operation: Literal["close_issue"] = Field(..., description="Operation to perform")
    issue_number: int = Field(..., description="Issue number", ge=1)
    state_reason: Optional[Literal["completed", "not_planned", "duplicate"]] = Field(
        "completed", description="Why the issue is being closed"
    )
    comment: Optional[str] = Field(None, description="Comment to add alongside the close")


class GitHubIssueReopenInput(GitHubRepoBase):
    """Reopen a closed issue."""
    operation: Literal["reopen_issue"] = Field(..., description="Operation to perform")
    issue_number: int = Field(..., description="Issue number", ge=1)


class GitHubIssueCommentCreateInput(GitHubRepoBase):
    """Add a comment to an issue."""
    operation: Literal["add_comment"] = Field(..., description="Operation to perform")
    issue_number: int = Field(..., description="Issue number", ge=1)
    body: str = Field(..., description="Comment body in Markdown", min_length=1)


class GitHubIssueCommentUpdateInput(GitHubRepoBase):
    """Edit an existing issue comment."""
    operation: Literal["update_comment"] = Field(..., description="Operation to perform")
    comment_id: int = Field(..., description="Comment id", ge=1)
    body: str = Field(..., description="Replacement body in Markdown", min_length=1)


class GitHubIssueCommentDeleteInput(GitHubRepoBase):
    """Delete an issue comment. Unlike issues, comments can be deleted outright."""
    operation: Literal["delete_comment"] = Field(..., description="Operation to perform")
    comment_id: int = Field(..., description="Comment id", ge=1)


class GitHubIssueAddLabelsInput(GitHubRepoBase):
    """Add labels to an issue, leaving existing ones in place."""
    operation: Literal["add_labels"] = Field(..., description="Operation to perform")
    issue_number: int = Field(..., description="Issue number", ge=1)
    labels: List[str] = Field(..., description="Labels to add", min_length=1)
    validate_labels: Optional[bool] = Field(
        None,
        description="Reject names that do not already exist on the repository. GitHub "
                    "silently creates unknown labels otherwise, so a typo becomes a new "
                    "label rather than an error."
    )


class GitHubIssueRemoveLabelsInput(GitHubRepoBase):
    """Remove labels from an issue. Removing an absent label is a no-op."""
    operation: Literal["remove_labels"] = Field(..., description="Operation to perform")
    issue_number: int = Field(..., description="Issue number", ge=1)
    labels: List[str] = Field(..., description="Labels to remove", min_length=1)


class GitHubIssueAddAssigneesInput(GitHubRepoBase):
    """Assign users to an issue."""
    operation: Literal["add_assignees"] = Field(..., description="Operation to perform")
    issue_number: int = Field(..., description="Issue number", ge=1)
    assignees: List[str] = Field(..., description="Logins to assign", min_length=1)


class GitHubIssueRemoveAssigneesInput(GitHubRepoBase):
    """Unassign users from an issue."""
    operation: Literal["remove_assignees"] = Field(..., description="Operation to perform")
    issue_number: int = Field(..., description="Issue number", ge=1)
    assignees: List[str] = Field(..., description="Logins to unassign", min_length=1)


class GitHubIssueFindByBodyInput(GitHubRepoBase):
    """Scan issue bodies for a marker, matching in code rather than via search.

    Exists because GitHub's search index lags creation by roughly a minute, and
    "have I already filed this?" is always asked inside exactly that window.
    The scan reads bodies the service already fetches and returns only matches,
    so no body reaches the caller.

    Note the defaults differ from `list_issues` deliberately: `state` is "all"
    because a closed duplicate is still a duplicate, and `match` is "line"
    because marker conventions are line-anchored.
    """
    operation: Literal["find_issues_by_body"] = Field(..., description="Operation to perform")
    contains: str = Field(..., description="Marker text to look for in issue bodies", min_length=1)
    match: Optional[Literal["substring", "line"]] = Field(
        "line", description="How `contains` must appear; 'line' matches a whole stripped line")
    state: Optional[Literal["open", "closed", "all"]] = Field(
        "all", description="Defaults to 'all', unlike list_issues")
    labels: Optional[List[str]] = Field(None, description="Narrow the scan to these labels")
    since: Optional[str] = Field(
        None,
        description="Only issues updated at or after this ISO 8601 timestamp. Filters on "
                    "updated_at, NOT created_at, so it is unsafe as a duplicate-detection "
                    "bound in either direction. Use it to speed up polling, never to scope "
                    "a duplicate check."
    )
    include_pull_requests: Optional[bool] = Field(
        False, description="GitHub returns pull requests from the issues endpoint")
    max_pages: Optional[int] = Field(5, description="Pages of 100 to scan before giving up", ge=1, le=20)
    max_results: Optional[int] = Field(10, description="Stop after this many matches", ge=1, le=100)
    page: Optional[int] = Field(
        None,
        description="Page to start scanning from. When a scan returns complete=false it also "
                    "returns next_page; passing that back resumes where the budget ran out.",
        ge=1
    )


class GitHubIssueListLabelsInput(GitHubRepoBase):
    """List the labels defined on the repository.

    The vocabulary side of `validate_labels`: an agent can discover which names
    exist rather than guessing and creating a stray label.
    """
    operation: Literal["list_labels"] = Field(..., description="Operation to perform")
    max_results: Optional[int] = Field(100, description="Maximum labels to return", ge=1, le=100)
    page: Optional[int] = Field(
        None, description="Page to fetch; this operation reads exactly one page", ge=1)


class GitHubIssueListMilestonesInput(GitHubRepoBase):
    """List milestones. `create_issue` takes a milestone number, not a title."""
    operation: Literal["list_milestones"] = Field(..., description="Operation to perform")
    state: Optional[Literal["open", "closed", "all"]] = Field(
        "open", description="Milestone state")
    max_results: Optional[int] = Field(30, description="Maximum milestones to return", ge=1, le=100)
    page: Optional[int] = Field(
        None, description="Page to fetch; this operation reads exactly one page", ge=1)


class GitHubIssueListAssignableUsersInput(GitHubRepoBase):
    """List logins that can be assigned to an issue in this repository."""
    operation: Literal["list_assignable_users"] = Field(..., description="Operation to perform")
    max_results: Optional[int] = Field(50, description="Maximum users to return", ge=1, le=100)
    page: Optional[int] = Field(
        None, description="Page to fetch; this operation reads exactly one page", ge=1)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class GitHubLabel(BaseModel):
    name: str = Field(..., description="Label name, as passed to add_labels")
    description: Optional[str] = Field(None, description="Label description")
    color: Optional[str] = Field(None, description="Hex colour, without the leading #")


class GitHubMilestone(BaseModel):
    number: int = Field(..., description="Milestone number, as passed to create_issue")
    title: str = Field(..., description="Milestone title")
    state: Optional[str] = Field(None, description="open or closed")
    description: Optional[str] = Field(None, description="Milestone description")
    open_issues: Optional[int] = Field(None, description="Open issues in this milestone")
    closed_issues: Optional[int] = Field(None, description="Closed issues in this milestone")
    due_on: Optional[str] = Field(None, description="Due date, if set")


class GitHubIssue(BaseModel):
    number: int = Field(..., description="Issue number")
    title: str = Field(..., description="Issue title")
    state: str = Field(..., description="open or closed")
    state_reason: Optional[str] = Field(None, description="Reason for the current state")
    body: Optional[str] = Field(None, description="Issue body, truncated if very long")
    body_truncated: bool = Field(False, description="True if the body was truncated")
    html_url: str = Field(..., description="Browser URL for the issue")
    user: Optional[str] = Field(None, description="Login of the issue author")
    assignees: List[str] = Field(default_factory=list, description="Logins of assignees")
    labels: List[str] = Field(default_factory=list, description="Label names")
    milestone: Optional[str] = Field(None, description="Milestone title")
    comments: int = Field(0, description="Number of comments")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    closed_at: Optional[str] = Field(None, description="Close timestamp if closed")
    is_pull_request: bool = Field(False, description="True if this record is actually a pull request")


class GitHubComment(BaseModel):
    id: int = Field(..., description="Comment id")
    body: Optional[str] = Field(None, description="Comment body, truncated if very long")
    body_truncated: bool = Field(False, description="True if the body was truncated")
    user: Optional[str] = Field(None, description="Login of the comment author")
    html_url: Optional[str] = Field(None, description="Browser URL for the comment")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class GitHubIssueToolOutput(GitHubOutputBase):
    """Output model for github_issue_tool, shared by all of its operations."""
    tool: Literal["github_issue_tool"] = Field("github_issue_tool", description="Tool identifier")
    operation: str = Field(..., description="Operation that was performed")
    issues: List[GitHubIssue] = Field(default_factory=list, description="Issues from list/search operations")
    issue: Optional[GitHubIssue] = Field(None, description="Issue from single-issue operations")
    comments: List[GitHubComment] = Field(default_factory=list, description="Comments from list operations")
    comment: Optional[GitHubComment] = Field(None, description="Comment from single-comment operations")
    deleted_id: Optional[int] = Field(None, description="Id of a deleted resource")
    labels: List[GitHubLabel] = Field(
        default_factory=list, description="Repository labels, from list_labels")
    milestones: List[GitHubMilestone] = Field(
        default_factory=list, description="Repository milestones, from list_milestones")
    assignable_users: List[str] = Field(
        default_factory=list, description="Logins that can be assigned, from list_assignable_users")
    created: Optional[bool] = Field(
        None,
        description="create_issue only. False when an existing issue was returned instead "
                    "of filing a new one."
    )
    idempotency_guard: Optional[str] = Field(
        None,
        description="Which mechanism answered: 'memorydb' (guaranteed), 'scan' (reconciled "
                    "against GitHub), 'none' (a guarantee was requested and could not be "
                    "provided). Null means no key was supplied, so none was requested -- "
                    "distinct from 'none'."
    )
    scanned: Optional[int] = Field(None, description="Issues examined by find_issues_by_body")
    complete: Optional[bool] = Field(
        None,
        description="find_issues_by_body only. True when the scan reached the end of the "
                    "matching issues; False when the page budget ran out first, in which "
                    "case an empty result does NOT establish absence."
    )
    total_count: Optional[int] = Field(
        None,
        description="Corpus total reported by GitHub, set only for search. Counts records "
                    "before the service's own filtering, so it can exceed returned_count."
    )


GITHUB_ISSUE_OPERATION_MODELS = {
    "get_issue": GitHubIssueGetInput,
    "list_issues": GitHubIssueListInput,
    "list_comments": GitHubIssueCommentListInput,
    "search_issues": GitHubIssueSearchInput,
    "create_issue": GitHubIssueCreateInput,
    "update_issue": GitHubIssueUpdateInput,
    "close_issue": GitHubIssueCloseInput,
    "reopen_issue": GitHubIssueReopenInput,
    "add_comment": GitHubIssueCommentCreateInput,
    "update_comment": GitHubIssueCommentUpdateInput,
    "delete_comment": GitHubIssueCommentDeleteInput,
    "add_labels": GitHubIssueAddLabelsInput,
    "remove_labels": GitHubIssueRemoveLabelsInput,
    "add_assignees": GitHubIssueAddAssigneesInput,
    "remove_assignees": GitHubIssueRemoveAssigneesInput,
    "list_labels": GitHubIssueListLabelsInput,
    "list_milestones": GitHubIssueListMilestonesInput,
    "list_assignable_users": GitHubIssueListAssignableUsersInput,
    "find_issues_by_body": GitHubIssueFindByBodyInput,
}
