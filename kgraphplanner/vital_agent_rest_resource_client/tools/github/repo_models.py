"""
Request/response models for github_repo_tool.

Mirrored from vital-agent-resource-rest at commit `cf9f411`
(vital_agent_resource_app/tools/github/repo_models.py).

`github_repo_tool` is **read-only by construction** (service commit `6543689`):
the operations that alter code moved to `github_code_tool`, so registering this
tool grants an agent the ability to read a repository and nothing more. The
schema-level split means the boundary holds regardless of how the config gates
are set.

Closes two gaps reported from this side (plan section 17.1): `get_repo` supplies
the default branch, so `create_pr` no longer needs the agent to guess `main`,
and the contents/refs reads make code legible to an agent for the first time.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from kgraphplanner.vital_agent_rest_resource_client.tools.github.common_models import (
    GitHubRepoBase, GitHubOutputBase
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class GitHubRepoGetInput(GitHubRepoBase):
    """Repository metadata, including the default branch and open issue count."""
    operation: Literal["get_repo"] = Field(..., description="Operation to perform")


class GitHubGetFileInput(GitHubRepoBase):
    """Read a file, or list a directory.

    Binary files are reported as such rather than returned as mojibake, and
    content is truncated at `max_chars` -- the `get_run_logs` treatment.
    """
    operation: Literal["get_file_contents"] = Field(..., description="Operation to perform")
    path: str = Field(..., description="Path within the repository", min_length=1)
    ref: Optional[str] = Field(None, description="Branch, tag or sha; defaults to the default branch")
    max_chars: Optional[int] = Field(
        None, description="Truncate content beyond this many characters",
        ge=100, le=200000)


class GitHubListBranchesInput(GitHubRepoBase):
    """List branches. Use to discover a valid `base` before opening a PR."""
    operation: Literal["list_branches"] = Field(..., description="Operation to perform")
    protected_only: Optional[bool] = Field(None, description="Only branches with protection rules")
    max_results: Optional[int] = Field(30, description="Maximum branches to return", ge=1, le=100)
    page: Optional[int] = Field(None, description="Page to fetch", ge=1)


class GitHubListCommitsInput(GitHubRepoBase):
    """List commits, optionally scoped to a ref, path, author or date range."""
    operation: Literal["list_commits"] = Field(..., description="Operation to perform")
    ref: Optional[str] = Field(None, description="Branch, tag or sha")
    path: Optional[str] = Field(None, description="Only commits touching this path")
    author: Optional[str] = Field(None, description="Only commits by this login or email")
    since: Optional[str] = Field(None, description="Only commits after this ISO 8601 timestamp")
    until: Optional[str] = Field(None, description="Only commits before this ISO 8601 timestamp")
    max_results: Optional[int] = Field(30, description="Maximum commits to return", ge=1, le=100)
    page: Optional[int] = Field(None, description="Page to fetch", ge=1)


class GitHubGetCommitInput(GitHubRepoBase):
    """One commit with the files it changed."""
    operation: Literal["get_commit"] = Field(..., description="Operation to perform")
    ref: str = Field(..., description="Commit sha, branch or tag", min_length=1)
    include_patch: Optional[bool] = Field(None, description="Include the diff hunk per file")
    max_files: Optional[int] = Field(
        None, description="Maximum files to return", ge=1, le=100)


class GitHubCompareRefsInput(GitHubRepoBase):
    """Compare two refs -- what changed between them."""
    operation: Literal["compare_refs"] = Field(..., description="Operation to perform")
    base: str = Field(..., description="Base ref", min_length=1)
    head: str = Field(..., description="Head ref", min_length=1)
    include_patch: Optional[bool] = Field(None, description="Include the diff hunk per file")
    max_files: Optional[int] = Field(
        None, description="Maximum files to return", ge=1, le=100)


class GitHubGetAuthenticatedUserInput(BaseModel):
    """Who the service's token authenticates as.

    Deliberately NOT built on GitHubRepoBase, mirroring the service: this is the
    one operation about the *token* rather than a repository, so requiring an
    owner/repo it never uses would make the caller invent a value and would imply
    an allowlist check that does not apply.
    """
    operation: Literal["get_authenticated_user"] = Field(..., description="Operation to perform")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class GitHubAuthenticatedUser(BaseModel):
    login: str = Field(..., description="Account the service's token authenticates as")
    type: Optional[str] = Field(None, description="User or Bot")

class GitHubRepository(BaseModel):
    full_name: str = Field(..., description="owner/name")
    name: Optional[str] = Field(None, description="Repository name")
    owner: Optional[str] = Field(None, description="Owner login")
    private: Optional[bool] = Field(None, description="True if the repository is private")
    description: Optional[str] = Field(None, description="Repository description")
    default_branch: Optional[str] = Field(None, description="Default branch name")
    html_url: Optional[str] = Field(None, description="Browser URL")
    language: Optional[str] = Field(None, description="Primary language")
    topics: List[str] = Field(default_factory=list, description="Repository topics")
    open_issues_count: Optional[int] = Field(
        None, description="Open issues, as GitHub counts them -- includes pull requests")
    archived: Optional[bool] = Field(None, description="True if archived")
    disabled: Optional[bool] = Field(None, description="True if disabled")
    has_issues: Optional[bool] = Field(None, description="True if issues are enabled")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    pushed_at: Optional[str] = Field(None, description="Last push timestamp")


class GitHubBranch(BaseModel):
    name: str = Field(..., description="Branch name")
    sha: Optional[str] = Field(None, description="Head commit sha")
    protected: Optional[bool] = Field(None, description="True if branch protection applies")
    is_default: bool = Field(False, description="True if this is the repository's default branch")


class GitHubCommit(BaseModel):
    sha: str = Field(..., description="Commit sha")
    message: Optional[str] = Field(None, description="Commit message")
    author: Optional[str] = Field(None, description="Author login or name")
    date: Optional[str] = Field(None, description="Commit timestamp")
    html_url: Optional[str] = Field(None, description="Browser URL for the commit")


class GitHubFileContent(BaseModel):
    path: str = Field(..., description="Path within the repository")
    type: Optional[str] = Field(None, description="file or dir")
    size: Optional[int] = Field(None, description="Size in bytes")
    sha: Optional[str] = Field(None, description="Blob sha")
    content: Optional[str] = Field(None, description="Decoded text content; absent for binaries")
    content_truncated: bool = Field(False, description="True if content was truncated")
    is_binary: bool = Field(False, description="True if the file is binary")
    html_url: Optional[str] = Field(None, description="Browser URL for the file")
    entries: List[str] = Field(default_factory=list, description="Names, when path is a directory")


class GitHubComparison(BaseModel):
    status: Optional[str] = Field(None, description="ahead, behind, identical or diverged")
    ahead_by: Optional[int] = Field(None, description="Commits head is ahead of base")
    behind_by: Optional[int] = Field(None, description="Commits head is behind base")
    total_commits: Optional[int] = Field(None, description="Commits in the comparison")
    files_changed: Optional[int] = Field(None, description="Files changed")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class GitHubRepoToolOutput(GitHubOutputBase):
    """Output model for github_repo_tool, shared by all of its operations."""
    tool: Literal["github_repo_tool"] = Field("github_repo_tool", description="Tool identifier")
    operation: str = Field(..., description="Operation that was performed")
    repository_info: Optional[GitHubRepository] = Field(None, description="Repository metadata")
    file: Optional[GitHubFileContent] = Field(None, description="File or directory contents")
    branches: List[GitHubBranch] = Field(default_factory=list, description="Branches")
    commits: List[GitHubCommit] = Field(default_factory=list, description="Commits")
    commit: Optional[GitHubCommit] = Field(None, description="A single commit")
    comparison: Optional[GitHubComparison] = Field(None, description="Comparison summary")
    files: List[dict] = Field(default_factory=list, description="Files changed, for commit/compare")
    authenticated_user: Optional[GitHubAuthenticatedUser] = Field(
        None, description="Identity the service's token authenticates as")


GITHUB_REPO_OPERATION_MODELS = {
    "get_repo": GitHubRepoGetInput,
    "get_file_contents": GitHubGetFileInput,
    "list_branches": GitHubListBranchesInput,
    "list_commits": GitHubListCommitsInput,
    "get_commit": GitHubGetCommitInput,
    "compare_refs": GitHubCompareRefsInput,
    "get_authenticated_user": GitHubGetAuthenticatedUserInput,
}
