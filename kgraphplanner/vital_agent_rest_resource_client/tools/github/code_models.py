"""
Request/response models for github_code_tool.

Mirrored from vital-agent-resource-rest at commit `cf9f411`
(vital_agent_resource_app/tools/github/code_models.py).

This tool did not exist when the other models were mirrored. The service split
its tools by **authority** rather than by GitHub resource (service commit
`6543689`): everything that alters code lives here, so "may this agent change
code?" is answered by whether this one tool is reachable, independently of the
per-deployment config gates.

That is why `merge_pr` is here rather than with the other pull request
operations -- merging lands commits on the base branch. The agent-facing tool
stays in `tools/github/pr_tools.py`, because an agent asks about merging as a
pull request action; only the service routing moved.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from kgraphplanner.vital_agent_rest_resource_client.tools.github.common_models import (
    GitHubRepoBase, GitHubOutputBase
)
from kgraphplanner.vital_agent_rest_resource_client.tools.github.repo_models import (
    GitHubBranch
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class GitHubCreateBranchInput(GitHubRepoBase):
    """Create a branch. Gated by `allow_writes` server-side."""
    operation: Literal["create_branch"] = Field(..., description="Operation to perform")
    branch: str = Field(..., description="Name of the branch to create", min_length=1)
    from_ref: Optional[str] = Field(
        None, description="Branch, tag or sha to branch from; defaults to the default branch")


class GitHubDeleteBranchInput(GitHubRepoBase):
    """Delete a branch.

    The service refuses the repository's default branch outright rather than
    gating it -- no configuration makes deleting the branch everything else is
    built on reachable.
    """
    operation: Literal["delete_branch"] = Field(..., description="Operation to perform")
    branch: str = Field(..., description="Name of the branch to delete", min_length=1)


class GitHubWriteFileInput(GitHubRepoBase):
    """Create or replace a file.

    Gated by `allow_content_writes` (default off) and, for the default branch,
    `allow_default_branch_writes` (default off). `branch` is required: GitHub
    commits to the default branch when it is omitted, and an omitted field must
    not be the route to writing on main.
    """
    operation: Literal["create_or_update_file"] = Field(..., description="Operation to perform")
    path: str = Field(..., description="Path of the file within the repository", min_length=1)
    content: str = Field(..., description="Full file content")
    message: str = Field(..., description="Commit message", min_length=1)
    branch: str = Field(..., description="Branch to commit to", min_length=1)
    sha: Optional[str] = Field(
        None, description="Blob sha of the file being replaced; looked up when omitted")


class GitHubDeleteFileInput(GitHubRepoBase):
    """Delete a file. Takes the same gate as the write it undoes."""
    operation: Literal["delete_file"] = Field(..., description="Operation to perform")
    path: str = Field(..., description="Path of the file within the repository", min_length=1)
    message: str = Field(..., description="Commit message", min_length=1)
    branch: str = Field(..., description="Branch to commit to", min_length=1)
    sha: Optional[str] = Field(
        None, description="Blob sha of the file being deleted; looked up when omitted")


class GitHubFileWrite(BaseModel):
    """One file in a multi-file commit."""
    path: str = Field(..., description="Repository-relative path", min_length=1)
    content: str = Field(..., description="Full file content as text (not base64)")


class GitHubWriteFilesInput(GitHubRepoBase):
    """Create, overwrite and delete several files in ONE commit.

    The difference from repeated `create_or_update_file` is not convenience: with
    per-file commits there is no way to say "the branch should now be exactly
    base + this tree", so a retried change stacks a commit per attempt instead of
    showing one reviewable diff. `from_ref` + `force` restores that.

    A force update against the default branch is refused outright, not gated --
    it would discard the history everything else is built on.
    """
    operation: Literal["write_files"] = Field(..., description="Operation to perform")
    branch: str = Field(..., description="Branch to commit on", min_length=1)
    message: str = Field(..., description="Commit message", min_length=1)
    files: List[GitHubFileWrite] = Field(
        default_factory=list, description="Files to create or overwrite")
    deletions: List[str] = Field(
        default_factory=list, description="Repository-relative paths to remove in the same commit")
    from_ref: Optional[str] = Field(
        None, description="Base the commit on this ref instead of the branch's current head")
    force: bool = Field(
        False, description="Allow a non-fast-forward branch update, discarding commits the "
                           "branch had. Required with from_ref.")


class GitHubMergeInput(GitHubRepoBase):
    """Merge a pull request. Gated by `allow_pr_merge`, off by default."""
    operation: Literal["merge_pr"] = Field(..., description="Operation to perform")
    pr_number: int = Field(..., description="Pull request number", ge=1)
    merge_method: Optional[Literal["merge", "squash", "rebase"]] = Field(
        "merge", description="How to merge")
    commit_title: Optional[str] = Field(None, description="Merge commit title")
    commit_message: Optional[str] = Field(None, description="Merge commit body")
    sha: Optional[str] = Field(None, description="Refuse the merge unless HEAD matches this sha")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class GitHubWriteResult(BaseModel):
    path: str = Field(..., description="Path written")
    branch: str = Field(..., description="Branch committed to")
    commit_sha: Optional[str] = Field(None, description="Resulting commit sha")
    blob_sha: Optional[str] = Field(None, description="Resulting blob sha")
    created: bool = Field(..., description="True if the file was created, false if replaced")
    html_url: Optional[str] = Field(None, description="Browser URL for the file")


class GitHubMergeResult(BaseModel):
    merged: bool = Field(..., description="True if the merge succeeded")
    sha: Optional[str] = Field(None, description="Resulting merge commit sha")
    message: Optional[str] = Field(None, description="Message returned by GitHub")


class GitHubCommitResult(BaseModel):
    branch: str = Field(..., description="Branch the commit landed on")
    commit_sha: Optional[str] = Field(None, description="SHA of the new commit")
    tree_sha: Optional[str] = Field(None, description="SHA of the new tree")
    parent_sha: Optional[str] = Field(None, description="Commit this was based on")
    written: List[str] = Field(default_factory=list, description="Paths created or updated")
    deleted: List[str] = Field(default_factory=list, description="Paths removed")
    branch_created: bool = Field(False, description="True if the branch did not exist")
    forced: bool = Field(False, description="True if the ref was force-updated")
    html_url: Optional[str] = Field(None, description="Browser URL for the commit")


class GitHubDeleteResult(BaseModel):
    target: str = Field(..., description="What was deleted -- a branch name or file path")
    kind: str = Field(..., description="branch or file")
    branch: Optional[str] = Field(None, description="Branch the deletion was committed to")
    commit_sha: Optional[str] = Field(None, description="Resulting commit sha, for file deletes")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class GitHubCodeToolOutput(GitHubOutputBase):
    """Output model for github_code_tool, shared by all of its operations."""
    tool: Literal["github_code_tool"] = Field("github_code_tool", description="Tool identifier")
    operation: str = Field(..., description="Operation that was performed")
    branch: Optional[GitHubBranch] = Field(None, description="Branch from a create operation")
    write_result: Optional[GitHubWriteResult] = Field(None, description="Outcome of a file write")
    merge_result: Optional[GitHubMergeResult] = Field(None, description="Outcome of a merge")
    delete_result: Optional[GitHubDeleteResult] = Field(None, description="Outcome of a delete")
    commit_result: Optional[GitHubCommitResult] = Field(
        None, description="Outcome of a multi-file commit")


GITHUB_CODE_OPERATION_MODELS = {
    "create_branch": GitHubCreateBranchInput,
    "delete_branch": GitHubDeleteBranchInput,
    "create_or_update_file": GitHubWriteFileInput,
    "delete_file": GitHubDeleteFileInput,
    "merge_pr": GitHubMergeInput,
    "write_files": GitHubWriteFilesInput,
}
