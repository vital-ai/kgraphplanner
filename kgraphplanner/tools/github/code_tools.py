"""
Agent-facing tools for github_code_tool operations.

**This is the module where an agent gains the ability to change code.** Every
other GitHub tool in this package reads, comments, labels, or moves an issue
between states -- all recoverable, all confined to the discussion around the
code. These four commit to a branch.

That is why the service put them behind a *separate tool name* rather than a
config flag (service commit 6543689): "may this agent change code?" is answered
by whether `github_code_tool` is reachable at all, and no combination of gate
settings on the other tools can grant it.

Registering these is therefore a deliberate decision per agent, not a default,
and it is made where every other per-agent decision is made -- in that agent's
own config, via `tools.enabled`. There is no global answer this module could
set: an issue-triage agent should never see these, a change-proposing agent
needs them, and both are correct.

The three derived sets in this package's __init__ are the vocabulary for saying
so. `SAFE_TOOLS` is the one to reach for here: it is everything except the code
writes, so "may comment on issues, may not touch code" is one name rather than a
hand-listed set that drifts. `READ_ONLY_TOOLS` will not do -- MUTATING is equally
true of adding a label, so it excludes far more than intended.

Server-side gates, all off by default and all reported as actionable api_error:

  - `allow_writes`                 -- create_branch, delete_branch
  - `allow_content_writes`         -- create_or_update_file, delete_file
  - `allow_default_branch_writes`  -- additionally required to write to the default branch
  - `allow_pr_merge`               -- merge_pr, whose agent-facing tool stays in pr_tools.py

merge_pr is routed to this service tool but presented as a pull request action,
because that is how an agent asks for it. Only the routing moved.

Deleting the default branch is refused outright rather than gated -- there is no
configuration under which it is reachable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from kgraphplanner.tools.github.github_service_tool import GitHubServiceTool

from kgraphplanner.vital_agent_rest_resource_client.tools.github.code_models import (
    GitHubCreateBranchInput, GitHubDeleteBranchInput,
    GitHubWriteFileInput, GitHubDeleteFileInput, GitHubWriteFilesInput,
)

SERVICE_TOOL = "github_code_tool"


class GitHubCreateBranchTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "create_branch"
    TOOL_NAME = "github_create_branch"
    MUTATING = True
    DESCRIPTION = (
        "Create a branch. Branches from the repository's default branch unless you pass "
        "from_ref. This is the first step in proposing a change: create a branch, write "
        "files to it with github_create_or_update_file, then open a pull request with "
        "github_create_pr. Use github_list_branches first to check the name is not taken."
    )
    INPUT_MODEL = GitHubCreateBranchInput
    AGENT_FIELDS = {
        "branch": (str, Field(..., description="Name of the branch to create", min_length=1)),
        "from_ref": (Optional[str], Field(
            None, description="Branch, tag or sha to branch from; defaults to the default branch")),
    }

    def project(self, output) -> Dict[str, Any]:
        b = output.branch
        return {
            "created": b is not None,
            "branch": b.name if b else None,
            "sha": b.sha if b else None,
        }


class GitHubDeleteBranchTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "delete_branch"
    TOOL_NAME = "github_delete_branch"
    MUTATING = True
    DESCRIPTION = (
        "Delete a branch. Unmerged commits on that branch are lost, so confirm the work is "
        "merged -- github_compare_refs against the default branch will show whether it is "
        "ahead. The repository's default branch cannot be deleted. Deleting a branch with an "
        "open pull request closes that pull request."
    )
    INPUT_MODEL = GitHubDeleteBranchInput
    AGENT_FIELDS = {
        "branch": (str, Field(..., description="Name of the branch to delete", min_length=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        d = output.delete_result
        return {"deleted": d is not None, "branch": d.target if d else None}


class GitHubCreateOrUpdateFileTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "create_or_update_file"
    TOOL_NAME = "github_create_or_update_file"
    MUTATING = True
    DESCRIPTION = (
        "Create a file, or replace an existing one, as a single commit. `content` is the "
        "COMPLETE new file, not a patch or a fragment -- whatever you pass becomes the whole "
        "file, so read the current contents with github_get_file_contents first when editing "
        "rather than creating. `branch` is required and writing to the default branch is "
        "normally refused; create a branch and open a pull request instead."
    )
    INPUT_MODEL = GitHubWriteFileInput
    AGENT_FIELDS = {
        "path": (str, Field(..., description="Path of the file within the repository", min_length=1)),
        "content": (str, Field(..., description="The COMPLETE new content of the file, not a diff")),
        "message": (str, Field(..., description="Commit message", min_length=1)),
        # Required rather than optional with a default: GitHub commits to the
        # default branch when branch is omitted, so an omitted field must not be
        # the route to writing on main.
        "branch": (str, Field(..., description="Branch to commit to. Required -- create one "
                                               "with github_create_branch.", min_length=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        w = output.write_result
        if w is None:
            return {"written": False}
        return {
            "written": True,
            "path": w.path,
            "branch": w.branch,
            "created": w.created,
            "commit_sha": w.commit_sha,
            "url": w.html_url,
        }


class GitHubDeleteFileTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "delete_file"
    TOOL_NAME = "github_delete_file"
    MUTATING = True
    DESCRIPTION = (
        "Delete a file as a single commit. Takes the same permissions as writing one, and "
        "`branch` is likewise required. The file remains in history and can be restored from "
        "an earlier commit."
    )
    INPUT_MODEL = GitHubDeleteFileInput
    AGENT_FIELDS = {
        "path": (str, Field(..., description="Path of the file to delete", min_length=1)),
        "message": (str, Field(..., description="Commit message", min_length=1)),
        "branch": (str, Field(..., description="Branch to commit to. Required.", min_length=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        d = output.delete_result
        if d is None:
            return {"deleted": False}
        return {
            "deleted": True,
            "path": d.target,
            "branch": d.branch,
            "commit_sha": d.commit_sha,
        }


class GitHubWriteFilesTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "write_files"
    TOOL_NAME = "github_write_files"
    MUTATING = True
    DESCRIPTION = (
        "Create, overwrite and delete several files in ONE commit. Prefer this over repeated "
        "github_create_or_update_file whenever a change spans more than one file: it produces "
        "a single reviewable commit instead of one per file. "
        "Each file's content is the COMPLETE new file, not a patch, so read anything you are "
        "editing first with github_get_file_contents. "
        "Creates the branch if it does not exist. To REPLACE a branch's contents rather than "
        "add to them -- for instance when redoing a change after a mistake -- set from_ref to "
        "the base and force=true, which makes the result depend only on what you pass rather "
        "than stacking another commit."
    )
    INPUT_MODEL = GitHubWriteFilesInput
    AGENT_FIELDS = {
        "branch": (str, Field(..., description="Branch to commit on", min_length=1)),
        "message": (str, Field(..., description="Commit message", min_length=1)),
        "files": (Optional[List[dict]], Field(
            None, description="Files to write, each {\"path\": ..., \"content\": ...} where "
                              "content is the complete file")),
        "deletions": (Optional[List[str]], Field(
            None, description="Paths to remove in the same commit")),
        "from_ref": (Optional[str], Field(
            None, description="Base the commit on this ref instead of the branch's head; "
                              "requires force=true")),
        "force": (Optional[bool], Field(
            False, description="Allow a non-fast-forward update, discarding commits the branch "
                               "had. Refused outright on the default branch.")),
    }

    def project(self, output) -> Dict[str, Any]:
        c = output.commit_result
        if c is None:
            return {"committed": False}
        return {
            "committed": True,
            "branch": c.branch,
            "branch_created": c.branch_created,
            "forced": c.forced,
            "written": c.written,
            "deleted": c.deleted,
            "commit_sha": c.commit_sha,
            "url": c.html_url,
        }


CODE_TOOLS = {
    cls.TOOL_NAME: cls for cls in (
        GitHubWriteFilesTool,
        GitHubCreateBranchTool,
        GitHubDeleteBranchTool,
        GitHubCreateOrUpdateFileTool,
        GitHubDeleteFileTool,
    )
}
