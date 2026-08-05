"""
Agent-facing tools for github_repo_tool operations.

This is the set that makes a repository *legible* to an agent. Until it existed
an agent could read everything written about the code -- issues, PRs, review
comments, CI logs -- and nothing of the code itself, which is why several plan
section 17.1 gaps all reduced to the same missing capability.

Read-only by construction: `github_repo_tool` is a separate service tool from
`github_code_tool` (service commit 6543689), so registering these grants reading
and nothing else, independently of how the server-side write gates are set.

Two of these exist to stop an agent guessing:

  - github_get_repo supplies `default_branch`, so opening a PR no longer needs
    the agent to assume "main". Several repositories in the wild are not main.
  - github_list_branches supplies valid refs, so a branch name is chosen from
    what exists rather than invented.

Both are the plan section 17.2 pattern -- an operation that fails because the
agent had to guess a value it was never given a way to look up.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from kgraphplanner.tools.github.github_service_tool import GitHubServiceTool
from kgraphplanner.vital_agent_rest_resource_client.tools.github.repo_models import (
    GitHubRepoGetInput, GitHubGetFileInput, GitHubListBranchesInput,
    GitHubListCommitsInput, GitHubGetCommitInput, GitHubCompareRefsInput,
    GitHubGetAuthenticatedUserInput,
)

SERVICE_TOOL = "github_repo_tool"


def _commit_summary(c) -> Dict[str, Any]:
    if c is None:
        return {}
    return {
        "sha": c.sha,
        "message": c.message,
        "author": c.author,
        "date": c.date,
        "url": c.html_url,
    }


def _file_summary(f) -> Dict[str, Any]:
    """Files changed come back as raw dicts from the service, one shape for
    commits and comparisons alike. Only the fields an agent reasons about are
    kept -- a full patch per file is the largest thing this tool can return."""
    return {
        "filename": f.get("filename"),
        "status": f.get("status"),
        "additions": f.get("additions"),
        "deletions": f.get("deletions"),
        "patch": f.get("patch"),
    }


class GitHubGetRepoTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "get_repo"
    TOOL_NAME = "github_get_repo"
    DESCRIPTION = (
        "Get repository metadata: the default branch, primary language, topics, and whether "
        "it is private or archived. Call this before creating a pull request or a branch -- "
        "it gives you the real default branch name rather than assuming 'main'. "
        "Note open_issues_count is GitHub's own count and includes pull requests, so do not "
        "report it as an issue count; use github_count_issues for that."
    )
    INPUT_MODEL = GitHubRepoGetInput
    AGENT_FIELDS: Dict[str, tuple] = {}

    def project(self, output) -> Dict[str, Any]:
        r = output.repository_info
        if r is None:
            return {"found": False}
        return {
            "found": True,
            "full_name": r.full_name,
            "description": r.description,
            "default_branch": r.default_branch,
            "private": r.private,
            "archived": r.archived,
            "has_issues": r.has_issues,
            "language": r.language,
            "topics": r.topics,
            # Named to say what it is. Called open_issues_count it reads as an
            # issue count, and it is not one -- see the description.
            "open_issues_and_prs_count": r.open_issues_count,
            "url": r.html_url,
            "created_at": r.created_at,
            "pushed_at": r.pushed_at,
        }


class GitHubGetFileContentsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "get_file_contents"
    TOOL_NAME = "github_get_file_contents"
    DESCRIPTION = (
        "Read a file from the repository, or list a directory by passing a directory path. "
        "Content is truncated at max_chars and the response says so; binary files are "
        "reported as binary rather than returned as garbled text. Defaults to the "
        "repository's default branch unless you pass a ref."
    )
    INPUT_MODEL = GitHubGetFileInput
    AGENT_FIELDS = {
        "path": (str, Field(..., description="Path within the repository, e.g. src/main.py. "
                                             "A directory path lists its entries.", min_length=1)),
        "ref": (Optional[str], Field(None, description="Branch, tag or commit sha; defaults to "
                                                       "the default branch")),
        # The service permits up to 200000. Defaulting there would let one call
        # dominate the agent's context, so this starts low and is raised
        # deliberately -- the same reasoning as get_run_logs.
        "max_chars": (Optional[int], Field(
            20000, description="Truncate file content beyond this many characters",
            ge=100, le=200000)),
    }

    def project(self, output) -> Dict[str, Any]:
        f = output.file
        if f is None:
            return {"found": False}
        if f.type == "dir":
            return {"found": True, "path": f.path, "type": "dir", "entries": f.entries}
        return {
            "found": True,
            "path": f.path,
            "type": f.type,
            "size": f.size,
            "sha": f.sha,
            "is_binary": f.is_binary,
            "content": f.content,
            "content_truncated": f.content_truncated,
            "url": f.html_url,
        }


class GitHubListBranchesTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_branches"
    TOOL_NAME = "github_list_branches"
    DESCRIPTION = (
        "List the branches in a repository, marking which one is the default and which are "
        "protected. Use this to pick a valid base before opening a pull request, rather than "
        "guessing a branch name."
    )
    INPUT_MODEL = GitHubListBranchesInput
    AGENT_FIELDS = {
        "protected_only": (Optional[bool], Field(
            None, description="Only return branches with protection rules")),
        "max_results": (Optional[int], Field(30, description="Maximum branches to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "branches": [{
                "name": b.name,
                "sha": b.sha,
                "protected": b.protected,
                "is_default": b.is_default,
            } for b in output.branches],
            **self.paging(output),
        }


class GitHubListCommitsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_commits"
    TOOL_NAME = "github_list_commits"
    DESCRIPTION = (
        "List commits, most recent first. Narrow with ref for a branch, path for commits "
        "touching one file or directory, author for one person, or since/until for a date "
        "range. This is how to answer 'what changed recently' or 'who last touched this "
        "file'. For the diff of one commit, follow with github_get_commit."
    )
    INPUT_MODEL = GitHubListCommitsInput
    AGENT_FIELDS = {
        "ref": (Optional[str], Field(None, description="Branch, tag or sha; defaults to the default branch")),
        "path": (Optional[str], Field(None, description="Only commits touching this file or directory")),
        "author": (Optional[str], Field(None, description="Only commits by this login or email")),
        "since": (Optional[str], Field(None, description="Only commits after this ISO 8601 timestamp")),
        "until": (Optional[str], Field(None, description="Only commits before this ISO 8601 timestamp")),
        "max_results": (Optional[int], Field(20, description="Maximum commits to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "commits": [_commit_summary(c) for c in output.commits],
            **self.paging(output),
        }


class GitHubGetCommitTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "get_commit"
    TOOL_NAME = "github_get_commit"
    DESCRIPTION = (
        "Get one commit and the files it changed. Set include_patch=true for the actual diff "
        "hunks -- leave it off when you only need to know which files were touched, since "
        "patches are large."
    )
    INPUT_MODEL = GitHubGetCommitInput
    AGENT_FIELDS = {
        "ref": (str, Field(..., description="Commit sha, branch or tag", min_length=1)),
        "include_patch": (Optional[bool], Field(
            False, description="Include the diff hunk for each file")),
        "max_files": (Optional[int], Field(20, description="Maximum files to return", ge=1, le=100)),
    }

    def project(self, output) -> Dict[str, Any]:
        if output.commit is None:
            return {"found": False}
        return {
            "found": True,
            **_commit_summary(output.commit),
            "files": [_file_summary(f) for f in output.files],
            **self.paging(output),
        }


class GitHubCompareRefsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "compare_refs"
    TOOL_NAME = "github_compare_refs"
    DESCRIPTION = (
        "Compare two branches, tags or commits and report what changed between them: how far "
        "ahead or behind head is, and which files differ. Use this to review a branch before "
        "opening a pull request, or to see what a release contains."
    )
    INPUT_MODEL = GitHubCompareRefsInput
    AGENT_FIELDS = {
        "base": (str, Field(..., description="Base ref to compare from", min_length=1)),
        "head": (str, Field(..., description="Head ref to compare to", min_length=1)),
        "include_patch": (Optional[bool], Field(
            False, description="Include the diff hunk for each file")),
        "max_files": (Optional[int], Field(20, description="Maximum files to return", ge=1, le=100)),
    }

    def project(self, output) -> Dict[str, Any]:
        c = output.comparison
        return {
            "status": c.status if c else None,
            "ahead_by": c.ahead_by if c else None,
            "behind_by": c.behind_by if c else None,
            "total_commits": c.total_commits if c else None,
            "files_changed": c.files_changed if c else None,
            "files": [_file_summary(f) for f in output.files],
            **self.paging(output),
        }


class GitHubGetAuthenticatedUserTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "get_authenticated_user"
    TOOL_NAME = "github_get_authenticated_user"
    # The one operation about the token rather than a repository, so it takes no
    # `repo` argument at all -- see REQUIRES_REPO on the base class.
    REQUIRES_REPO = False
    DESCRIPTION = (
        "Get the account this service acts as on GitHub. Use it to recognise your own "
        "comments, issues and reviews -- anything authored by this login was written by you, "
        "not by a person you are working with."
    )
    INPUT_MODEL = GitHubGetAuthenticatedUserInput
    AGENT_FIELDS: Dict[str, tuple] = {}

    def build_wire_input(self, repo, **kwargs):
        # No owner/repo: the wire model does not carry them, and the base class
        # default would pass values this operation has no use for.
        return self.INPUT_MODEL(operation=self.OPERATION, **kwargs)

    def project(self, output) -> Dict[str, Any]:
        u = output.authenticated_user
        if u is None:
            return {"found": False}
        return {"found": True, "login": u.login, "type": u.type}


REPO_TOOLS = {
    cls.TOOL_NAME: cls for cls in (
        GitHubGetAuthenticatedUserTool,
        GitHubGetRepoTool,
        GitHubGetFileContentsTool,
        GitHubListBranchesTool,
        GitHubListCommitsTool,
        GitHubGetCommitTool,
        GitHubCompareRefsTool,
    )
}
