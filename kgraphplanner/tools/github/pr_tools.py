"""
Agent-facing tools for github_pr_tool operations.

merge_pr and create_pr_review(APPROVE) are additionally gated server-side by
ALLOW_PR_MERGE, which is off by default.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from kgraphplanner.tools.github.github_service_tool import GitHubServiceTool
from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
    GitHubIssueSearchInput,
)
from kgraphplanner.vital_agent_rest_resource_client.tools.github.pr_models import (
    GitHubPRListInput, GitHubPRGetInput, GitHubPRFilesInput,
    GitHubPRCommentListInput, GitHubPRReviewListInput, GitHubPRCreateInput,
    GitHubPRUpdateInput, GitHubPRCommentCreateInput, GitHubPRReviewCreateInput,
    GitHubPRRequestReviewersInput,
)
from kgraphplanner.vital_agent_rest_resource_client.tools.github.code_models import (
    GitHubMergeInput,
)

SERVICE_TOOL = "github_pr_tool"


def _pr_summary(pr) -> Dict[str, Any]:
    if pr is None:
        return {}
    return {
        "number": pr.number,
        "title": pr.title,
        "state": pr.state,
        "draft": pr.draft,
        "merged": pr.merged,
        "user": pr.user,
        "head": pr.head,
        "base": pr.base,
        "labels": pr.labels,
        "created_at": pr.created_at,
        "updated_at": pr.updated_at,
        "url": pr.html_url,
    }


class GitHubGetPRTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "get_pr"
    TOOL_NAME = "github_get_pr"
    DESCRIPTION = (
        "Get one pull request by number, including its description, branches, review state "
        "and change counts. Use github_list_pr_files to see which files it touches."
    )
    INPUT_MODEL = GitHubPRGetInput
    AGENT_FIELDS = {
        "pr_number": (int, Field(..., description="Pull request number", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        pr = output.pull_request
        if pr is None:
            return {"found": False}
        result = _pr_summary(pr)
        result.update({
            "found": True,
            "body": pr.body,
            "body_truncated": pr.body_truncated,
            "mergeable": pr.mergeable,
            "mergeable_state": pr.mergeable_state,
            "commits": pr.commits,
            "changed_files": pr.changed_files,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "comments": pr.comments,
            "review_comments": pr.review_comments,
            "requested_reviewers": pr.requested_reviewers,
            "merged_at": pr.merged_at,
            "closed_at": pr.closed_at,
        })
        return result


class GitHubListPRsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_prs"
    TOOL_NAME = "github_list_prs"
    DESCRIPTION = (
        "List pull requests in a repository, optionally filtered by state or base branch. "
        "Returns a summary per PR. If truncated is true, call again with the returned next_page. This tool reports no total -- for a count, use github_count_prs."
    )
    INPUT_MODEL = GitHubPRListInput
    AGENT_FIELDS = {
        "state": (Optional[str], Field("open", description="open, closed, or all")),
        "base": (Optional[str], Field(None, description="Only PRs targeting this base branch")),
        "max_results": (Optional[int], Field(10, description="Maximum PRs to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "pull_requests": [_pr_summary(p) for p in output.pull_requests],
            **self.paging(output),
        }


class GitHubListPRFilesTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_pr_files"
    TOOL_NAME = "github_list_pr_files"
    DESCRIPTION = (
        "List the files a pull request changes, with per-file line counts. "
        "Set include_patch only when the actual diff text is needed -- patches are large "
        "and consume a lot of context."
    )
    INPUT_MODEL = GitHubPRFilesInput
    AGENT_FIELDS = {
        "pr_number": (int, Field(..., description="Pull request number", ge=1)),
        "include_patch": (Optional[bool], Field(
            False, description="Include the diff text per file. Leave off unless the diff is needed.")),
        "max_results": (Optional[int], Field(20, description="Maximum files to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        files = []
        for f in output.files:
            entry = {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
            }
            if f.previous_filename:
                entry["previous_filename"] = f.previous_filename
            if f.patch is not None:
                entry["patch"] = f.patch
                entry["patch_truncated"] = f.patch_truncated
            files.append(entry)
        return {"files": files, **self.paging(output)}


class GitHubListPRCommentsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_pr_comments"
    TOOL_NAME = "github_list_pr_comments"
    DESCRIPTION = (
        "List the conversation comments on a pull request. These are the discussion "
        "comments, not review comments anchored to specific diff lines -- for those use "
        "github_list_pr_reviews."
    )
    INPUT_MODEL = GitHubPRCommentListInput
    AGENT_FIELDS = {
        "pr_number": (int, Field(..., description="Pull request number", ge=1)),
        "max_results": (Optional[int], Field(10, description="Maximum comments to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "comments": [{
                "id": c.id, "user": c.user, "body": c.body,
                "body_truncated": c.body_truncated, "created_at": c.created_at, "url": c.html_url,
            } for c in output.comments],
            **self.paging(output),
        }


class GitHubListPRReviewsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_pr_reviews"
    TOOL_NAME = "github_list_pr_reviews"
    DESCRIPTION = (
        "List the reviews submitted on a pull request, with each reviewer's verdict "
        "(APPROVED, CHANGES_REQUESTED or COMMENTED)."
    )
    INPUT_MODEL = GitHubPRReviewListInput
    AGENT_FIELDS = {
        "pr_number": (int, Field(..., description="Pull request number", ge=1)),
        "max_results": (Optional[int], Field(10, description="Maximum reviews to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "reviews": [{
                "id": r.id, "user": r.user, "state": r.state, "body": r.body,
                "submitted_at": r.submitted_at, "url": r.html_url,
            } for r in output.reviews],
            **self.paging(output),
        }


class GitHubCountPRsTool(GitHubServiceTool):
    """The pull-request counterpart of github_count_issues (plan section 16).

    Predicted by section 16.1 and confirmed: list_prs supplies no total, exactly
    like list_issues, so "how many PRs?" had no tool shaped to answer it. Search
    does supply one -- `is:pr` returns 23, matching a full paged enumeration.

    Note the service tool is github_issue_tool, not github_pr_tool: `search_issues`
    is an operation of the issue tool and searches both issues and PRs. The
    agent-facing tool is grouped by what a user asks about (pull requests); the
    service operation it rides on is an implementation detail (plan section 2).

    Workflow runs need no equivalent -- list_workflow_runs already reports
    GitHub's total_count.
    """

    SERVICE_TOOL = "github_issue_tool"
    OPERATION = "search_issues"
    TOOL_NAME = "github_count_prs"
    DESCRIPTION = (
        "Count the pull requests in a repository, optionally filtered. Use this for any "
        "'how many pull requests' question -- it returns an exact total in one call. Do "
        "NOT count by listing PRs and counting the results, and never infer a count from "
        "a PR number: numbers are identifiers shared with issues and are never reused, so "
        "the highest number is not a total."
    )
    INPUT_MODEL = GitHubIssueSearchInput
    AGENT_FIELDS = {
        "state": (Optional[str], Field(
            "open", description="open, closed, merged, unmerged, or all")),
        "filter": (Optional[str], Field(
            None,
            description="Optional extra GitHub search terms to narrow the count, "
                        "e.g. 'base:main' or 'author:octocat'. Omit to count everything.")),
    }

    def build_wire_input(self, repo: str, **kwargs):
        state = (kwargs.pop("state", None) or "open").lower()
        extra = kwargs.pop("filter", None)

        terms = ["is:pr"]
        if state in ("open", "closed", "merged", "unmerged"):
            terms.append(f"is:{state}")
        if extra:
            terms.append(extra.strip())

        owner, name = self.split_repo(repo)
        return self.INPUT_MODEL(
            operation=self.OPERATION, owner=owner, repo=name,
            query=" ".join(terms),
            # Counting PRs, so PRs must not be filtered out of the result set.
            include_pull_requests=True,
            max_results=1,
        )

    def project(self, output) -> Dict[str, Any]:
        return {"count": output.total_count, "counted": "pull requests only"}


PR_TOOLS = {
    cls.TOOL_NAME: cls for cls in (
        GitHubGetPRTool,
        GitHubListPRsTool,
        GitHubListPRFilesTool,
        GitHubListPRCommentsTool,
        GitHubListPRReviewsTool,
        GitHubCountPRsTool,
    )
}


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

class GitHubCreatePRTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "create_pr"
    TOOL_NAME = "github_create_pr"
    MUTATING = True
    DESCRIPTION = (
        "Open a pull request from an existing branch. The head branch must already exist "
        "with commits on it -- these tools cannot create branches or push code."
    )
    INPUT_MODEL = GitHubPRCreateInput
    AGENT_FIELDS = {
        "title": (str, Field(..., description="Pull request title", min_length=1)),
        "head": (str, Field(..., description="Existing branch containing the changes", min_length=1)),
        "base": (str, Field(..., description="Branch to merge into, usually main", min_length=1)),
        "body": (Optional[str], Field(None, description="Description in Markdown")),
        "draft": (Optional[bool], Field(False, description="Open as a draft")),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"created": output.pull_request is not None, **_pr_summary(output.pull_request)}


class GitHubUpdatePRTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "update_pr"
    TOOL_NAME = "github_update_pr"
    MUTATING = True
    DESCRIPTION = (
        "Update a pull request's title, description or state. Only the fields you provide "
        "are changed. Set state='closed' to close a PR without merging it -- GitHub has no "
        "way to delete one."
    )
    INPUT_MODEL = GitHubPRUpdateInput
    AGENT_FIELDS = {
        "pr_number": (int, Field(..., description="Pull request number", ge=1)),
        "title": (Optional[str], Field(None, description="New title", min_length=1)),
        "body": (Optional[str], Field(None, description="New description in Markdown")),
        "state": (Optional[str], Field(None, description="open or closed")),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"updated": output.pull_request is not None, **_pr_summary(output.pull_request)}


class GitHubAddPRCommentTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "add_pr_comment"
    TOOL_NAME = "github_add_pr_comment"
    MUTATING = True
    DESCRIPTION = (
        "Add a conversation comment to a pull request. This is a general comment, not a "
        "review -- to approve or request changes use github_create_pr_review."
    )
    INPUT_MODEL = GitHubPRCommentCreateInput
    AGENT_FIELDS = {
        "pr_number": (int, Field(..., description="Pull request number", ge=1)),
        "body": (str, Field(..., description="Comment body in Markdown", min_length=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        c = output.comment
        return {"added": c is not None, "id": c.id if c else None,
                "url": c.html_url if c else None}


class GitHubCreatePRReviewTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "create_pr_review"
    TOOL_NAME = "github_create_pr_review"
    MUTATING = True
    DESCRIPTION = (
        "Submit a review on a pull request: APPROVE, REQUEST_CHANGES or COMMENT. "
        "APPROVE is treated as a step toward merging and may be disabled by policy, in "
        "which case the response explains that rather than failing."
    )
    INPUT_MODEL = GitHubPRReviewCreateInput
    AGENT_FIELDS = {
        "pr_number": (int, Field(..., description="Pull request number", ge=1)),
        "event": (str, Field(..., description="APPROVE, REQUEST_CHANGES, or COMMENT")),
        "body": (Optional[str], Field(None, description="Review body in Markdown")),
    }

    def project(self, output) -> Dict[str, Any]:
        r = output.review
        return {"submitted": r is not None, "id": r.id if r else None,
                "state": r.state if r else None, "url": r.html_url if r else None}


class GitHubMergePRTool(GitHubServiceTool):
    # Merging lands commits on the base branch, so the service groups it with the
    # other code-altering operations rather than with PR metadata. The tool stays
    # here because an agent asks about it as a pull-request action; only the
    # routing moved (service commit 6543689).
    SERVICE_TOOL = "github_code_tool"
    OPERATION = "merge_pr"
    TOOL_NAME = "github_merge_pr"
    MUTATING = True
    DESCRIPTION = (
        "Merge a pull request. This is the highest-impact operation available and is "
        "disabled by policy in most deployments -- if so, the response says so rather than "
        "failing. Prefer proposing a merge to a human over attempting one."
    )
    INPUT_MODEL = GitHubMergeInput
    AGENT_FIELDS = {
        "pr_number": (int, Field(..., description="Pull request number", ge=1)),
        "merge_method": (Optional[str], Field("merge", description="merge, squash, or rebase")),
        "commit_title": (Optional[str], Field(None, description="Merge commit title")),
    }

    def project(self, output) -> Dict[str, Any]:
        # output is now a GitHubCodeToolOutput; merge_result is the same shape.
        m = output.merge_result
        return {"merged": bool(m and m.merged), "sha": m.sha if m else None,
                "message": m.message if m else None}


class GitHubRequestReviewersTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "request_reviewers"
    TOOL_NAME = "github_request_reviewers"
    MUTATING = True
    DESCRIPTION = (
        "Request review on a pull request from users or teams. Pass logins in reviewers, "
        "team slugs in team_reviewers, or both. A user cannot review their own pull request, "
        "and a login that cannot review this repository is reported as an error -- use "
        "github_list_assignable_users to see who can."
    )
    INPUT_MODEL = GitHubPRRequestReviewersInput
    AGENT_FIELDS = {
        "pr_number": (int, Field(..., description="Pull request number", ge=1)),
        "reviewers": (Optional[List[str]], Field(
            None, description="User logins to request review from")),
        "team_reviewers": (Optional[List[str]], Field(
            None, description="Team slugs to request review from")),
    }

    def project(self, output) -> Dict[str, Any]:
        pr = output.pull_request
        return {
            "requested": True,
            "pr_number": pr.number if pr else None,
            "requested_reviewers": pr.requested_reviewers if pr else [],
            "url": pr.html_url if pr else None,
        }


PR_TOOLS.update({
    cls.TOOL_NAME: cls for cls in (
        GitHubRequestReviewersTool,
        GitHubCreatePRTool,
        GitHubUpdatePRTool,
        GitHubAddPRCommentTool,
        GitHubCreatePRReviewTool,
        GitHubMergePRTool,
    )
})
