"""
Agent-facing tools for github_issue_tool operations.

One tool per operation (plan section 2, option B): each schema is a handful of
self-describing fields rather than an `operation` discriminator plus the union of
every field any operation might need. Which of these an agent actually sees is
decided by `tools.enabled` in its config.

Read operations first, then writes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from kgraphplanner.tools.github.github_service_tool import GitHubServiceTool
from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
    GitHubIssueGetInput, GitHubIssueListInput, GitHubIssueCommentListInput,
    GitHubIssueSearchInput, GitHubIssueCreateInput, GitHubIssueUpdateInput,
    GitHubIssueCloseInput, GitHubIssueReopenInput, GitHubIssueCommentCreateInput,
    GitHubIssueCommentUpdateInput, GitHubIssueCommentDeleteInput,
    GitHubIssueAddLabelsInput, GitHubIssueRemoveLabelsInput,
    GitHubIssueAddAssigneesInput, GitHubIssueRemoveAssigneesInput,
    GitHubIssueListLabelsInput, GitHubIssueListMilestonesInput,
    GitHubIssueListAssignableUsersInput, GitHubIssueFindByBodyInput,
)

SERVICE_TOOL = "github_issue_tool"


def _issue_summary(issue) -> Dict[str, Any]:
    """The fields an agent reasons over. Bodies are already truncated service-side."""
    if issue is None:
        return {}
    return {
        "number": issue.number,
        "title": issue.title,
        "state": issue.state,
        "state_reason": issue.state_reason,
        "labels": issue.labels,
        "assignees": issue.assignees,
        "comments": issue.comments,
        "created_at": issue.created_at,
        "updated_at": issue.updated_at,
        "closed_at": issue.closed_at,
        "url": issue.html_url,
    }


def _comment_summary(comment) -> Dict[str, Any]:
    return {
        "id": comment.id,
        "user": comment.user,
        "body": comment.body,
        "body_truncated": comment.body_truncated,
        "created_at": comment.created_at,
        "url": comment.html_url,
    }


class GitHubGetIssueTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "get_issue"
    TOOL_NAME = "github_get_issue"
    DESCRIPTION = "Get one GitHub issue by number, including its body, labels and assignees."
    INPUT_MODEL = GitHubIssueGetInput
    AGENT_FIELDS = {
        "issue_number": (int, Field(..., description="Issue number", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        issue = output.issue
        if issue is None:
            return {"found": False}
        result = _issue_summary(issue)
        result["body"] = issue.body
        result["body_truncated"] = issue.body_truncated
        result["found"] = True
        return result


class GitHubListIssuesTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_issues"
    TOOL_NAME = "github_list_issues"
    DESCRIPTION = (
        "List GitHub issues in a repository, optionally filtered by state, label or assignee. "
        "Returns a summary per issue, not full bodies -- use github_get_issue for one issue's "
        "body. If truncated is true, more issues exist: call again with the returned next_page. "
        "next_page may repeat a partly-read page, so deduplicate by issue number. This tool reports no total -- for a count, use github_count_issues instead of listing and counting."
    )
    INPUT_MODEL = GitHubIssueListInput
    AGENT_FIELDS = {
        "state": (Optional[str], Field("open", description="open, closed, or all")),
        "labels": (Optional[List[str]], Field(None, description="Only issues carrying all of these labels")),
        "assignee": (Optional[str], Field(None, description="Assignee login, or 'none' for unassigned")),
        # Exposed for polling ("what changed since I last looked"). The
        # updated_at semantics are stated because a model will otherwise read
        # this as created_at and use it to bound a search for something new.
        "since": (Optional[str], Field(
            None, description="Only issues UPDATED at or after this ISO 8601 timestamp. This "
                              "is last-modified, not creation time: an old issue edited today "
                              "is included, and a new issue is excluded if its last update "
                              "predates the bound. Good for polling; do not use it to check "
                              "whether something exists -- use github_find_issues_by_body.")),
        # Added after the section 15.9 pool comparison: `recent_issues` asks for
        # the most recently UPDATED issues, and the tool defaulted to sort=created
        # with no way to change it, so the question was unanswerable and the agent
        # was graded down for the tool's gap. This is section 15.3's "can be added
        # if a case needs them" clause being exercised by a case that needed them.
        "sort": (Optional[str], Field(
            "created", description="Sort by created, updated, or comments")),
        "direction": (Optional[str], Field("desc", description="asc or desc")),
        "max_results": (Optional[int], Field(10, description="Maximum issues to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "issues": [_issue_summary(i) for i in output.issues],
            **self.paging(output),
        }


class GitHubListIssueCommentsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_comments"
    TOOL_NAME = "github_list_issue_comments"
    DESCRIPTION = (
        "List the comments on a GitHub issue, oldest first. "
        "If truncated is true, call again with the returned next_page."
    )
    INPUT_MODEL = GitHubIssueCommentListInput
    AGENT_FIELDS = {
        "issue_number": (int, Field(..., description="Issue number", ge=1)),
        "max_results": (Optional[int], Field(10, description="Maximum comments to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "comments": [_comment_summary(c) for c in output.comments],
            **self.paging(output),
        }


class GitHubSearchIssuesTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "search_issues"
    TOOL_NAME = "github_search_issues"
    DESCRIPTION = (
        "Search issues in a repository using GitHub search syntax, e.g. 'is:open label:bug timeout'. "
        "The query is repo-relative: do not include repo:, org: or user: qualifiers, they are "
        "rejected. total_count is GitHub's count of all matches and may exceed the number returned."
    )
    INPUT_MODEL = GitHubIssueSearchInput
    AGENT_FIELDS = {
        "query": (str, Field(..., description="GitHub search syntax, without repo:/org:/user:", min_length=1)),
        "max_results": (Optional[int], Field(10, description="Maximum results to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "issues": [_issue_summary(i) for i in output.issues],
            "total_count": output.total_count,
            **self.paging(output),
        }


class GitHubCountIssuesTool(GitHubServiceTool):
    """Answers "how many issues are there?" in one call.

    Added because the eval found two independent models answering that question
    with the highest issue *number* they had seen. That is not a model quirk:
    GitHub's list endpoint supplies no total, so github_list_issues sets
    total_count to null, and an agent asked for a count has nothing to count
    with -- it either pages the whole repository or reaches for a proxy.

    GitHub's *search* endpoint does return a corpus total, so the count was
    always one call away; the tools simply never offered it. Agent-facing tool
    granularity is independent of service operation granularity (plan section 2),
    so this is a distinct tool over the same search_issues operation.
    """

    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "search_issues"
    TOOL_NAME = "github_count_issues"
    DESCRIPTION = (
        "Count the issues in a repository, optionally filtered. Use this for any "
        "'how many' question -- it returns an exact total in one call. Do NOT try to "
        "count by listing issues and counting the results, and never infer a count from "
        "an issue number: issue numbers are identifiers shared with pull requests and "
        "are never reused, so the highest number is not a total."
    )
    INPUT_MODEL = GitHubIssueSearchInput
    AGENT_FIELDS = {
        "state": (Optional[str], Field("open", description="open, closed, or all")),
        "filter": (Optional[str], Field(
            None,
            description="Optional extra GitHub search terms to narrow the count, "
                        "e.g. 'label:bug' or 'author:octocat'. Omit to count everything.")),
    }

    def build_wire_input(self, repo: str, **kwargs):
        state = (kwargs.pop("state", None) or "open").lower()
        extra = kwargs.pop("filter", None)

        # `is:issue` excludes pull requests, matching what the other issue tools
        # return by default.
        terms = ["is:issue"]
        if state in ("open", "closed"):
            terms.append(f"is:{state}")
        if extra:
            terms.append(extra.strip())

        owner, name = self.split_repo(repo)
        return self.INPUT_MODEL(
            operation=self.OPERATION, owner=owner, repo=name,
            query=" ".join(terms),
            # Only the total is wanted; asking for records would pull payload the
            # caller never reads.
            max_results=1,
        )

    def project(self, output) -> Dict[str, Any]:
        return {
            "count": output.total_count,
            "counted": "issues only, excluding pull requests",
        }


ISSUE_TOOLS = {
    cls.TOOL_NAME: cls for cls in (
        GitHubGetIssueTool,
        GitHubListIssuesTool,
        GitHubListIssueCommentsTool,
        GitHubSearchIssuesTool,
        GitHubCountIssuesTool,
    )
}


# ---------------------------------------------------------------------------
# Writes
#
# The real gates are server-side: allow_writes covers all of these, and a denial
# arrives as an actionable api_error rather than a failure. The agent-side lever
# is which of these get registered at all.
# ---------------------------------------------------------------------------

class GitHubCreateIssueTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "create_issue"
    TOOL_NAME = "github_create_issue"
    MUTATING = True
    DESCRIPTION = (
        "Open a new GitHub issue. Check first to avoid filing a duplicate -- use "
        "github_find_issues_by_body when you have an exact marker to look for, since search "
        "is indexed and lags new issues by about a minute. "
        "If the response says created=false, the issue ALREADY EXISTED and was returned "
        "instead of filed again: report it as already filed, not as newly created."
    )
    INPUT_MODEL = GitHubIssueCreateInput
    # `idempotency_key` is on the wire model but deliberately not here: it must
    # derive deterministically from a source event, which a model cannot do.
    # Programmatic callers supply it through build_wire_input. Consequence worth
    # knowing: agent-initiated creates carry no idempotency guarantee.
    AGENT_FIELDS = {
        "title": (str, Field(..., description="Issue title", min_length=1)),
        "body": (Optional[str], Field(None, description="Issue body in Markdown")),
        "labels": (Optional[List[str]], Field(None, description="Labels to apply")),
        "assignees": (Optional[List[str]], Field(None, description="Logins to assign")),
    }

    def project(self, output) -> Dict[str, Any]:
        # The service now reports `created` itself, and it means something this
        # layer cannot derive: false where an EXISTING issue was returned. The
        # old local `output.issue is not None` is true in exactly that case too,
        # so trusting it would report a deduplicated create as a fresh one --
        # the opposite of the truth. Prefer the service's value; fall back only
        # when it is absent (idempotency disabled, or no key supplied).
        created = output.created if output.created is not None else (output.issue is not None)
        result: Dict[str, Any] = {"created": created}
        if output.idempotency_guard is not None:
            result["idempotency_guard"] = output.idempotency_guard

        if output.issue is None:
            # created=false with no issue is the in-flight case: another request
            # holds the reservation and has not recorded a number yet. Saying
            # "already filed as #None" would be worse than saying nothing, so
            # this is called out as its own outcome.
            result["outcome"] = (
                "A create for this key is already in flight and has not completed. "
                "No issue number is available yet; retry shortly rather than filing again."
            )
            return result

        result.update(_issue_summary(output.issue))
        return result


class GitHubUpdateIssueTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "update_issue"
    TOOL_NAME = "github_update_issue"
    MUTATING = True
    DESCRIPTION = (
        "Update an existing issue's title, body, labels or assignees. Only the fields you "
        "provide are changed. Note that labels and assignees REPLACE the current set -- to "
        "add without removing, use github_add_labels or github_add_assignees. "
        "To close or reopen, use github_close_issue or github_reopen_issue."
    )
    INPUT_MODEL = GitHubIssueUpdateInput
    AGENT_FIELDS = {
        "issue_number": (int, Field(..., description="Issue number", ge=1)),
        "title": (Optional[str], Field(None, description="New title", min_length=1)),
        "body": (Optional[str], Field(None, description="New body in Markdown")),
        "labels": (Optional[List[str]], Field(None, description="Replace all labels with this set")),
        "assignees": (Optional[List[str]], Field(None, description="Replace all assignees with this set")),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"updated": output.issue is not None, **_issue_summary(output.issue)}


class GitHubCloseIssueTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "close_issue"
    TOOL_NAME = "github_close_issue"
    MUTATING = True
    DESCRIPTION = (
        "Close an issue, optionally with a comment explaining why. GitHub has no way to "
        "delete an issue, so closing is as far as it goes. Use state_reason='not_planned' "
        "when the issue is being rejected rather than resolved."
    )
    INPUT_MODEL = GitHubIssueCloseInput
    AGENT_FIELDS = {
        "issue_number": (int, Field(..., description="Issue number", ge=1)),
        "state_reason": (Optional[str], Field(
            "completed", description="completed, not_planned, or duplicate")),
        "comment": (Optional[str], Field(None, description="Comment to add alongside the close")),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "closed": output.issue is not None and output.issue.state == "closed",
            "comment_added": output.comment is not None,
            **_issue_summary(output.issue),
        }


class GitHubReopenIssueTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "reopen_issue"
    TOOL_NAME = "github_reopen_issue"
    MUTATING = True
    DESCRIPTION = "Reopen a closed issue."
    INPUT_MODEL = GitHubIssueReopenInput
    AGENT_FIELDS = {
        "issue_number": (int, Field(..., description="Issue number", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "reopened": output.issue is not None and output.issue.state == "open",
            **_issue_summary(output.issue),
        }


class GitHubAddIssueCommentTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "add_comment"
    TOOL_NAME = "github_add_issue_comment"
    MUTATING = True
    DESCRIPTION = "Add a comment to a GitHub issue. Returns the new comment's id."
    INPUT_MODEL = GitHubIssueCommentCreateInput
    AGENT_FIELDS = {
        "issue_number": (int, Field(..., description="Issue number", ge=1)),
        "body": (str, Field(..., description="Comment body in Markdown", min_length=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"added": output.comment is not None, **_comment_summary(output.comment)} \
            if output.comment else {"added": False}


class GitHubUpdateIssueCommentTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "update_comment"
    TOOL_NAME = "github_update_issue_comment"
    MUTATING = True
    DESCRIPTION = (
        "Replace the body of an existing issue comment. Takes the comment id, not the issue "
        "number -- get it from github_list_issue_comments."
    )
    INPUT_MODEL = GitHubIssueCommentUpdateInput
    AGENT_FIELDS = {
        "comment_id": (int, Field(..., description="Comment id", ge=1)),
        "body": (str, Field(..., description="Replacement body in Markdown", min_length=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"updated": output.comment is not None, **_comment_summary(output.comment)} \
            if output.comment else {"updated": False}


class GitHubDeleteIssueCommentTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "delete_comment"
    TOOL_NAME = "github_delete_issue_comment"
    MUTATING = True
    DESCRIPTION = (
        "Delete an issue comment permanently. Unlike issues, comments really are deleted "
        "and cannot be recovered."
    )
    INPUT_MODEL = GitHubIssueCommentDeleteInput
    AGENT_FIELDS = {
        "comment_id": (int, Field(..., description="Comment id", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"deleted": output.deleted_id is not None, "comment_id": output.deleted_id}


class GitHubAddLabelsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "add_labels"
    TOOL_NAME = "github_add_labels"
    MUTATING = True
    DESCRIPTION = (
        "Add labels to an issue, keeping any it already has. Label names must already exist "
        "on the repository -- github_list_labels shows which do. A name that does not exist "
        "is reported as an error rather than applied."
    )
    INPUT_MODEL = GitHubIssueAddLabelsInput
    AGENT_FIELDS = {
        "issue_number": (int, Field(..., description="Issue number", ge=1)),
        "labels": (List[str], Field(..., description="Labels to add", min_length=1)),
        # Defaults to true, unlike the service, which defaults it off for
        # backwards compatibility. GitHub silently *creates* an unknown label,
        # so without this a typo does not fail -- it quietly adds a near-duplicate
        # to the repository's vocabulary, and nothing surfaces that. An agent
        # inventing a plausible-sounding label is exactly the expected failure,
        # so the safe behaviour is the default and the escape hatch is explicit.
        "validate_labels": (Optional[bool], Field(
            True, description="Reject label names that do not already exist on the "
                              "repository. Set false only to deliberately create a new label.")),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"labels": output.issue.labels if output.issue else [],
                **_issue_summary(output.issue)}


class GitHubRemoveLabelsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "remove_labels"
    TOOL_NAME = "github_remove_labels"
    MUTATING = True
    DESCRIPTION = (
        "Remove labels from an issue. Removing a label the issue does not carry is a no-op. "
        "If some labels are removed and one then fails, the response says which."
    )
    INPUT_MODEL = GitHubIssueRemoveLabelsInput
    AGENT_FIELDS = {
        "issue_number": (int, Field(..., description="Issue number", ge=1)),
        "labels": (List[str], Field(..., description="Labels to remove", min_length=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"labels": output.issue.labels if output.issue else [],
                **_issue_summary(output.issue)}


class GitHubAddAssigneesTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "add_assignees"
    TOOL_NAME = "github_add_assignees"
    MUTATING = True
    DESCRIPTION = "Assign users to an issue by GitHub login."
    INPUT_MODEL = GitHubIssueAddAssigneesInput
    AGENT_FIELDS = {
        "issue_number": (int, Field(..., description="Issue number", ge=1)),
        "assignees": (List[str], Field(..., description="Logins to assign", min_length=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"assignees": output.issue.assignees if output.issue else [],
                **_issue_summary(output.issue)}


class GitHubRemoveAssigneesTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "remove_assignees"
    TOOL_NAME = "github_remove_assignees"
    MUTATING = True
    DESCRIPTION = "Unassign users from an issue by GitHub login."
    INPUT_MODEL = GitHubIssueRemoveAssigneesInput
    AGENT_FIELDS = {
        "issue_number": (int, Field(..., description="Issue number", ge=1)),
        "assignees": (List[str], Field(..., description="Logins to unassign", min_length=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"assignees": output.issue.assignees if output.issue else [],
                **_issue_summary(output.issue)}


# ---------------------------------------------------------------------------
# Repository vocabulary
#
# These three exist because of a recurring failure: an operation that takes a
# name or number the agent was never given a way to look up. add_labels needs
# label names that exist, add_assignees needs logins that can be assigned, and
# create_issue takes a milestone *number* while a human refers to it by title.
# Without these an agent guesses, and a guess either errors or -- worse, in the
# label case -- silently succeeds by creating something new.
# ---------------------------------------------------------------------------

class GitHubFindIssuesByBodyTool(GitHubServiceTool):
    """Find issues by a marker in their body, without the search index.

    `github_search_issues` is the obvious tool and is wrong for one case:
    GitHub's search index lags creation by roughly a minute, and "have I already
    filed this?" is always asked inside exactly that window.

    The scan runs service-side over bodies it already fetches, and returns only
    matched summaries -- no body reaches the model, so the section 6 token budget
    is untouched. This is a thin wrapper over one operation like every other
    tool here; the paging, the dedupe-by-number and the page budget all live in
    the service, which owns the loop and the rate-limit budget.
    """

    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "find_issues_by_body"
    TOOL_NAME = "github_find_issues_by_body"
    DESCRIPTION = (
        "Find issues containing an exact marker string in their body. Use this instead of "
        "github_search_issues when checking whether something was ALREADY FILED: search is "
        "indexed and lags new issues by about a minute, while this reads the issues directly. "
        "Searches open and closed issues by default, since a closed duplicate is still a "
        "duplicate. "
        "IMPORTANT: check the `complete` field. If complete is false the scan ran out of "
        "budget before reaching the end, so finding no match does NOT mean none exists -- say "
        "so rather than reporting the item as new, and pass next_page to resume."
    )
    INPUT_MODEL = GitHubIssueFindByBodyInput
    AGENT_FIELDS = {
        "contains": (str, Field(..., description="Exact marker text to look for in issue bodies",
                                min_length=1)),
        "match": (Optional[str], Field(
            "line", description="'line' matches only a whole stripped line, the right choice "
                                "for marker conventions; 'substring' matches anywhere")),
        "state": (Optional[str], Field("all", description="open, closed, or all")),
        "labels": (Optional[List[str]], Field(None, description="Narrow the scan to these labels")),
        "max_results": (Optional[int], Field(10, description="Stop after this many matches", ge=1, le=100)),
        "max_pages": (Optional[int], Field(5, description="Pages of 100 to scan before giving up", ge=1, le=20)),
        "page": (Optional[int], Field(None, description="Pass next_page from a previous "
                                                        "incomplete scan to resume", ge=1)),
    }
    # `since` is deliberately omitted, unlike on github_list_issues. It filters on
    # updated_at rather than created_at, which makes it unsafe as a bound on a
    # duplicate check in both directions -- and duplicate checking is what this
    # tool is for. A model offered a plausible-looking "speed this up" knob will
    # use it. Section 15.3: AGENT_FIELDS is a deliberate subset.

    def project(self, output) -> Dict[str, Any]:
        return {
            "matches": [_issue_summary(i) for i in output.issues],
            "match_count": len(output.issues),
            "scanned": output.scanned,
            # Deliberately first-class rather than folded into paging(): an
            # incomplete scan that found nothing is not "no duplicate exists",
            # and a false absence is what files the duplicate.
            "complete": output.complete,
            "next_page": output.next_page,
        }


class GitHubListLabelsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_labels"
    TOOL_NAME = "github_list_labels"
    DESCRIPTION = (
        "List the labels defined on the repository, with their descriptions. Call this "
        "before github_add_labels to use a name that exists rather than inventing one."
    )
    INPUT_MODEL = GitHubIssueListLabelsInput
    AGENT_FIELDS = {
        "max_results": (Optional[int], Field(100, description="Maximum labels to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "labels": [{"name": l.name, "description": l.description} for l in output.labels],
            **self.paging(output),
        }


class GitHubListMilestonesTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_milestones"
    TOOL_NAME = "github_list_milestones"
    DESCRIPTION = (
        "List the repository's milestones with their numbers, titles and progress. "
        "github_create_issue and github_update_issue take a milestone NUMBER, not a title, "
        "so use this to translate one to the other."
    )
    INPUT_MODEL = GitHubIssueListMilestonesInput
    AGENT_FIELDS = {
        "state": (Optional[str], Field("open", description="Milestone state: open, closed or all")),
        "max_results": (Optional[int], Field(30, description="Maximum milestones to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "milestones": [{
                "number": m.number,
                "title": m.title,
                "state": m.state,
                "open_issues": m.open_issues,
                "closed_issues": m.closed_issues,
                "due_on": m.due_on,
            } for m in output.milestones],
            **self.paging(output),
        }


class GitHubListAssignableUsersTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_assignable_users"
    TOOL_NAME = "github_list_assignable_users"
    DESCRIPTION = (
        "List the user logins that can be assigned to an issue or pull request in this "
        "repository. Use this before github_add_assignees -- GitHub ignores a login that "
        "cannot be assigned rather than refusing it."
    )
    INPUT_MODEL = GitHubIssueListAssignableUsersInput
    AGENT_FIELDS = {
        "max_results": (Optional[int], Field(50, description="Maximum users to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"assignable_users": output.assignable_users, **self.paging(output)}


ISSUE_TOOLS.update({
    cls.TOOL_NAME: cls for cls in (
        GitHubFindIssuesByBodyTool,
        GitHubListLabelsTool,
        GitHubListMilestonesTool,
        GitHubListAssignableUsersTool,
    )
})


ISSUE_TOOLS.update({
    cls.TOOL_NAME: cls for cls in (
        GitHubCreateIssueTool,
        GitHubUpdateIssueTool,
        GitHubCloseIssueTool,
        GitHubReopenIssueTool,
        GitHubAddIssueCommentTool,
        GitHubUpdateIssueCommentTool,
        GitHubDeleteIssueCommentTool,
        GitHubAddLabelsTool,
        GitHubRemoveLabelsTool,
        GitHubAddAssigneesTool,
        GitHubRemoveAssigneesTool,
    )
})
