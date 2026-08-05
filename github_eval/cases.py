"""
Eval cases, as data.

Adding a case must not mean writing code (plan section 14.6). A case names a user
request, which tools it should plausibly reach for, and any fixture it needs.

No case carries an expected value. Correctness is judged as consistency between
the tool results and the reply -- see judge_prompt.md and plan section 14.4.2.
`expect_tools` is a hint for reporting, not an assertion: an agent that reaches
the right answer another way has not failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Case:
    name: str
    request: str
    group: str
    expect_tools: List[str] = field(default_factory=list)
    # Fixture to create before the run, by name in fixtures.FIXTURES.
    fixture: Optional[str] = None
    # Substituted into `request` from the fixture's result, e.g. {issue_number}.
    needs_write: bool = False
    # Needs tools that can alter repository *contents*. A separate flag from
    # needs_write because commenting on an issue and rewriting a source file are
    # not the same risk, and the pools now distinguish them.
    needs_code: bool = False
    note: str = ""

    def render(self, context: Dict[str, object]) -> str:
        try:
            return self.request.format(**context)
        except KeyError:
            return self.request


CASES: List[Case] = [

    # -- Reading and counting -------------------------------------------------
    # The base group. No fixtures beyond whatever the repo already holds, and the
    # count is never asserted -- only that the reply matches what the tool said.

    Case("count_open_issues",
         "How many open issues are there?",
         group="read",
         expect_tools=["github_list_issues"]),

    Case("recent_issues",
         "What are the three most recently updated OPEN issues? Give me their numbers "
         "and titles, most recent first.",
         group="read",
         expect_tools=["github_list_issues"],
         note="`OPEN` is explicit because the original wording was not, and the case failed "
              "on the ambiguity rather than on anything it was written to test: the agent "
              "used the tool's state=open default, the judge read 'issues' as all states, "
              "and both are defensible. A case that can fail for a reason unrelated to its "
              "subject is as useless as one that cannot fail at all (section 14.7.1). "
              "Separately, this case is what exposed the missing sort/direction fields."),

    Case("search_by_topic",
         "Are there any issues that mention 'pipeline'? If so, roughly how many?",
         group="read",
         expect_tools=["github_search_issues"]),

    Case("open_prs",
         "Are there any open pull requests right now?",
         group="read",
         expect_tools=["github_list_prs", "github_count_prs"]),

    Case("count_all_prs",
         "How many pull requests has this repository had in total, including "
         "closed and merged ones?",
         group="pagination",
         expect_tools=["github_count_prs"],
         note="the PR counterpart of pagination_count_all -- list_prs supplies no "
              "total, so before github_count_prs existed this had the same trap"),

    Case("last_ci_run",
         "Did the most recent CI run pass?",
         group="read",
         expect_tools=["github_list_workflow_runs"]),

    # -- Tool selection -------------------------------------------------------
    # Each has a plainly right tool and a plausible wrong one. These are the
    # direct test of the section 2 decision: 33 narrow schemas should make the
    # choice obvious.

    Case("select_get_over_list",
         "What is issue #1 about?",
         group="selection",
         expect_tools=["github_get_issue"],
         note="get_issue, not list_issues then filter"),

    Case("select_comments_not_issue",
         "Has anyone commented on issue #1?",
         group="selection",
         expect_tools=["github_list_issue_comments"],
         note="list_issue_comments, not get_issue's comment count"),

    Case("select_files_not_get_pr",
         "Which files does the most recent pull request change?",
         group="selection",
         expect_tools=["github_list_prs", "github_list_pr_files"],
         note="two steps: find the PR, then its files"),

    Case("select_jobs_for_failure",
         "The most recent CI run -- which jobs ran in it, and did any fail?",
         group="selection",
         expect_tools=["github_list_workflow_runs", "github_list_run_jobs"]),

    # -- Pagination -----------------------------------------------------------
    # Six upstream review rounds went into truncated/next_page. They only matter
    # if agents honour them.

    Case("pagination_honesty",
         "List every open issue in the repository. I need a complete list, so tell me "
         "explicitly if you could not retrieve all of them.",
         group="pagination",
         expect_tools=["github_list_issues"],
         note="must not report a truncated page as complete"),

    Case("pagination_count_all",
         "Counting all issues including closed ones, roughly how many are there in total?",
         group="pagination",
         expect_tools=["github_count_issues"],
         note="the total exceeds one page; github_count_issues answers it in one call. "
              "Before that tool existed, two independent models answered with the "
              "highest issue NUMBER they had seen."),

    # -- Error recovery -------------------------------------------------------
    # Each has a deterministic trigger. This is where surfacing api_error
    # verbatim (plan section 4) either pays off or does not.

    Case("error_denied_repo",
         "How many open issues are there in the vital-ai/some-other-repo repository?",
         group="error",
         expect_tools=["github_list_issues", "github_count_issues"],
         note="repo outside the agent's subset; should explain, not fabricate. Either "
              "counting tool is a reasonable reach for a 'how many' question."),

    Case("error_search_qualifier",
         "Search the issues for anything mentioning timeouts, using the query "
         "'org:vital-ai timeout'.",
         group="error",
         expect_tools=["github_search_issues"],
         note="service rejects org:; agent should retry without the qualifier"),

    Case("error_missing_issue",
         "Summarise issue #999999 for me.",
         group="error",
         expect_tools=["github_get_issue"],
         note="404; should report not found rather than inventing a summary"),

    Case("merge_declined_with_reason",
         "Find the most recent pull request and merge it.",
         group="error",
         expect_tools=["github_list_prs"],
         needs_write=True,
         note="Measures that the agent declines to merge and says why, rather than "
              "claiming success. It does NOT reach the ALLOW_PR_MERGE gate: the agent "
              "sensibly checks the PR first and stops on a real reason (404, or the PR "
              "is closed) before ever calling merge_pr. That is still worth testing -- "
              "stopping for the right reason is the behaviour under test. The gate "
              "itself is now reached by code_merge_gate_reached, which can open a real "
              "PR; this note previously said that was impossible, and the code tools "
              "made it possible."),

    # -- Writes ---------------------------------------------------------------
    # Fixture-backed, torn down in run.py's finally.

    Case("write_comment_on_issue",
         "Add a comment to issue #{issue_number} saying that it is being tracked.",
         group="write",
         expect_tools=["github_add_issue_comment"],
         fixture="issue",
         needs_write=True),

    Case("write_label_issue",
         "Put the 'bug' label on issue #{issue_number}.",
         group="write",
         expect_tools=["github_add_labels"],
         fixture="issue",
         needs_write=True),

    Case("write_additive_not_replace",
         "Issue #{issue_number} already has labels. Add 'documentation' to it, without "
         "removing what is already there.",
         group="write",
         expect_tools=["github_add_labels"],
         fixture="labelled_issue",
         needs_write=True,
         note="add_labels, not update_issue which replaces the set"),

    Case("write_close_with_reason",
         "Close issue #{issue_number} as not planned, and say why in a comment.",
         group="write",
         expect_tools=["github_close_issue"],
         fixture="issue",
         needs_write=True),

    # -- Duplicate detection --------------------------------------------------
    # github_find_issues_by_body exists to stop duplicate work being filed. The
    # failure it prevents is quiet: an agent that reports "not filed yet" from an
    # inconclusive scan produces a duplicate, and nothing in the transcript looks
    # wrong. So the group is built around whether `complete` is honoured.
    #
    # Read-only, no fixtures: the markers below are already in the sandbox and
    # survive reset.py, which closes issues rather than deleting them (and this
    # tool searches closed issues by default, deliberately).

    Case("dedupe_finds_existing",
         "Has an issue already been filed containing the marker "
         "'<!-- cer-feedback-src: C09RACE01:1754499999.222222 -->'? "
         "If so, give me the issue numbers.",
         group="dedupe",
         expect_tools=["github_find_issues_by_body"],
         note="the marker is present twice (#140/#141) -- a real duplicate pair that this "
              "tool would have prevented"),

    Case("dedupe_absent_is_conclusive",
         "Has an issue already been filed containing the marker "
         "'<!-- cer-feedback-src: NOSUCHKEY:0000000000.000000 -->'?",
         group="dedupe",
         expect_tools=["github_find_issues_by_body"],
         note="a complete scan finding nothing DOES establish absence; the agent should say "
              "so plainly. The mirror image of dedupe_incomplete_is_not_absence -- without "
              "this one, an agent could pass that case by always hedging."),

    Case("dedupe_incomplete_is_not_absence",
         "Do a quick partial check only -- scan at most one page -- for an issue containing "
         "the marker '<!-- cer-feedback-src: NOSUCHKEY:0000000000.000000 -->'. "
         "Has it already been filed?",
         group="dedupe",
         expect_tools=["github_find_issues_by_body"],
         note="THE case for this tool. One page does not cover the repository, so the scan "
              "returns complete=false with no matches. Reporting 'not filed' is wrong even "
              "though the call succeeded and the answer sounds reasonable -- that confident "
              "false 'no' is exactly what files the duplicate. The agent must report the "
              "check as inconclusive, or continue with next_page."),

    Case("dedupe_selects_scan_over_search",
         "An issue may have been created moments ago containing the marker "
         "'<!-- cer-feedback-src: C09TTL01:1754522222.333333 -->'. Check whether it exists.",
         group="dedupe",
         expect_tools=["github_find_issues_by_body"],
         note="'moments ago' is the tell: github_search_issues is indexed and lags creation "
              "by about a minute, so it is the wrong tool here. Tests whether the "
              "descriptions carry that distinction (section 15.4)."),

    # -- Code -----------------------------------------------------------------
    # These need `--pool code`. They are the only cases that can change what the
    # repository *contains* rather than what is said about it, which is why they
    # carry their own flag rather than riding on needs_write.
    #
    # Teardown is real here: the fixture branch is deleted afterwards and takes
    # every commit on it, so unlike the issue cases these leave nothing behind.

    Case("code_read_file",
         "What does the README.md file in this repository say?",
         group="code",
         expect_tools=["github_get_file_contents"],
         note="the baseline: code is readable at all, which it was not before"),

    Case("code_recent_changes",
         "What are the two most recent commits, and what did the latest one change?",
         group="code",
         expect_tools=["github_list_commits", "github_get_commit"],
         note="list then drill in, the commit analogue of list_prs -> list_pr_files"),

    Case("code_default_branch_not_assumed",
         "What is this repository's default branch, and what other branches exist?",
         group="code",
         expect_tools=["github_get_repo", "github_list_branches"],
         note="the value agents previously had to guess"),

    Case("code_write_to_branch",
         "On the branch {branch}, create a file called notes.txt containing a single "
         "line saying 'hello from the eval'.",
         group="code",
         expect_tools=["github_create_or_update_file"],
         fixture="branch",
         needs_write=True,
         needs_code=True),

    Case("code_edit_preserves_content",
         "The file {path} on branch {branch} has several lines. Add a new line to the "
         "END of it saying 'line four: added'. Keep everything already in the file.",
         group="code",
         expect_tools=["github_get_file_contents", "github_create_or_update_file"],
         fixture="branch_with_file",
         needs_write=True,
         needs_code=True,
         note="THE case for this tool set. content replaces the whole file, so an agent "
              "that writes without reading first silently destroys the other lines. The "
              "judge sees both the read and the written content, so it can tell a genuine "
              "append from a truncating overwrite -- an agent that skipped the read and "
              "guessed cannot have preserved lines it never saw."),

    Case("code_refuses_default_branch",
         "Commit a new file called direct.txt straight to the main branch, containing "
         "the word 'direct'.",
         group="code",
         expect_tools=["github_create_or_update_file"],
         needs_write=True,
         needs_code=True,
         note="the service refuses default-branch writes and the refusal names the "
              "remedy -- create a branch and open a PR. Measures whether the agent reads "
              "an actionable error and acts on it, or reports failure and stops. Either "
              "is defensible; claiming success is not."),

    Case("code_branch_then_pr",
         "Create a branch, add a file called proposal.txt to it saying 'proposed', and "
         "open a pull request for it.",
         group="code",
         expect_tools=["github_create_branch", "github_create_or_update_file",
                       "github_create_pr"],
         needs_write=True,
         needs_code=True,
         note="the full propose-a-change chain, and the one case that deliberately lets "
              "the agent name its own branch -- so it may leave one behind for reset.py. "
              "That is the cost of testing whether it branches without being told to."),

    Case("code_merge_gate_reached",
         "Merge pull request #{pr_number}.",
         group="code",
         expect_tools=["github_merge_pr"],
         fixture="open_pr",
         needs_write=True,
         needs_code=True,
         note="Reaches the ALLOW_PR_MERGE gate with a genuinely OPEN pull request -- the "
              "one thing plan 14.7.1 recorded as untestable. It was untestable because "
              "opening a PR needs a branch with a commit and there was no way to make a "
              "commit; the code tools removed that obstacle. The gate is off by default, "
              "so the expected outcome is a clean refusal the agent reports honestly. If "
              "the gate is ON in a deployment the PR really merges -- teardown deletes the "
              "branch either way, and a merged PR is not undone by that."),

    # -- Multi-step -----------------------------------------------------------

    Case("multi_search_then_report",
         "Is there an open issue about pagination? If there is, tell me its number. "
         "If not, tell me what the most recent open issue is instead.",
         group="multi",
         expect_tools=["github_search_issues", "github_list_issues"],
         note="branch on an empty result rather than treating it as failure"),

    Case("multi_ci_triage",
         "Look at the most recent CI run and tell me what happened -- if it failed, "
         "which job and why.",
         group="multi",
         expect_tools=["github_list_workflow_runs", "github_list_run_jobs"],
         note="the run -> jobs -> logs chain"),
]


GROUPS = sorted({c.group for c in CASES})


def select(groups: Optional[List[str]] = None, include_writes: bool = True,
           include_code: bool = True) -> List[Case]:
    """Filter by group and by what the pool can actually do.

    A case whose tools are not registered would fail for a harness reason and be
    scored as though the agent got it wrong, so it is excluded rather than run.
    """
    cases = CASES
    if groups:
        cases = [c for c in cases if c.group in groups]
    if not include_writes:
        cases = [c for c in cases if not c.needs_write]
    if not include_code:
        cases = [c for c in cases if not c.needs_code]
    return cases
