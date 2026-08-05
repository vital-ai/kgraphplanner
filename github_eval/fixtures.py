"""
Sandbox fixtures: create what a case needs, best-effort remove it afterwards.

Never asserts (plan section 14.6). It creates and it cleans up; the runner decides
what a fixture failure means.

Teardown is best-effort, not restoration. GitHub has no delete-issue endpoint, so
an issue can only be closed -- see plan section 14.4 for the full table of what
can and cannot be undone. Everything created carries a run marker so leftovers
from a crashed run are identifiable, and reset.py can sweep them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MARKER_PREFIX = "[eval"

# Branches cannot carry the bracketed marker -- git refs disallow those
# characters -- so they get their own prefix. reset.py sweeps by this, which is
# also why it must be distinctive enough that no real branch collides with it.
BRANCH_PREFIX = "eval-harness/"


def marker(run_id: str) -> str:
    return f"{MARKER_PREFIX} {run_id}]"


# What each fixture puts into a case's substitution context. Declared rather than
# discovered so a case can be checked against its fixture offline, without a
# sandbox: a case saying {branch} while asking for the `issue` fixture would
# otherwise only fail at run time, and would look like an agent failure.
FIXTURE_CONTEXT_KEYS: Dict[str, set] = {
    "issue": {"issue_number"},
    "labelled_issue": {"issue_number"},
    "branch": {"branch"},
    "branch_with_file": {"branch", "path", "original_lines"},
    "open_pr": {"branch", "pr_number"},
}


def branch_name(run_id: str, suffix: str = "") -> str:
    return f"{BRANCH_PREFIX}{run_id}{('-' + suffix) if suffix else ''}"


@dataclass
class Fixture:
    """What a case was given, and how to undo it."""
    name: str
    context: Dict[str, Any] = field(default_factory=dict)
    created_issues: List[int] = field(default_factory=list)
    # Unlike issues, branches can genuinely be deleted, and deleting one takes
    # every commit made on it with it. So teardown for the code cases is real
    # restoration rather than the best-effort closing the issue cases settle for
    # -- provided the agent worked on the branch it was given, which is the
    # reason the code cases hand it one rather than letting it invent a name.
    created_branches: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class FixtureFactory:
    """Builds fixtures through the tools themselves.

    Deliberately uses the agent-facing tools rather than a separate GitHub client:
    if a write tool is broken, the fixture fails loudly here rather than the case
    failing for a reason that looks like the agent's fault.
    """

    def __init__(self, tools: Dict[str, Any], run_id: str):
        self.tools = tools
        self.run_id = run_id
        # Branch names must be unique WITHIN a run, not just between runs.
        # Teardown happens once at the end, so every branch a run creates
        # coexists -- naming them all after the run id made the second fixture
        # collide with the first ("Reference already exists"). Issues do not
        # have this problem because GitHub assigns their numbers.
        self._branch_seq = 0

    async def _call(self, name: str, **kwargs) -> Dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"tool {name} is not registered"}
        try:
            return json.loads(await tool.ainvoke(kwargs))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    async def create(self, name: str) -> Fixture:
        builder = getattr(self, f"_build_{name}", None)
        if builder is None:
            return Fixture(name, errors=[f"unknown fixture {name!r}"])
        return await builder()

    async def _build_issue(self) -> Fixture:
        fixture = Fixture("issue")
        result = await self._call(
            "github_create_issue",
            title=f"{marker(self.run_id)} eval fixture",
            body="Created by the GitHub tools eval harness. Safe to close.",
        )
        if "error" in result or not result.get("number"):
            fixture.errors.append(f"could not create issue: {result.get('error', result)}")
            return fixture
        number = result["number"]
        fixture.created_issues.append(number)
        fixture.context["issue_number"] = number
        return fixture

    async def _build_labelled_issue(self) -> Fixture:
        """An issue that already carries a label.

        Needed by the additive-vs-replace case: with no existing label there is
        nothing for a wrong tool choice to destroy, so the case would pass either way.
        """
        fixture = await self._build_issue()
        if fixture.errors:
            return fixture
        fixture.name = "labelled_issue"
        result = await self._call("github_add_labels",
                                  issue_number=fixture.context["issue_number"],
                                  labels=["bug"])
        if "error" in result:
            fixture.errors.append(f"could not label issue: {result['error']}")
        return fixture

    # --- code fixtures -------------------------------------------------------

    async def _build_branch(self) -> Fixture:
        """A branch off the default branch for a code case to work on.

        The case is handed a branch rather than asked to invent one so that
        teardown knows what to delete. An agent that creates its own extra
        branch leaves it behind for reset.py; that is a deliberate trade, since
        constraining the agent to a fixed branch would stop the case testing
        whether it chooses to branch at all.
        """
        fixture = Fixture("branch")
        self._branch_seq += 1
        name = branch_name(self.run_id, str(self._branch_seq))
        result = await self._call("github_create_branch", branch=name)
        if "error" in result or not result.get("created"):
            fixture.errors.append(f"could not create branch: {result.get('error', result)}")
            return fixture
        fixture.created_branches.append(name)
        fixture.context["branch"] = name
        return fixture

    async def _build_branch_with_file(self) -> Fixture:
        """A branch that already has a file on it, for the edit cases.

        Editing an existing file is where create_or_update_file is dangerous:
        content replaces the whole file, so an agent that writes without reading
        first silently drops everything it did not repeat. With no pre-existing
        content there is nothing to lose and the case would pass either way --
        the same reasoning as labelled_issue for additive labelling.
        """
        fixture = await self._build_branch()
        if fixture.errors:
            return fixture
        fixture.name = "branch_with_file"
        body = (
            "# Eval fixture\n\n"
            "line one: do not lose me\n"
            "line two: do not lose me either\n"
            "line three: nor me\n"
        )
        result = await self._call(
            "github_create_or_update_file",
            path="eval_fixture.md", content=body,
            message=f"{marker(self.run_id)} fixture file",
            branch=fixture.context["branch"],
        )
        if "error" in result or not result.get("written"):
            fixture.errors.append(f"could not write fixture file: {result.get('error', result)}")
            return fixture
        fixture.context["path"] = "eval_fixture.md"
        fixture.context["original_lines"] = 3
        return fixture

    async def _build_open_pr(self) -> Fixture:
        """A genuinely open pull request.

        This was impossible until the code tools existed, and its absence was
        the single gap the harness recorded as untestable (plan 14.7.1): reaching
        the ALLOW_PR_MERGE gate needs an open PR, an open PR needs a branch with
        a commit on it, and there was no way to make a commit. Now there is.

        Teardown deletes the branch, which closes the PR with it.
        """
        fixture = await self._build_branch()
        if fixture.errors:
            return fixture
        fixture.name = "open_pr"
        branch = fixture.context["branch"]

        written = await self._call(
            "github_create_or_update_file",
            path="merge_gate_fixture.txt",
            content="Created by the eval harness to open a pull request.\n",
            message=f"{marker(self.run_id)} fixture commit",
            branch=branch,
        )
        if "error" in written or not written.get("written"):
            fixture.errors.append(f"could not commit to fixture branch: "
                                  f"{written.get('error', written)}")
            return fixture

        pr = await self._call(
            "github_create_pr",
            title=f"{marker(self.run_id)} eval fixture PR",
            head=branch, base="main",
            body="Opened by the GitHub tools eval harness. Safe to close.",
        )
        if "error" in pr or not pr.get("number"):
            fixture.errors.append(f"could not open PR: {pr.get('error', pr)}")
            return fixture
        fixture.context["pr_number"] = pr["number"]
        return fixture

    async def teardown(self, fixture: Fixture) -> List[str]:
        """Close what was created. Returns problems; never raises.

        A failed teardown must not mask a case result, but silently leaving
        fixtures is how a sandbox becomes unusable -- so problems are returned
        for the runner to report.
        """
        problems: List[str] = []
        for number in fixture.created_issues:
            result = await self._call(
                "github_close_issue",
                issue_number=number,
                state_reason="not_planned",
                comment=f"{marker(self.run_id)} eval finished; closing fixture.",
            )
            if "error" in result or not result.get("closed"):
                problems.append(f"issue #{number}: {result.get('error', 'close reported failure')}")

        # Deleting the branch removes every commit made on it, so unlike the
        # issue case this genuinely restores the repository -- whatever the agent
        # wrote there goes with it. Done last: a branch with an open PR against
        # it would take the PR too, and the issue closes above should have
        # happened regardless of whether this succeeds.
        for name in fixture.created_branches:
            result = await self._call("github_delete_branch", branch=name)
            if "error" in result or not result.get("deleted"):
                problems.append(
                    f"branch {name}: {result.get('error', 'delete reported failure')}")
        return problems
