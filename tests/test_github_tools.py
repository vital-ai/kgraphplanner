"""Agent-facing GitHub tools: schema construction, repo scoping, error contract.

Offline. The service call is stubbed, so these cover the layer this repo owns:
the per-instance schema, the repo subset, and what an agent is handed back when
something goes wrong.
"""

import json

import pytest
from pydantic import ValidationError

from kgraphplanner.tools.github import GITHUB_TOOLS, GitHubToolConfigError
from kgraphplanner.tools.github.issue_tools import (
    GitHubGetIssueTool, GitHubListIssuesTool, GitHubSearchIssuesTool,
)
from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
    GitHubIssue, GitHubIssueToolOutput,
)

SANDBOX = "vital-ai/vital-ai-sandbox"
OTHER = "vital-ai/other"


def _cfg(repos=(SANDBOX,), endpoint="http://localhost:8008"):
    # Do not coerce: a bare string is a supported config shape, and list() would
    # turn it into characters before the tool ever sees it.
    return {"tool_endpoint": endpoint,
            "repos": repos if isinstance(repos, str) else list(repos)}


def _placeholder(annotation):
    """A value satisfying `annotation`, for tests that call every tool generically.

    The 33 operations take ints, strings, string lists and literals; a wrong type
    fails validation before the behaviour under test is reached, so this has to
    follow the declared type rather than guess one.
    """
    import typing
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Literal:
        return args[0]
    if origin in (list, typing.List):
        return [_placeholder(args[0]) if args else "x"]
    if annotation is bool:
        return True
    if annotation is int:
        return 1
    return "x"


def _issue(number=1, **kw):
    base = dict(number=number, title=f"issue {number}", state="open",
                html_url=f"https://github.com/{SANDBOX}/issues/{number}")
    base.update(kw)
    return GitHubIssue(**base)


class TestRepoConfig:

    def test_missing_repos_fails_at_registration(self):
        """Fail closed and loudly: a tool that can reach nothing is a config
        error, not a tool that refuses every call at run time."""
        with pytest.raises(GitHubToolConfigError, match="No repos configured"):
            GitHubGetIssueTool({"tool_endpoint": "x"})

    def test_empty_repos_list_fails(self):
        with pytest.raises(GitHubToolConfigError):
            GitHubGetIssueTool(_cfg(repos=[]))

    def test_malformed_repo_fails(self):
        with pytest.raises(GitHubToolConfigError, match="Malformed"):
            GitHubGetIssueTool(_cfg(repos=["not-a-repo"]))

    def test_a_bare_string_is_accepted(self):
        assert GitHubGetIssueTool(_cfg(repos=SANDBOX)).repos == [SANDBOX]

    def test_order_is_preserved_and_duplicates_dropped(self):
        t = GitHubGetIssueTool(_cfg(repos=["a/b", "c/d", "a/b"]))
        assert t.repos == ["a/b", "c/d"]


class TestSchema:

    def test_single_repo_is_optional_and_defaulted(self):
        schema = GitHubGetIssueTool(_cfg()).get_tool_schema().model_json_schema()
        assert schema["properties"]["repo"]["default"] == SANDBOX
        assert "repo" not in schema.get("required", [])

    def test_multiple_repos_are_required_and_enumerated(self):
        schema = GitHubGetIssueTool(_cfg(repos=["a/b", "c/d"])).get_tool_schema().model_json_schema()
        assert schema["properties"]["repo"]["enum"] == ["a/b", "c/d"]
        assert "repo" in schema["required"]

    def test_operation_is_not_exposed_to_the_model(self):
        """The service discriminates on `operation`; the agent should not have to."""
        schema = GitHubGetIssueTool(_cfg()).get_tool_schema().model_json_schema()
        assert "operation" not in schema["properties"]

    def test_owner_and_repo_are_not_exposed_separately(self):
        schema = GitHubGetIssueTool(_cfg()).get_tool_schema().model_json_schema()
        assert "owner" not in schema["properties"]

    def test_operation_fields_are_present(self):
        schema = GitHubListIssuesTool(_cfg()).get_tool_schema().model_json_schema()
        assert {"state", "labels", "assignee", "max_results", "page"} <= set(schema["properties"])

    def test_description_names_the_repo(self):
        assert SANDBOX in GitHubGetIssueTool(_cfg()).get_tool_description()

    def test_description_lists_all_repos_when_several(self):
        desc = GitHubGetIssueTool(_cfg(repos=["a/b", "c/d"])).get_tool_description()
        assert "a/b" in desc and "c/d" in desc


class TestRepoScoping:

    async def test_out_of_subset_repo_is_rejected_by_the_schema(self):
        """The Literal means the model cannot emit a repo that would be rejected --
        it fails validation before any call is made."""
        fn = GitHubGetIssueTool(_cfg(repos=["a/b", "c/d"])).get_tool_function()
        with pytest.raises(ValidationError):
            await fn.ainvoke({"repo": OTHER, "issue_number": 1})

    async def test_direct_call_bypassing_the_schema_is_still_refused(self):
        """Defence in depth for callers that do not go through args_schema."""
        result = json.loads(await GitHubGetIssueTool(_cfg()).run(repo=OTHER, issue_number=1))
        assert "not available to this tool" in result["error"]
        assert SANDBOX in result["error"]

    async def test_single_repo_is_filled_in_when_omitted(self, monkeypatch):
        tool = GitHubGetIssueTool(_cfg())
        seen = {}

        async def fake(wire_input):
            seen["owner"], seen["repo"] = wire_input.owner, wire_input.repo
            return GitHubIssueToolOutput(operation="get_issue", issue=_issue()), None

        monkeypatch.setattr(tool, "call_service", fake)
        await tool.run(issue_number=1)
        assert (seen["owner"], seen["repo"]) == tuple(SANDBOX.split("/"))

    async def test_repo_required_when_several_and_omitted(self):
        result = json.loads(await GitHubGetIssueTool(_cfg(repos=["a/b", "c/d"])).run(issue_number=1))
        assert "required" in result["error"]


class TestErrorContract:
    """The service returns expected failures as HTTP 200 with api_error set, and
    those messages are written to be actionable. They must reach the model."""

    async def test_api_error_is_surfaced_verbatim(self, monkeypatch):
        message = ("Write operations are disabled ({ENV}__TOOL__GITHUB__ALLOW_WRITES=false); "
                   "'create_issue' was rejected.")
        tool = GitHubGetIssueTool(_cfg())

        async def fake(wire_input):
            return GitHubIssueToolOutput(operation="get_issue", api_error=message,
                                         api_status_code=403), None

        monkeypatch.setattr(tool, "call_service", fake)
        result = json.loads(await tool.run(issue_number=1))
        assert result["error"] == message
        assert result["status_code"] == 403

    async def test_auth_failure_is_reported_not_raised(self, monkeypatch):
        """The client raises PermissionError on 401/403. Uncaught, that ends the
        agent turn instead of becoming something the agent can report."""
        tool = GitHubGetIssueTool(_cfg())

        async def boom(*a, **k):
            raise PermissionError("Tool server auth error (401)")

        monkeypatch.setattr(
            "kgraphplanner.tools.github.github_service_tool.VitalAgentRestResourceClient",
            lambda *a, **k: type("C", (), {"handle_tool_request": boom})(),
        )
        result = json.loads(await tool.run(issue_number=1))
        assert "Not authorised" in result["error"]

    async def test_missing_endpoint_is_reported(self):
        tool = GitHubGetIssueTool({"repos": [SANDBOX]})
        result = json.loads(await tool.run(issue_number=1))
        assert "endpoint is not configured" in result["error"]

    async def test_bad_arguments_are_reported_not_raised(self, monkeypatch):
        tool = GitHubGetIssueTool(_cfg())
        result = json.loads(await tool.run(issue_number="not-a-number"))
        assert "Invalid arguments" in result["error"]


class TestProjection:

    async def test_list_carries_the_pagination_contract(self, monkeypatch):
        """Dropping these makes six rounds of upstream work invisible to agents,
        which then report partial results as complete."""
        tool = GitHubListIssuesTool(_cfg())

        async def fake(wire_input):
            return GitHubIssueToolOutput(
                operation="list_issues", issues=[_issue(1), _issue(2)],
                returned_count=2, truncated=True, next_page=3,
            ), None

        monkeypatch.setattr(tool, "call_service", fake)
        result = json.loads(await tool.run(max_results=2))
        assert result["truncated"] is True
        assert result["next_page"] == 3
        assert result["returned_count"] == 2
        assert [i["number"] for i in result["issues"]] == [1, 2]

    async def test_get_issue_returns_the_body_list_does_not(self, monkeypatch):
        """Lists are summaries; only the single-issue read carries a body."""
        out = GitHubIssueToolOutput(operation="x", issue=_issue(1, body="full body"),
                                    issues=[_issue(1, body="full body")])

        get_tool = GitHubGetIssueTool(_cfg())
        list_tool = GitHubListIssuesTool(_cfg())

        async def fake(wire_input):
            return out, None

        monkeypatch.setattr(get_tool, "call_service", fake)
        monkeypatch.setattr(list_tool, "call_service", fake)

        assert json.loads(await get_tool.run(issue_number=1))["body"] == "full body"
        assert "body" not in json.loads(await list_tool.run())["issues"][0]

    async def test_missing_issue_reports_not_found_rather_than_empty(self, monkeypatch):
        tool = GitHubGetIssueTool(_cfg())

        async def fake(wire_input):
            return GitHubIssueToolOutput(operation="get_issue", issue=None), None

        monkeypatch.setattr(tool, "call_service", fake)
        assert json.loads(await tool.run(issue_number=1))["found"] is False

    async def test_search_reports_total_count_separately(self, monkeypatch):
        """total_count counts records before filtering, so it can exceed what was
        returned. Both go to the model so it can tell the difference."""
        tool = GitHubSearchIssuesTool(_cfg())

        async def fake(wire_input):
            return GitHubIssueToolOutput(operation="search_issues", issues=[_issue(1)],
                                         returned_count=1, total_count=48), None

        monkeypatch.setattr(tool, "call_service", fake)
        result = json.loads(await tool.run(query="x"))
        assert result["total_count"] == 48
        assert result["returned_count"] == 1


class TestRegistry:

    def test_every_registered_tool_is_uniquely_named(self):
        for name, cls in GITHUB_TOOLS.items():
            assert cls.TOOL_NAME == name

    def test_every_tool_declares_its_service_contract(self):
        for cls in GITHUB_TOOLS.values():
            assert cls.SERVICE_TOOL and cls.OPERATION and cls.DESCRIPTION
            assert cls.INPUT_MODEL is not None

    def test_every_tool_builds_a_function(self):
        for cls in GITHUB_TOOLS.values():
            fn = cls(_cfg()).get_tool_function()
            assert fn.name == cls.TOOL_NAME
            assert fn.description

    def test_every_operation_the_service_offers_is_reachable(self):
        """The registry should not silently lag the service. Each (service tool,
        operation) pair the client can express should have an agent-facing tool,
        or the capability exists in the mirrors and no agent can use it."""
        from kgraphplanner.vital_agent_rest_resource_client.tools.github import (
            issue_models, pr_models, actions_models, code_models, repo_models,
        )
        expected = set()
        for tool_name, mod, attr in (
            ("github_issue_tool", issue_models, "GITHUB_ISSUE_OPERATION_MODELS"),
            ("github_pr_tool", pr_models, "GITHUB_PR_OPERATION_MODELS"),
            ("github_actions_tool", actions_models, "GITHUB_ACTIONS_OPERATION_MODELS"),
            ("github_code_tool", code_models, "GITHUB_CODE_OPERATION_MODELS"),
            ("github_repo_tool", repo_models, "GITHUB_REPO_OPERATION_MODELS"),
        ):
            expected |= {(tool_name, op) for op in getattr(mod, attr)}

        covered = {(c.SERVICE_TOOL, c.OPERATION) for c in GITHUB_TOOLS.values()}
        # merge_pr routes to github_code_tool but is presented as a PR action --
        # it is covered, under the service tool it actually posts to.
        missing = expected - covered
        assert not missing, f"service operations with no agent-facing tool: {sorted(missing)}"


class TestRequestUnion:
    """Every input model must be a member of ToolRequest's tool_input union.

    This is the one registration step nothing else catches. The offline suite
    stubs call_service, so a model missing from the union passes every test here
    and then fails on the first real call with 50-odd validation errors, one per
    union member -- which reads as a mystery rather than a missing entry. It has
    now happened twice: once when the service split its tools, and again when
    write_files and get_authenticated_user were added.
    """

    def _union_members(self):
        import typing
        from kgraphplanner.vital_agent_rest_resource_client.tools.tool_request import ToolRequest
        annotation = ToolRequest.model_fields["tool_input"].annotation
        return {a for a in typing.get_args(annotation) if a is not type(None)}

    def _all_operation_models(self):
        from kgraphplanner.vital_agent_rest_resource_client.tools.github import (
            issue_models, pr_models, actions_models, code_models, repo_models,
        )
        models = set()
        for mod, attr in (
            (issue_models, "GITHUB_ISSUE_OPERATION_MODELS"),
            (pr_models, "GITHUB_PR_OPERATION_MODELS"),
            (actions_models, "GITHUB_ACTIONS_OPERATION_MODELS"),
            (code_models, "GITHUB_CODE_OPERATION_MODELS"),
            (repo_models, "GITHUB_REPO_OPERATION_MODELS"),
        ):
            models |= set(getattr(mod, attr).values())
        return models

    def test_every_operation_model_is_in_the_union(self):
        missing = self._all_operation_models() - self._union_members()
        assert not missing, (
            "not in ToolRequest.tool_input: "
            f"{sorted(m.__name__ for m in missing)} -- every real call will fail"
        )

    def test_every_tools_input_model_is_in_the_union(self):
        """Same check from the agent-facing side, since a tool can name a model
        that no operation map references."""
        union = self._union_members()
        missing = {c.INPUT_MODEL for c in GITHUB_TOOLS.values()} - union
        assert not missing, sorted(m.__name__ for m in missing)


class TestAuthorityBoundaries:
    """CODE_WRITE_TOOLS is the set that can alter repository contents. It is
    derived from the service tool name rather than declared, so these check the
    derivation rather than restating the list."""

    def test_code_write_tools_are_exactly_the_code_service_tool(self):
        from kgraphplanner.tools.github import CODE_WRITE_TOOLS
        assert {c.SERVICE_TOOL for c in CODE_WRITE_TOOLS.values()} == {"github_code_tool"}

    def test_merge_pr_counts_as_a_code_write(self):
        """It reads as a PR action and is named like one, but it lands commits on
        the base branch. Classifying by service tool catches that; classifying by
        the tool's own module would not."""
        from kgraphplanner.tools.github import CODE_WRITE_TOOLS
        assert "github_merge_pr" in CODE_WRITE_TOOLS

    def test_safe_tools_can_write_issues_but_not_code(self):
        """The distinction READ_ONLY_TOOLS cannot express: commenting on an issue
        is a write, and is nothing like replacing a source file."""
        from kgraphplanner.tools.github import SAFE_TOOLS, CODE_WRITE_TOOLS
        assert "github_add_issue_comment" in SAFE_TOOLS
        assert not (set(SAFE_TOOLS) & set(CODE_WRITE_TOOLS))

    def test_every_repo_tool_is_read_only(self):
        """github_repo_tool is read-only by construction service-side. If a
        mutating tool ever appears on it, that assumption has broken."""
        for cls in GITHUB_TOOLS.values():
            if cls.SERVICE_TOOL == "github_repo_tool":
                assert not cls.MUTATING, f"{cls.TOOL_NAME} claims to mutate via a read-only tool"

    def test_file_writes_require_an_explicit_branch(self):
        """GitHub commits to the default branch when branch is omitted, so an
        omitted field must not be the route to writing on main."""
        from kgraphplanner.tools.github.code_tools import (
            GitHubCreateOrUpdateFileTool, GitHubDeleteFileTool,
        )
        for cls in (GitHubCreateOrUpdateFileTool, GitHubDeleteFileTool):
            schema = cls(_cfg()).get_tool_schema().model_json_schema()
            assert "branch" in schema.get("required", []), f"{cls.TOOL_NAME}: branch is optional"


class TestFindIssuesByBody:
    """The duplicate-detection tool. Its whole value is that an inconclusive
    scan is distinguishable from a conclusive empty one."""

    def _tool(self):
        from kgraphplanner.tools.github.issue_tools import GitHubFindIssuesByBodyTool
        return GitHubFindIssuesByBodyTool(_cfg())

    def _output(self, **kw):
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
            GitHubIssueToolOutput,
        )
        return GitHubIssueToolOutput(operation="find_issues_by_body", **kw)

    def test_incomplete_empty_scan_is_not_reported_as_absence(self):
        """The single most important behaviour here. A scan that ran out of
        budget and found nothing must not look identical to one that searched
        everything and found nothing -- a false absence is what files the
        duplicate this tool exists to prevent."""
        result = self._tool().project(
            self._output(issues=[], scanned=500, complete=False, next_page=6))
        assert result["match_count"] == 0
        assert result["complete"] is False
        assert result["next_page"] == 6

    def test_a_conclusive_empty_scan_says_so(self):
        result = self._tool().project(self._output(issues=[], scanned=12, complete=True))
        assert result["match_count"] == 0 and result["complete"] is True

    def test_bodies_are_never_returned(self):
        """The reason this tool is affordable at all: it spends the bodies in
        the service and returns only summaries (plan section 6)."""
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
            GitHubIssue,
        )
        issue = GitHubIssue(number=1, title="t", state="open", html_url="u",
                            body="MARKER " + "x" * 4000)
        result = self._tool().project(self._output(issues=[issue], scanned=3, complete=True))
        assert result["match_count"] == 1
        assert "body" not in result["matches"][0]
        assert "x" * 100 not in json.dumps(result)

    def test_state_defaults_to_all_not_open(self):
        """A closed duplicate is still a duplicate. Defaulting to `open` --
        which is what list_issues does -- would miss it and refile."""
        schema = self._tool().get_tool_schema().model_json_schema()
        assert schema["properties"]["state"]["default"] == "all"

    def test_match_defaults_to_line(self):
        """Marker conventions are line-anchored; a substring hit that is not a
        line hit is a false positive against such a scheme."""
        schema = self._tool().get_tool_schema().model_json_schema()
        assert schema["properties"]["match"]["default"] == "line"

    def test_since_is_not_offered_to_the_model(self):
        """`since` filters on updated_at, which makes it unsafe as a bound on a
        duplicate check in both directions. It stays on the wire model for
        programmatic callers and off the agent-facing schema."""
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
            GitHubIssueFindByBodyInput,
        )
        assert "since" not in self._tool().get_tool_schema().model_json_schema()["properties"]
        assert "since" in GitHubIssueFindByBodyInput.model_fields

    def test_list_issues_does_offer_since(self):
        """...but list_issues does, for polling, where updated_at is the right
        field and the semantics are spelled out in the description."""
        from kgraphplanner.tools.github.issue_tools import GitHubListIssuesTool
        props = GitHubListIssuesTool(_cfg()).get_tool_schema().model_json_schema()["properties"]
        assert "since" in props
        assert "UPDATED" in props["since"]["description"]

    def test_it_is_one_service_call_like_every_other_tool(self):
        """The scan lives in the service. If this tool ever needed its own page
        loop, the base class invariant would be broken -- this asserts the
        design decision, not just the code."""
        tool = self._tool()
        wire = tool.build_wire_input("a/b", contains="MARKER")
        assert wire.operation == "find_issues_by_body"
        assert wire.contains == "MARKER"


class TestIdempotentCreate:
    """create_issue gained a service-reported `created` that this layer used to
    derive locally, with the opposite meaning on the deduplicated path."""

    def _tool(self):
        from kgraphplanner.tools.github.issue_tools import GitHubCreateIssueTool
        return GitHubCreateIssueTool(_cfg())

    def _out(self, **kw):
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
            GitHubIssueToolOutput, GitHubIssue,
        )
        if kw.pop("with_issue", True):
            kw["issue"] = GitHubIssue(number=7, title="t", state="open", html_url="u")
        return GitHubIssueToolOutput(operation="create_issue", **kw)

    def test_a_deduplicated_create_is_not_reported_as_new(self):
        """The bug this guards: the old projection derived created from
        `issue is not None`, which is TRUE when an existing issue was returned.
        A duplicate-suppressed create would have been reported as freshly filed
        -- the exact opposite of what happened, and invisible in the response."""
        result = self._tool().project(self._out(created=False, idempotency_guard="memorydb"))
        assert result["created"] is False
        assert result["number"] == 7
        assert result["idempotency_guard"] == "memorydb"

    def test_a_real_create_is_reported_as_new(self):
        result = self._tool().project(self._out(created=True, idempotency_guard="memorydb"))
        assert result["created"] is True

    def test_it_falls_back_when_the_service_reports_nothing(self):
        """No key supplied, or idempotency disabled: the service leaves `created`
        null and the local derivation is still correct."""
        result = self._tool().project(self._out())
        assert result["created"] is True
        assert "idempotency_guard" not in result

    def test_guard_none_is_distinguished_from_no_guard_requested(self):
        """'none' means a guarantee was asked for and could not be given; null
        means none was asked for. Collapsing them hides a degraded MemoryDB."""
        degraded = self._tool().project(self._out(created=True, idempotency_guard="none"))
        unguarded = self._tool().project(self._out(created=True))
        assert degraded["idempotency_guard"] == "none"
        assert "idempotency_guard" not in unguarded

    def test_in_flight_does_not_claim_an_issue_number(self):
        """created=false with no issue is the in-flight case. Rendering that as
        'already filed as #None' would be worse than saying nothing."""
        result = self._tool().project(
            self._out(created=False, idempotency_guard="memorydb", with_issue=False))
        assert result["created"] is False
        assert "number" not in result
        assert "in flight" in result["outcome"]

    def test_the_key_is_not_offered_to_the_model(self):
        """An idempotency key must derive deterministically from a source event,
        which a model cannot do. Wire model yes, agent schema no."""
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
            GitHubIssueCreateInput,
        )
        schema = self._tool().get_tool_schema().model_json_schema()
        assert "idempotency_key" not in schema["properties"]
        assert "idempotency_key" in GitHubIssueCreateInput.model_fields

    def test_a_programmatic_caller_can_still_supply_it(self):
        wire = self._tool().build_wire_input("a/b", title="t", idempotency_key="evt-123")
        assert wire.idempotency_key == "evt-123"


class TestLabelValidation:

    def test_add_labels_validates_by_default(self):
        """GitHub silently creates an unknown label rather than refusing it, so a
        typo would quietly enlarge the repository's vocabulary. The agent-facing
        default is stricter than the service's."""
        from kgraphplanner.tools.github.issue_tools import GitHubAddLabelsTool
        schema = GitHubAddLabelsTool(_cfg()).get_tool_schema().model_json_schema()
        assert schema["properties"]["validate_labels"]["default"] is True

    def test_the_validated_flag_reaches_the_wire(self):
        from kgraphplanner.tools.github.issue_tools import GitHubAddLabelsTool
        wire = GitHubAddLabelsTool(_cfg()).build_wire_input(
            "a/b", issue_number=1, labels=["bug"], validate_labels=True)
        assert wire.validate_labels is True
        assert wire.operation == "add_labels"


class TestPRProjection:

    async def test_files_omit_patch_unless_requested(self, monkeypatch):
        """Patches are the largest thing this tool can return; they should not
        appear in the payload when the agent did not ask for them."""
        from kgraphplanner.tools.github.pr_tools import GitHubListPRFilesTool
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.pr_models import (
            GitHubPRFile, GitHubPRToolOutput,
        )
        tool = GitHubListPRFilesTool(_cfg())

        async def fake(wire_input):
            return GitHubPRToolOutput(
                operation="list_pr_files",
                files=[GitHubPRFile(filename="a.py", status="modified",
                                    additions=3, deletions=1, changes=4)],
            ), None

        monkeypatch.setattr(tool, "call_service", fake)
        entry = json.loads(await tool.run(pr_number=1))["files"][0]
        assert "patch" not in entry
        assert entry["changes"] == 4

    async def test_patch_is_included_when_requested(self, monkeypatch):
        from kgraphplanner.tools.github.pr_tools import GitHubListPRFilesTool
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.pr_models import (
            GitHubPRFile, GitHubPRToolOutput,
        )
        tool = GitHubListPRFilesTool(_cfg())

        async def fake(wire_input):
            assert wire_input.include_patch is True
            return GitHubPRToolOutput(
                operation="list_pr_files",
                files=[GitHubPRFile(filename="a.py", patch="@@ -1 +1 @@", patch_truncated=True)],
            ), None

        monkeypatch.setattr(tool, "call_service", fake)
        entry = json.loads(await tool.run(pr_number=1, include_patch=True))["files"][0]
        assert entry["patch"] == "@@ -1 +1 @@"
        assert entry["patch_truncated"] is True


class TestActionsProjection:

    async def test_jobs_report_only_failing_steps(self, monkeypatch):
        """A green job's step list is noise, and a matrix build's is a lot of it.
        The count is kept so the agent knows steps were elided."""
        from kgraphplanner.tools.github.actions_tools import GitHubListRunJobsTool
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.actions_models import (
            GitHubActionsToolOutput, GitHubWorkflowJob, GitHubWorkflowStep,
        )
        tool = GitHubListRunJobsTool(_cfg())

        async def fake(wire_input):
            return GitHubActionsToolOutput(operation="list_run_jobs", jobs=[
                GitHubWorkflowJob(id=1, name="smoke", conclusion="failure", steps=[
                    GitHubWorkflowStep(name="checkout", conclusion="success", number=1),
                    GitHubWorkflowStep(name="test", conclusion="failure", number=2),
                    GitHubWorkflowStep(name="upload", conclusion="skipped", number=3),
                ])
            ]), None

        monkeypatch.setattr(tool, "call_service", fake)
        job = json.loads(await tool.run(run_id=1))["jobs"][0]
        assert [s["name"] for s in job["failed_steps"]] == ["test"]
        assert job["step_count"] == 3

    async def test_log_defaults_are_smaller_than_the_service(self):
        """Log text is the largest payload any of these tools can produce, and the
        agent pays for it in context."""
        from kgraphplanner.tools.github.actions_tools import GitHubGetRunLogsTool
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.actions_models import (
            GitHubActionsRunLogsInput,
        )
        schema = GitHubGetRunLogsTool(_cfg()).get_tool_schema().model_json_schema()
        service_defaults = GitHubActionsRunLogsInput.model_fields
        assert schema["properties"]["max_files"]["default"] < service_defaults["max_files"].default
        assert (schema["properties"]["max_lines_per_file"]["default"]
                < service_defaults["max_lines_per_file"].default)


class TestAllToolsShareTheContract:
    """Applies to every tool in the registry, so a new one cannot skip these."""

    @pytest.mark.parametrize("name", sorted(GITHUB_TOOLS))
    def test_schema_hides_wire_plumbing(self, name):
        props = GITHUB_TOOLS[name](_cfg()).get_tool_schema().model_json_schema()["properties"]
        assert "operation" not in props
        assert "owner" not in props

    @pytest.mark.parametrize("name", sorted(GITHUB_TOOLS))
    def test_repo_is_enumerated_not_free_text(self, name):
        """Every repository-scoped tool must make an out-of-subset repo
        unrepresentable rather than merely rejected (section 13.2)."""
        cls = GITHUB_TOOLS[name]
        schema = cls(_cfg(repos=["a/b", "c/d"])).get_tool_schema().model_json_schema()
        if not cls.REQUIRES_REPO:
            # get_authenticated_user is about the token, not a repository. It must
            # offer no `repo` at all: a field the operation ignores would invite
            # the model to supply one and read meaning into the answer.
            assert "repo" not in schema["properties"]
            return
        assert schema["properties"]["repo"]["enum"] == ["a/b", "c/d"]

    def test_only_the_token_operation_is_repo_less(self):
        """REQUIRES_REPO=False bypasses the repo allowlist check, so it must stay
        confined to the one operation that genuinely has no repository."""
        repo_less = {n for n, c in GITHUB_TOOLS.items() if not c.REQUIRES_REPO}
        assert repo_less == {"github_get_authenticated_user"}

    def test_a_repo_less_tool_never_mutates(self):
        """Skipping the allowlist is only safe for a read. If a repo-less tool
        could write, it would write somewhere no allowlist had approved."""
        for cls in GITHUB_TOOLS.values():
            if not cls.REQUIRES_REPO:
                assert not cls.MUTATING, cls.TOOL_NAME

    @pytest.mark.parametrize("name", sorted(GITHUB_TOOLS))
    async def test_api_error_reaches_the_caller(self, name, monkeypatch):
        tool = GITHUB_TOOLS[name](_cfg())
        model = tool.INPUT_MODEL

        async def fake(wire_input):
            from kgraphplanner.vital_agent_rest_resource_client.tools.github.tool_handler import HANDLERS
            output_model = HANDLERS[tool.SERVICE_TOOL].OUTPUT_MODEL
            return output_model(operation=tool.OPERATION, api_error="gate closed"), None

        monkeypatch.setattr(tool, "call_service", fake)
        required = {n: _placeholder(f.annotation)
                    for n, f in model.model_fields.items()
                    if f.is_required() and n not in ("operation", "owner", "repo")}
        result = json.loads(await tool.run(**required))
        assert result["error"] == "gate closed"


class TestCountIssues:
    """Added because the eval found two independent models answering "how many
    issues?" with the highest issue *number* they had seen. The list endpoint
    supplies no total, so an agent asked to count had nothing to count with."""

    def _tool(self):
        from kgraphplanner.tools.github.issue_tools import GitHubCountIssuesTool
        return GitHubCountIssuesTool(_cfg())

    def test_exposes_state_and_filter_not_raw_query(self):
        """The agent should not have to know GitHub search syntax to count."""
        props = self._tool().get_tool_schema().model_json_schema()["properties"]
        assert {"state", "filter"} <= set(props)
        assert "query" not in props
        assert "max_results" not in props

    @pytest.mark.parametrize("state,expected", [
        ("open", "is:issue is:open"),
        ("closed", "is:issue is:closed"),
        ("all", "is:issue"),
    ])
    def test_state_maps_to_a_search_query(self, state, expected):
        wire = self._tool().build_wire_input(SANDBOX, state=state)
        assert wire.query == expected

    def test_state_defaults_to_open(self):
        assert self._tool().build_wire_input(SANDBOX).query == "is:issue is:open"

    def test_filter_is_appended(self):
        wire = self._tool().build_wire_input(SANDBOX, state="all", filter="label:bug")
        assert wire.query == "is:issue label:bug"

    def test_pull_requests_are_excluded(self):
        """`is:issue` matches what the other issue tools return by default."""
        assert self._tool().build_wire_input(SANDBOX, state="all").query.startswith("is:issue")

    def test_asks_for_one_record_not_a_page(self):
        """Only the total is wanted; records would be payload nobody reads."""
        assert self._tool().build_wire_input(SANDBOX).max_results == 1

    async def test_returns_the_corpus_total_not_the_returned_count(self, monkeypatch):
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
            GitHubIssueToolOutput,
        )
        tool = self._tool()

        async def fake(wire_input):
            # Search returns one record but reports the full corpus total.
            return GitHubIssueToolOutput(operation="search_issues", issues=[_issue(1)],
                                         returned_count=1, total_count=54), None

        monkeypatch.setattr(tool, "call_service", fake)
        result = json.loads(await tool.run(state="all"))
        assert result["count"] == 54

    def test_description_warns_against_the_failure_it_exists_to_prevent(self):
        desc = self._tool().get_tool_description().lower()
        assert "issue number" in desc and "not a total" in desc

    def test_list_issues_points_at_the_count_tool(self):
        """An agent reading list_issues should learn counting lives elsewhere."""
        assert "github_count_issues" in GitHubListIssuesTool(_cfg()).get_tool_description()


class TestCountPRs:
    """Predicted by plan 16.1 and confirmed: list_prs has the same missing-total
    gap as list_issues. Workflow runs do not -- list_workflow_runs already
    reports GitHub's total_count, so no third count tool is needed."""

    def _tool(self):
        from kgraphplanner.tools.github.pr_tools import GitHubCountPRsTool
        return GitHubCountPRsTool(_cfg())

    def test_rides_the_issue_service_tool(self):
        """search_issues belongs to github_issue_tool and covers both issues and
        PRs. Agent-facing grouping is by what the user asks about, not by which
        service tool owns the operation (plan section 2)."""
        tool = self._tool()
        assert tool.SERVICE_TOOL == "github_issue_tool"
        assert tool.OPERATION == "search_issues"

    @pytest.mark.parametrize("state,expected", [
        ("open", "is:pr is:open"),
        ("closed", "is:pr is:closed"),
        ("merged", "is:pr is:merged"),
        ("all", "is:pr"),
    ])
    def test_state_maps_to_a_search_query(self, state, expected):
        assert self._tool().build_wire_input(SANDBOX, state=state).query == expected

    def test_pull_requests_are_not_filtered_out(self):
        """The default filters PRs from search results -- fatal when counting them."""
        assert self._tool().build_wire_input(SANDBOX, state="all").include_pull_requests is True

    def test_asks_for_one_record_not_a_page(self):
        assert self._tool().build_wire_input(SANDBOX).max_results == 1

    def test_does_not_expose_raw_search_syntax(self):
        props = self._tool().get_tool_schema().model_json_schema()["properties"]
        assert "query" not in props and {"state", "filter"} <= set(props)

    def test_list_prs_points_at_the_count_tool(self):
        from kgraphplanner.tools.github.pr_tools import GitHubListPRsTool
        assert "github_count_prs" in GitHubListPRsTool(_cfg()).get_tool_description()


class TestCountingCoverage:
    """Every list tool either reports a total or has a counting tool beside it."""

    def test_workflow_runs_already_report_a_total(self):
        """So it needs no count tool -- the gap was specific to endpoints where
        GitHub supplies no corpus total."""
        from kgraphplanner.tools.github.actions_tools import GitHubListWorkflowRunsTool
        import inspect
        src = inspect.getsource(GitHubListWorkflowRunsTool.project)
        assert "total_count" in src

    def test_both_count_tools_are_read_only(self):
        from kgraphplanner.tools.github import READ_ONLY_TOOLS
        assert {"github_count_issues", "github_count_prs"} <= set(READ_ONLY_TOOLS)


class TestRunTimestamps:
    """The eval caught an agent reporting `created_at` as the completion time when
    asked "did the last run pass?" -- it had only a start timestamp and relabelled
    it. Same class of failure as the missing count: absent field, invented answer."""

    def _summary(self):
        from kgraphplanner.tools.github.actions_tools import _run_summary
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.actions_models import (
            GitHubWorkflowRun,
        )
        return _run_summary(GitHubWorkflowRun(
            id=1, name="Pipeline Smoke", status="completed", conclusion="success",
            created_at="2026-08-04T01:20:00Z", updated_at="2026-08-04T01:25:00Z"))

    def test_both_timestamps_are_exposed(self):
        s = self._summary()
        assert s["created_at"] == "2026-08-04T01:20:00Z"
        assert s["updated_at"] == "2026-08-04T01:25:00Z"

    def test_get_run_description_distinguishes_them(self):
        from kgraphplanner.tools.github.actions_tools import GitHubGetWorkflowRunTool
        desc = GitHubGetWorkflowRunTool(_cfg()).get_tool_description()
        assert "created_at is when the run started" in desc
        assert "finished" in desc
