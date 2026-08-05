"""Eval harness pieces that can be tested without live dependencies.

The judge/GitHub separation in the harness exists partly for this: given a
transcript, `judge.py` returns a verdict, so verdict parsing and transcript
assembly are testable offline even though the harness as a whole is not.
"""

import json

import pytest

from github_eval import cases as cases_mod
from github_eval.agent import extract_transcript, select_tool_names
from github_eval.judge import Transcript, ToolCall, parse_verdict, VERDICTS
from github_eval.report import CaseResult


class TestVerdictParsing:
    """The judge is a model, so its output shape is not guaranteed. A judge that
    wraps valid JSON in a fence should not be scored as a harness failure."""

    @pytest.mark.parametrize("payload", [
        '{"verdict": "correct", "reason": "matches the tool result"}',
        '```json\n{"verdict": "correct", "reason": "matches the tool result"}\n```',
        '```\n{"verdict": "correct", "reason": "matches the tool result"}\n```',
        'Here is my assessment:\n{"verdict": "correct", "reason": "matches the tool result"}',
    ])
    def test_json_is_found_in_various_wrappings(self, payload):
        verdict = parse_verdict(payload)
        assert verdict.verdict == "correct"
        assert verdict.passed

    @pytest.mark.parametrize("verdict_name", VERDICTS)
    def test_every_declared_verdict_parses(self, verdict_name):
        assert parse_verdict(json.dumps({"verdict": verdict_name, "reason": "x"})).verdict == verdict_name

    def test_unknown_verdict_is_flagged_rather_than_accepted(self):
        assert parse_verdict('{"verdict": "probably_fine", "reason": "x"}').verdict == "unparseable"

    def test_non_json_is_flagged(self):
        assert parse_verdict("The agent did well.").verdict == "unparseable"

    def test_empty_is_flagged(self):
        assert parse_verdict("").verdict == "unparseable"

    def test_only_correct_counts_as_passed(self):
        for name in VERDICTS:
            verdict = parse_verdict(json.dumps({"verdict": name, "reason": "x"}))
            assert verdict.passed == (name == "correct")


class TestTranscript:

    def test_prompt_carries_calls_results_and_reply(self):
        """The judge needs all three to tell 'the tool failed' from 'the agent
        misread a good response'."""
        prompt = Transcript(
            request="how many open issues?",
            tool_calls=[ToolCall("github_list_issues", {"state": "open"}, '{"returned_count": 7}')],
            reply="There are 7 open issues.",
        ).to_prompt()
        assert "how many open issues?" in prompt
        assert "github_list_issues" in prompt
        assert '"state": "open"' in prompt
        assert '"returned_count": 7' in prompt
        assert "There are 7 open issues." in prompt

    def test_absence_of_tool_calls_is_stated_explicitly(self):
        """Silence would let the judge assume the calls were merely omitted."""
        assert "none" in Transcript(request="x", reply="y").to_prompt().lower()

    def test_large_results_are_truncated_with_the_full_size_noted(self):
        prompt = Transcript(
            request="x",
            tool_calls=[ToolCall("github_get_run_logs", {}, "L" * 20000)],
            reply="y",
        ).to_prompt()
        assert len(prompt) < 20000
        assert "20000 chars total" in prompt


class _Msg:
    def __init__(self, type_, content="", tool_calls=None, tool_call_id=None):
        self.type = type_
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id


class TestTranscriptExtraction:

    def test_calls_are_stitched_to_their_results(self):
        """Results arrive as separate ToolMessages keyed by id, so they have to be
        matched back to the call that produced them."""
        result = {"messages": [
            _Msg("human", "how many?"),
            _Msg("ai", "", tool_calls=[{"id": "c1", "name": "github_list_issues",
                                        "args": {"state": "open"}}]),
            _Msg("tool", '{"returned_count": 7}', tool_call_id="c1"),
            _Msg("ai", "There are 7."),
        ]}
        transcript = extract_transcript("how many?", result)
        assert len(transcript.tool_calls) == 1
        assert transcript.tool_calls[0].name == "github_list_issues"
        assert transcript.tool_calls[0].result == '{"returned_count": 7}'
        assert transcript.reply == "There are 7."

    def test_multiple_calls_keep_their_own_results(self):
        result = {"messages": [
            _Msg("ai", "", tool_calls=[{"id": "a", "name": "t1", "args": {}},
                                       {"id": "b", "name": "t2", "args": {}}]),
            _Msg("tool", "result-b", tool_call_id="b"),
            _Msg("tool", "result-a", tool_call_id="a"),
            _Msg("ai", "done"),
        ]}
        transcript = extract_transcript("x", result)
        by_name = {c.name: c.result for c in transcript.tool_calls}
        assert by_name == {"t1": "result-a", "t2": "result-b"}

    def test_no_tool_calls_yields_an_empty_list_not_an_error(self):
        transcript = extract_transcript("x", {"messages": [_Msg("ai", "I think 4.")]})
        assert transcript.tool_calls == []
        assert transcript.reply == "I think 4."

    def test_block_style_content_is_flattened(self):
        result = {"messages": [_Msg("ai", [{"type": "text", "text": "There are 7."}])]}
        assert extract_transcript("x", result).reply == "There are 7."

    def test_the_last_non_empty_ai_message_is_the_reply(self):
        result = {"messages": [
            _Msg("ai", "", tool_calls=[{"id": "c", "name": "t", "args": {}}]),
            _Msg("tool", "r", tool_call_id="c"),
            _Msg("ai", "final answer"),
        ]}
        assert extract_transcript("x", result).reply == "final answer"


class TestToolPools:
    """Pool size is a parameter because it is one of the things the harness
    exists to measure (plan 2.0.1)."""

    def test_read_pool_contains_no_mutating_tool(self):
        from kgraphplanner.tools.github import GITHUB_TOOLS
        for name in select_tool_names("read"):
            assert not GITHUB_TOOLS[name].MUTATING, name

    def test_all_pool_is_every_tool(self):
        from kgraphplanner.tools.github import GITHUB_TOOLS
        assert set(select_tool_names("all")) == set(GITHUB_TOOLS)

    def test_read_pool_is_a_strict_subset(self):
        assert set(select_tool_names("read")) < set(select_tool_names("all"))

    def test_write_pool_cannot_change_code(self):
        """The safety property of the pool split. `write` used to mean "every
        tool"; when code-write tools were added, leaving that alone would have
        silently granted commit access to anyone running the same command as
        before. Gaining it must take an explicit --pool code."""
        from kgraphplanner.tools.github import CODE_WRITE_TOOLS
        assert not (set(select_tool_names("write")) & set(CODE_WRITE_TOOLS))

    def test_write_pool_still_contains_issue_writes(self):
        """...without becoming read-only in the process."""
        assert "github_add_issue_comment" in select_tool_names("write")

    def test_code_pool_can_change_code(self):
        from kgraphplanner.tools.github import CODE_WRITE_TOOLS
        assert set(CODE_WRITE_TOOLS) <= set(select_tool_names("code"))

    @pytest.mark.parametrize("pool,expected", [
        ("read", False), ("write", False), ("code", True), ("all", True),
    ])
    def test_pool_code_write_predicate(self, pool, expected):
        from github_eval.agent import pool_has_code_writes
        assert pool_has_code_writes(pool) is expected


class TestCases:

    def test_no_case_carries_an_expected_value(self):
        """Grading is judge-driven consistency, never a fixed count. A Case has
        nowhere to put an expected answer, and that is deliberate."""
        assert not any(hasattr(c, "expected") or hasattr(c, "expected_count")
                       for c in cases_mod.CASES)

    def test_case_names_are_unique(self):
        names = [c.name for c in cases_mod.CASES]
        assert len(names) == len(set(names))

    def test_write_cases_are_flagged(self):
        """--pool read must be able to exclude everything that mutates."""
        for case in cases_mod.CASES:
            if case.fixture:
                assert case.needs_write, case.name

    def test_read_selection_excludes_writes(self):
        assert all(not c.needs_write
                   for c in cases_mod.select(include_writes=False))

    def test_every_expected_tool_exists(self):
        """A typo in expect_tools would silently look like a selection failure."""
        from kgraphplanner.tools.github import GITHUB_TOOLS
        for case in cases_mod.CASES:
            for tool in case.expect_tools:
                assert tool in GITHUB_TOOLS, f"{case.name}: unknown tool {tool}"

    def test_every_placeholder_is_supplied_by_the_cases_own_fixture(self):
        """A case asking for {branch} while requesting the `issue` fixture would
        render with a literal brace and reach the agent as gibberish -- and would
        be scored as though the agent failed. Checked against the fixture the case
        actually declares, not against a union of every key any fixture provides."""
        from string import Formatter
        from github_eval.fixtures import FIXTURE_CONTEXT_KEYS

        for case in cases_mod.CASES:
            placeholders = {
                name for _, name, _, _ in Formatter().parse(case.request) if name
            }
            if not placeholders:
                continue
            assert case.fixture, f"{case.name}: uses {placeholders} but declares no fixture"
            available = FIXTURE_CONTEXT_KEYS.get(case.fixture, set())
            missing = placeholders - available
            assert not missing, (
                f"{case.name}: fixture {case.fixture!r} does not supply {sorted(missing)}"
            )

    def test_every_declared_fixture_can_be_built(self):
        """A typo in `fixture=` would fail at run time as a harness error."""
        from github_eval.fixtures import FixtureFactory, FIXTURE_CONTEXT_KEYS
        for case in cases_mod.CASES:
            if case.fixture:
                assert hasattr(FixtureFactory, f"_build_{case.fixture}"), case.name
                assert case.fixture in FIXTURE_CONTEXT_KEYS, case.name

    def test_code_cases_are_flagged(self):
        """Cases needing code writes must say so, or --pool write would run them
        without the tools and score a harness gap as an agent failure."""
        code_tool_names = set()
        from kgraphplanner.tools.github import CODE_WRITE_TOOLS
        code_tool_names |= set(CODE_WRITE_TOOLS)
        for case in cases_mod.CASES:
            if set(case.expect_tools) & code_tool_names:
                assert case.needs_code, f"{case.name} expects a code tool but is not flagged"

    def test_selection_can_exclude_code_cases(self):
        assert all(not c.needs_code
                   for c in cases_mod.select(include_code=False))

    def test_groups_are_covered(self):
        assert {"read", "selection", "pagination", "error", "write", "multi",
                "code"} <= set(cases_mod.GROUPS)


class TestReporting:

    def test_expected_tool_hit_is_reported_not_asserted(self):
        result = CaseResult(name="x", group="g", request="r", verdict="correct", reason="",
                            tools_called=["github_search_issues"],
                            tools_expected=["github_list_issues"])
        # Wrong tool, but the verdict is what decides pass/fail.
        assert result.used_expected_tool is False
        assert result.passed

    def test_no_expectation_yields_none_rather_than_false(self):
        assert CaseResult(name="x", group="g", request="r",
                          verdict="correct", reason="").used_expected_tool is None

    @pytest.mark.parametrize("verdict,expected", [
        ("correct", True), ("incorrect", False),
        ("no_tool_call", False), ("tool_error", False),
    ])
    def test_only_correct_passes(self, verdict, expected):
        assert CaseResult(name="x", group="g", request="r",
                          verdict=verdict, reason="").passed is expected
