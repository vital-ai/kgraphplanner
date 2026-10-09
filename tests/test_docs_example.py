"""
The example in docs/exec_graph.md must run as written.

It is executed with its own testing snippet (ScriptedLLM), so the doc's graph,
its redraft-with-revision-notes loop and its node order are checked by the
suite rather than by hand.
"""
import re
import textwrap
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "exec_graph.md"


def _blocks():
    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(), re.S)
    example = blocks[0]
    testing = next(b for b in blocks if "ScriptedLLM(script=" in b)
    return example, testing


async def test_docs_example_runs_as_written():
    example, testing = _blocks()
    setup, run_and_assert = testing.split("# ... build workers")[0], testing.split("result = await agent.run")[1]
    # The example's own run() is replaced by the testing snippet's.
    build = example.split("result = await agent.run")[0]
    source = (
        "async def main():\n"
        + textwrap.indent(setup, "    ")
        + "    CANNED_REPLY = 'canned'\n"
        + "    case_data = {'id': 'c1'}\n"
        + "    def check(draft):\n"
        + "        return ['too vague'] if 'first' in draft else []\n"
        + textwrap.indent(build, "    ")
        + textwrap.indent("result = await agent.run" + run_and_assert, "    ")
        + "    return result, llm\n"
    )
    namespace: dict = {}
    exec(compile(source, str(DOC), "exec"), namespace)
    result, llm = await namespace["main"]()

    assert result.errors == {}
    assert result.results["format"]["data"] == {"headline": "Approved", "body": "..."}
    # The rate-limited first attempt was retried and counted.
    assert result.usage["outcomes"] == {"ok": 4, "llm_error": 1}
    # The redraft got the validator's problems and the previous draft.
    redraft = next(m.content for m in llm.calls_for("content")[-1]["messages"] if m.type == "human")
    assert "revision_notes:" in redraft and "too vague" in redraft
    assert "prior_draft: first draft" in redraft
