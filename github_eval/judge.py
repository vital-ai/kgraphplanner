"""
LLM judge: given a transcript, return a verdict.

Never touches GitHub or the tool service. That boundary is deliberate -- it means
the judge can be exercised offline against recorded transcripts, which is the only
part of this harness that can be tested without live dependencies.

Grading is always consistency between what the tools returned and what the agent
said, never agreement with a fixed expected value. See plan section 14.4.2 for why:
a hardcoded count tests GitHub's state rather than the agent, fails for reasons
that are not defects, and is weaker even when it passes -- an agent that ignored
the tool and guessed the right number would sail through it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROMPT_PATH = Path(__file__).parent / "judge_prompt.md"

VERDICTS = ("correct", "incorrect", "no_tool_call", "tool_error")

# Tool results can be large -- a log tail or a 100-issue list. The judge needs to
# see enough to check consistency, not the whole payload.
MAX_RESULT_CHARS = 4000


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    result: Optional[str] = None


@dataclass
class Transcript:
    """Everything the judge is allowed to see."""
    request: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    reply: str = ""

    def to_prompt(self) -> str:
        parts = [f"## User request\n\n{self.request}\n"]
        if not self.tool_calls:
            parts.append("## Tool calls\n\n(none -- the agent called no tools)\n")
        else:
            parts.append("## Tool calls\n")
            for i, call in enumerate(self.tool_calls, 1):
                result = call.result or "(no result recorded)"
                if len(result) > MAX_RESULT_CHARS:
                    result = result[:MAX_RESULT_CHARS] + f"\n... [{len(result)} chars total, truncated]"
                parts.append(
                    f"### {i}. {call.name}\n"
                    f"arguments: {json.dumps(call.args, default=str)}\n"
                    f"result:\n```\n{result}\n```\n"
                )
        parts.append(f"## Agent's final reply\n\n{self.reply or '(empty)'}\n")
        return "\n".join(parts)


@dataclass
class Verdict:
    verdict: str
    reason: str
    raw: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "correct"


def parse_verdict(text: str) -> Verdict:
    """Pull a verdict out of the judge's reply.

    Tolerant of fenced JSON and of prose around it, because a judge that wraps
    its answer in ```json should not be scored as a harness failure.
    """
    raw = text or ""
    candidate = raw.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S)
    if fenced:
        candidate = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", candidate, re.S)
        if brace:
            candidate = brace.group(0)

    try:
        data = json.loads(candidate)
        verdict = str(data.get("verdict", "")).strip().lower()
        reason = str(data.get("reason", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return Verdict("unparseable", f"Judge did not return JSON: {raw[:200]}", raw)

    if verdict not in VERDICTS:
        return Verdict("unparseable", f"Unknown verdict {verdict!r}", raw)

    return Verdict(verdict, reason, raw)


class Judge:
    """Wraps a chat model. Construction is separate from use so the model is
    built once per run rather than once per case."""

    def __init__(self, model: str = "openai:gpt-5.6-sol", prompt_path: Path = PROMPT_PATH):
        self.model_name = model
        self.system_prompt = prompt_path.read_text()
        self._model = None

    def _get_model(self):
        if self._model is None:
            from langchain.chat_models import init_chat_model
            self._model = init_chat_model(self.model_name, temperature=0)
        return self._model

    async def judge(self, transcript: Transcript) -> Verdict:
        from langchain_core.messages import SystemMessage, HumanMessage

        try:
            response = await self._get_model().ainvoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=transcript.to_prompt()),
            ])
        except Exception as e:
            # A judge failure is a harness failure, not a verdict on the agent.
            return Verdict("unparseable", f"Judge call failed: {e}")

        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        return parse_verdict(str(content))
