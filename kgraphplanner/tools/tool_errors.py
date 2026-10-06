"""
Shared failure-reporting contract for agent-facing tools.

The problem this exists to prevent: a tool that catches an exception and returns
an empty result -- `""`, `{"results": []}`, `PlaceSearchOutput(results=[])` -- is
indistinguishable from a tool that ran fine and genuinely found nothing. The
`ToolMessage` is stamped `status="success"` either way, so nothing reading a run
can tell an outage from a real negative answer without matching on message text.
In production that turns a stopped tool server into "this business has no
results", confidently, for hours.

The contract:

  - A tool that could not complete raises `ToolException` via `fail()`.
  - Every such tool is wrapped in `reports_failures()`, which sets
    `handle_tool_error = True`. langchain-core then puts the exception message in
    the `ToolMessage` -- so the model sees the same text it saw before -- and sets
    `status="error"`, which callers can check without parsing anything.
  - A genuine zero-result stays a normal, successful return. An unknown business
    really does have no results, and that must not be reported as a failure.

`handle_tool_error` cannot be passed to the `@tool` decorator (it is not one of
its parameters); it has to be set on the decorated object, which is what
`reports_failures` is for.
"""
import json
from typing import Any, NoReturn, Optional

from langchain_core.tools import ToolException


def fail(message: str, *, status: Optional[int] = None) -> NoReturn:
    """Report a tool failure.

    Raises `ToolException`, not a bare `Exception`: the LangGraph tool node
    re-raises anything else, which aborts the whole run instead of handing the
    failure back to the model.
    """
    raise ToolException(message if status is None else f"{message} (status_code={status})")


def check_api_error(output: Any) -> None:
    """Fail if a service output carries an `api_error`.

    The tool service reports *expected* failures as HTTP 200 with `success=true`
    and `api_error` set, writing those messages to be actionable -- see
    `tools/github/github_service_tool.py`, which has always honoured this. The
    search tools parsed `api_error` onto their output models and then ignored it,
    so an upstream provider outage arrived as a successful search with zero
    results. Nothing raises in that path, so this check is the only thing that
    catches it.
    """
    api_error = getattr(output, "api_error", None)
    if api_error:
        fail(api_error, status=getattr(output, "api_status_code", None))


def reports_failures(tool):
    """Mark a tool so `ToolException` becomes `ToolMessage(status="error")`.

    Without this, `handle_tool_error` is unset and langchain-core re-raises the
    `ToolException` instead of reporting it, which aborts the run.
    """
    tool.handle_tool_error = True
    return tool


def fail_if_error_payload(result: str) -> None:
    """Re-report a JSON `{"error": ...}` payload as a `ToolException`.

    Some tools report failures by *returning* an error object -- the github family
    does, and that shape is its documented internal contract, relied on by callers
    and asserted in tests. Returning it is still `status="success"` at the agent
    boundary, so this is applied there instead: the model sees the same message
    verbatim, and the `ToolMessage` gains a status callers can check. The internal
    contract is untouched.
    """
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return
    if isinstance(payload, dict) and payload.get("error"):
        fail(payload["error"])
