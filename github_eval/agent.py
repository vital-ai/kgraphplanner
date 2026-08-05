"""
Agent construction and transcript capture.

Builds a deep agent over a chosen subset of the GitHub tools, runs one request,
and extracts what the judge needs: the tool calls with their arguments, the
results they returned, and the final reply.

The tool subset is a parameter because pool size is one of the things this harness
exists to measure (plan section 2.0.1): the same cases against the full pool and
against a filtered one, comparing selection accuracy.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.tools.github import (
    GITHUB_TOOLS, READ_ONLY_TOOLS, SAFE_TOOLS, CODE_WRITE_TOOLS
)

from github_eval.judge import ToolCall, Transcript

logger = logging.getLogger(__name__)

# Paging through a repository costs one step per page plus the agent's own
# reasoning: 8 pages of a 54-issue repo already exceeded 30. Raised so that a
# "count everything" case fails on its answer rather than on the step budget.
RECURSION_LIMIT = 60
DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = """You are an assistant with access to GitHub tools for a single repository.

Answer questions by calling the tools -- never guess a repository's state, and never
report a number you did not get from a tool.

When a tool result has "truncated": true, more records exist than were returned.
Say so rather than presenting a partial result as complete.

When a tool returns an "error" field, read it. These messages explain what went
wrong and are usually actionable -- a rejected query can be rewritten, a denied
repository is not available to you, a disabled operation will not become enabled
by retrying. Explain the situation to the user instead of pretending the call
succeeded.

If you have tools that change code, treat them as consequential. github_create_or_update_file
replaces the WHOLE file with whatever you pass as content -- it is not a patch -- so read the
current contents first when editing an existing file, or you will silently delete everything
you did not repeat. Prefer creating a branch over committing to the default branch."""


async def get_jwt_token() -> Tuple[Optional[str], Optional[str]]:
    """Keycloak direct grant. The tool service rejects unauthenticated calls."""
    user, password = os.getenv("KEYCLOAK_USER"), os.getenv("KEYCLOAK_PASSWORD")
    realm = os.getenv("KEYCLOAK_REALM")
    if not user or not password or not realm:
        return None, "KEYCLOAK_USER / KEYCLOAK_PASSWORD / KEYCLOAK_REALM not set"

    base = os.getenv("KEYCLOAK_URL", "http://localhost:8085")
    data = {
        "grant_type": "password",
        "client_id": os.getenv("KEYCLOAK_CLIENT_ID"),
        "username": user,
        "password": password,
    }
    if os.getenv("KEYCLOAK_CLIENT_SECRET"):
        data["client_secret"] = os.getenv("KEYCLOAK_CLIENT_SECRET")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base}/realms/{realm}/protocol/openid-connect/token", data=data, timeout=10
            )
        response.raise_for_status()
        return response.json().get("access_token"), None
    except Exception as e:
        return None, f"Keycloak token request failed: {e}"


def select_tool_names(pool: str = "read", extra: Optional[List[str]] = None) -> List[str]:
    """Which tools the agent sees.

    'read'  -- read-only tools; safe to run with no teardown at all
    'write' -- everything EXCEPT the tools that can alter repository contents
    'code'  -- everything, including code writes
    'all'   -- synonym for 'code'

    `write` deliberately excludes code writes rather than meaning "all tools", as
    it used to when no code-write tool existed. Leaving it as "everything" would
    have silently granted an existing `--pool write` invocation the ability to
    commit to the repository the first time those tools were registered, which is
    not a change anyone would have chosen by running the same command as before.

    Gaining code access is therefore an explicit `--pool code`.
    """
    if pool in ("all", "code"):
        names = sorted(GITHUB_TOOLS)
    elif pool == "write":
        names = sorted(SAFE_TOOLS)
    else:
        names = sorted(READ_ONLY_TOOLS)
    for name in extra or []:
        if name not in names and name in GITHUB_TOOLS:
            names.append(name)
    return names


def pool_has_code_writes(pool: str) -> bool:
    return bool(set(select_tool_names(pool)) & set(CODE_WRITE_TOOLS))


async def build(repo: str, endpoint: str, tool_names: List[str], model: str):
    """Returns (agent, tool_manager, tool_functions_by_name)."""
    from kgraphplanner.sample.auto_tools import build_agent

    config = AgentConfig.from_dict({"tools": {
        "endpoint": endpoint,
        "enabled": tool_names,
        "tool_configs": {name: {"repos": [repo]} for name in tool_names},
    }})
    manager = ToolManager(config=config)
    manager.load_tools_from_config()

    token, error = await get_jwt_token()
    if error:
        logger.warning(f"no JWT: {error}")
    else:
        manager.set_jwt_token(token)

    registered = sorted(manager.get_tool_names())
    agent = build_agent(model=model, tool_manager=manager, tool_ids=registered,
                        system_prompt=SYSTEM_PROMPT, include_memory=False)
    functions = {t.name: t for t in manager.get_enabled_tool_functions()}
    return agent, manager, functions


def extract_transcript(request: str, result: Dict[str, Any]) -> Transcript:
    """Pull tool calls, their results, and the final reply out of a run.

    Tool results arrive as separate ToolMessages keyed by tool_call_id, so the
    call and its result have to be stitched back together -- the judge needs both
    to tell "the tool failed" from "the agent misread a good response".
    """
    messages = result.get("messages", [])

    results_by_id: Dict[str, str] = {}
    for message in messages:
        if getattr(message, "type", "") == "tool":
            call_id = getattr(message, "tool_call_id", None)
            if call_id:
                results_by_id[call_id] = str(getattr(message, "content", ""))

    calls: List[ToolCall] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            calls.append(ToolCall(name=name, args=args or {},
                                  result=results_by_id.get(call_id)))

    reply = ""
    for message in reversed(messages):
        if getattr(message, "type", "") != "ai":
            continue
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        if str(content).strip():
            reply = str(content).strip()
            break

    return Transcript(request=request, tool_calls=calls, reply=reply)


async def run_case(agent, request: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[Transcript, Optional[str]]:
    """Run one request. Returns (transcript, error). Never raises."""
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": request}]},
                          config={"recursion_limit": RECURSION_LIMIT}),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return Transcript(request=request), f"timed out after {timeout}s"
    except Exception as e:
        return Transcript(request=request), f"{type(e).__name__}: {e}"
    return extract_transcript(request, result), None
