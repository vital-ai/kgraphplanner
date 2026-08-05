"""
Shared base for every github_* agent-facing tool.

Every existing category-1 tool in this repo repeats the same ~30 lines of client
plumbing. With one tool per service operation there will eventually be ~33 of
them, so that plumbing lives here once, along with the two contracts that are
easy to get wrong and expensive to get wrong in 33 places:

  - the error contract (section 4 of the plan): the service returns expected
    failures as HTTP 200 with `api_error` set, and those messages are written to
    be actionable. They are surfaced to the model verbatim.
  - repo scoping (section 13): the agent-facing schema takes a single `repo` in
    `owner/name` form, enumerated from this instance's configured subset, so the
    model cannot name a repo that would be rejected.

See planning/kg_tools/github_tools_plan.md.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field, create_model
from langchain_core.tools import tool

from kgraphplanner.tool_manager.tool_inf import AbstractTool
from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
    VitalAgentRestResourceClient
)

logger = logging.getLogger(__name__)

_REPO_RE = re.compile(r'^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$')


class GitHubToolConfigError(Exception):
    """The tool is misconfigured and cannot be registered."""


class GitHubServiceTool(AbstractTool):
    """Base for one agent-facing tool wrapping one service operation.

    Subclasses set SERVICE_TOOL, OPERATION, TOOL_NAME, DESCRIPTION, INPUT_MODEL
    and AGENT_FIELDS, then implement ``project()``.
    """

    # --- subclass contract ---------------------------------------------------

    SERVICE_TOOL: str = ""      # "github_issue_tool" | "github_pr_tool" | "github_actions_tool"
    OPERATION: str = ""         # the service operation, e.g. "list_issues"
    TOOL_NAME: str = ""         # agent-facing name, e.g. "github_list_issues"
    DESCRIPTION: str = ""
    INPUT_MODEL: Optional[Type[BaseModel]] = None   # the wire model to build

    # True if the operation changes state on GitHub. The real gates are
    # server-side (allow_writes / allow_pr_merge / allow_workflow_dispatch) and
    # the agent-side lever is which tools get registered -- this flag exists so
    # a read-only set can be selected programmatically, e.g. READ_ONLY_TOOLS,
    # rather than by maintaining a second hand-written list that drifts.
    MUTATING: bool = False

    # False for the one operation that is about the *token* rather than a
    # repository (get_authenticated_user). Such a tool takes no `repo` argument
    # at all: offering one the operation ignores would invite the model to supply
    # a value and read meaning into it, and would imply an allowlist check that
    # does not apply. The service draws the same line by not building that input
    # on its repo base.
    REQUIRES_REPO: bool = True

    # Agent-facing fields other than `repo`, as {name: (type, Field(...))}.
    # Deliberately a subset of the wire model: parameters the model has no
    # business setting (pagination internals, filters nothing asks for) are
    # omitted here and defaulted in build_wire_input().
    AGENT_FIELDS: Dict[str, tuple] = {}

    def project(self, output: Any) -> Dict[str, Any]:
        """Reduce the service output to what this operation should return.

        The shared output model carries every field any operation of that
        service tool might set, so returning it whole is mostly nulls.
        """
        raise NotImplementedError

    def build_wire_input(self, repo: str, **kwargs) -> BaseModel:
        """Build the wire model. Override when defaults need filling in."""
        owner, name = self.split_repo(repo)
        return self.INPUT_MODEL(operation=self.OPERATION, owner=owner, repo=name, **kwargs)

    # --- construction --------------------------------------------------------

    def __init__(self, config: Dict[str, Any], tool_manager: Optional[Any] = None):
        self.repos: List[str] = self._parse_repos(config.get("repos"))
        super().__init__(
            config=config,
            tool_manager=tool_manager,
            name=self.TOOL_NAME,
            description=self._build_description(),
        )

    @staticmethod
    def _parse_repos(raw: Any) -> List[str]:
        """Validate the configured subset. Fails closed and fails loudly.

        An empty list means the tool can reach nothing, which is a configuration
        error rather than a tool that quietly refuses every call -- catching it
        at registration is far cheaper than at the first agent turn.
        """
        if not raw:
            raise GitHubToolConfigError(
                "No repos configured. A GitHub tool needs a `repos` list in its config "
                "block naming the subset of the service allowlist it may use."
            )
        repos = [raw] if isinstance(raw, str) else list(raw)
        bad = [r for r in repos if not _REPO_RE.match(str(r).strip())]
        if bad:
            raise GitHubToolConfigError(
                f"Malformed repos entries {bad}: expected 'owner/name'."
            )
        # Order is preserved: repos[0] is the default for a one-entry list and
        # the order the model sees in the enumeration.
        seen, ordered = set(), []
        for r in repos:
            r = str(r).strip()
            if r not in seen:
                seen.add(r)
                ordered.append(r)
        return ordered

    def _build_description(self) -> str:
        if not self.REQUIRES_REPO:
            return self.DESCRIPTION
        if len(self.repos) == 1:
            return f"{self.DESCRIPTION} Operates on the {self.repos[0]} repository."
        return f"{self.DESCRIPTION} Repositories available: {', '.join(self.repos)}."

    @staticmethod
    def split_repo(repo: str) -> Tuple[str, str]:
        owner, _, name = str(repo).partition("/")
        return owner, name

    # --- schema --------------------------------------------------------------

    def get_tool_schema(self) -> Type[BaseModel]:
        """Built per instance, so `repo` enumerates *this* agent's subset.

        A plain string field validated after the fact would let the model emit a
        repo that is then rejected; a Literal means the only values it can
        produce are the allowed ones.
        """
        from typing import Literal as _Literal

        if not self.REQUIRES_REPO:
            return create_model(f"{self.TOOL_NAME}_input", **self.AGENT_FIELDS)

        repo_type = _Literal[tuple(self.repos)]  # type: ignore[valid-type]
        if len(self.repos) == 1:
            repo_field = (Optional[repo_type], Field(
                self.repos[0],
                description=f"Repository to act on. Only {self.repos[0]} is available, "
                            f"so this may be omitted.",
            ))
        else:
            repo_field = (repo_type, Field(
                ...,
                description="Repository to act on, as owner/name. Must be one of: "
                            + ", ".join(self.repos),
            ))

        return create_model(
            f"{self.TOOL_NAME}_input",
            repo=repo_field,
            **self.AGENT_FIELDS,
        )

    # --- execution -----------------------------------------------------------

    def get_tool_function(self) -> Callable:
        schema = self.get_tool_schema()
        tool_self = self

        # The description must be passed here rather than set afterwards: the
        # decorator reads it at decoration time and rejects a function with
        # neither. It is also what the model actually sees, so it carries the
        # repo scope and the pagination rules from the subclass DESCRIPTION.
        @tool(tool_self.TOOL_NAME, description=self.description, args_schema=schema)
        async def _run(**kwargs) -> str:
            return await tool_self.run(**kwargs)

        return _run

    async def run(self, **kwargs) -> str:
        if not self.REQUIRES_REPO:
            kwargs.pop("repo", None)
            return await self._dispatch(None, **kwargs)

        repo = kwargs.pop("repo", None) or (self.repos[0] if len(self.repos) == 1 else None)
        if not repo:
            return self._error(f"`repo` is required. Available: {', '.join(self.repos)}.")

        # Client-side, pre-network. Mirrors the service's own check so the agent
        # sees one consistent shape whichever layer rejected it.
        if repo not in self.repos:
            return self._error(
                f"Repository '{repo}' is not available to this tool. "
                f"Available: {', '.join(self.repos)}."
            )

        return await self._dispatch(repo, **kwargs)

    async def _dispatch(self, repo: Optional[str], **kwargs) -> str:
        try:
            wire_input = self.build_wire_input(repo, **kwargs)
        except Exception as e:
            return self._error(f"Invalid arguments for {self.TOOL_NAME}: {e}")

        output, error = await self.call_service(wire_input)
        if error:
            return self._error(error)

        api_error = getattr(output, "api_error", None)
        if api_error:
            # An expected failure. The service wrote this to be actionable, so it
            # reaches the model unchanged -- flattening it to "no results" is the
            # single most wasteful thing this layer could do.
            status = getattr(output, "api_status_code", None)
            return self._error(api_error, status)

        return json.dumps(self.project(output), default=str)

    async def call_service(self, wire_input: BaseModel) -> Tuple[Any, Optional[str]]:
        """Post to the service. Returns (output, error_text); one is always None."""
        endpoint = self.config.get("tool_endpoint")
        if not endpoint:
            return None, "The tool service endpoint is not configured."

        jwt_token = self.tool_manager.get_jwt_token() if self.tool_manager else None
        client = VitalAgentRestResourceClient({"tool_endpoint": endpoint}, jwt_token)

        try:
            response = await client.handle_tool_request(self.SERVICE_TOOL, wire_input)
        except PermissionError as e:
            # The client raises rather than returning on 401/403. Uncaught, this
            # ends the agent turn instead of becoming something it can report.
            logger.warning(f"{self.TOOL_NAME}: auth failure calling the tool service: {e}")
            return None, ("Not authorised to call the tool service -- the session token is "
                          "missing or expired. This is an infrastructure problem, not a "
                          "GitHub one; retrying will not help.")
        except Exception as e:
            logger.error(f"{self.TOOL_NAME}: tool service call failed: {e}")
            return None, f"The tool service could not be reached: {e}"

        if response is None or not response.success:
            message = getattr(response, "error_message", None) or "no response"
            return None, f"The tool service reported a failure: {message}"

        if response.tool_output is None:
            return None, "The tool service returned an empty response."

        return response.tool_output, None

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _error(message: str, status: Optional[int] = None) -> str:
        payload: Dict[str, Any] = {"error": message}
        if status is not None:
            payload["status_code"] = status
        return json.dumps(payload)

    @staticmethod
    def paging(output: Any) -> Dict[str, Any]:
        """The pagination contract, carried through to the model.

        Dropping these would make six rounds of upstream work invisible to
        agents, which would then report partial results as complete.
        """
        return {
            "returned_count": getattr(output, "returned_count", None),
            "truncated": getattr(output, "truncated", False),
            "next_page": getattr(output, "next_page", None),
        }
