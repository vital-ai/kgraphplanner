"""
Response handlers for the github_* tools.

One handler per service tool, not per operation: all operations of a service tool
share a single output model, so github_issue_tool's fifteen operations parse
through the same path. The parsing itself is identical across the three, so it
lives in a base and each subclass supplies only its output model.

The service returns expected failures -- allowlist denial, write gate off, 404,
rate limiting -- as HTTP 200 with `api_error` set on the output. That is not a
transport failure and must not be flattened into a generic message: those strings
are written to be actionable. See planning/kg_tools/github_tools_plan.md section 4.
"""

import logging
from typing import Dict, Type

from pydantic import BaseModel

from kgraphplanner.vital_agent_rest_resource_client.tools.tool_handler import ToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.github.issue_models import (
    GitHubIssueToolOutput
)
from kgraphplanner.vital_agent_rest_resource_client.tools.github.pr_models import (
    GitHubPRToolOutput
)
from kgraphplanner.vital_agent_rest_resource_client.tools.github.actions_models import (
    GitHubActionsToolOutput
)
from kgraphplanner.vital_agent_rest_resource_client.tools.github.code_models import (
    GitHubCodeToolOutput
)
from kgraphplanner.vital_agent_rest_resource_client.tools.github.repo_models import (
    GitHubRepoToolOutput
)

logger = logging.getLogger(__name__)


class GitHubToolHandler(ToolHandler):
    """Shared parsing for every github_* service tool."""

    OUTPUT_MODEL: Type[BaseModel] = None
    TOOL_NAME: str = ""

    def handle_response(self, tool_parameters: ToolParameters, response_json: Dict):
        # The service wraps the tool payload; older shapes return it bare.
        tool_output = response_json.get('tool_output', response_json) or {}
        operation = str(tool_output.get('operation') or '')

        if not response_json.get('success', True):
            # The tool itself raised server-side. Distinct from api_error, which
            # is an expected GitHub/config failure the agent can act on.
            message = response_json.get('error_message') or 'tool execution failed'
            logger.warning(f"{self.TOOL_NAME} reported failure: {message}")
            return self.OUTPUT_MODEL(
                operation=operation,
                api_error=f"Tool execution failed on the service: {message}",
            )

        try:
            return self.OUTPUT_MODEL(**tool_output)
        except Exception as e:
            # A shape we do not recognise. Surface it rather than returning an
            # empty result that reads as "nothing found".
            logger.error(f"Could not parse {self.TOOL_NAME} output: {e}")
            return self.OUTPUT_MODEL(
                operation=operation,
                api_error=f"Unrecognised response from the tool service: {e}",
            )


class GitHubIssueToolHandler(GitHubToolHandler):
    OUTPUT_MODEL = GitHubIssueToolOutput
    TOOL_NAME = "github_issue_tool"


class GitHubPRToolHandler(GitHubToolHandler):
    OUTPUT_MODEL = GitHubPRToolOutput
    TOOL_NAME = "github_pr_tool"


class GitHubActionsToolHandler(GitHubToolHandler):
    OUTPUT_MODEL = GitHubActionsToolOutput
    TOOL_NAME = "github_actions_tool"


class GitHubCodeToolHandler(GitHubToolHandler):
    OUTPUT_MODEL = GitHubCodeToolOutput
    TOOL_NAME = "github_code_tool"


class GitHubRepoToolHandler(GitHubToolHandler):
    OUTPUT_MODEL = GitHubRepoToolOutput
    TOOL_NAME = "github_repo_tool"


HANDLERS = {
    "github_issue_tool": GitHubIssueToolHandler,
    "github_pr_tool": GitHubPRToolHandler,
    "github_actions_tool": GitHubActionsToolHandler,
    "github_code_tool": GitHubCodeToolHandler,
    "github_repo_tool": GitHubRepoToolHandler,
}
