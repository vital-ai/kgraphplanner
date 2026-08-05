"""
Agent-facing GitHub tools.

GITHUB_TOOLS maps agent-facing tool name -> class. ToolManager registers from
this rather than growing a branch per operation: with one tool per service
operation the elif chain would eventually carry ~33 entries, and a registry also
makes it possible to tell an agent that an enabled name is unknown rather than
skipping it silently.
"""

from kgraphplanner.tools.github.github_service_tool import (
    GitHubServiceTool, GitHubToolConfigError
)
from kgraphplanner.tools.github.issue_tools import ISSUE_TOOLS
from kgraphplanner.tools.github.pr_tools import PR_TOOLS
from kgraphplanner.tools.github.actions_tools import ACTIONS_TOOLS
from kgraphplanner.tools.github.repo_tools import REPO_TOOLS
from kgraphplanner.tools.github.code_tools import CODE_TOOLS

GITHUB_TOOLS = {}
GITHUB_TOOLS.update(ISSUE_TOOLS)
GITHUB_TOOLS.update(PR_TOOLS)
GITHUB_TOOLS.update(ACTIONS_TOOLS)
GITHUB_TOOLS.update(REPO_TOOLS)
GITHUB_TOOLS.update(CODE_TOOLS)

# Derived, not hand-maintained: a second literal list would drift from the
# MUTATING flags the moment a tool was added.
READ_ONLY_TOOLS = {n: c for n, c in GITHUB_TOOLS.items() if not c.MUTATING}
WRITE_TOOLS = {n: c for n, c in GITHUB_TOOLS.items() if c.MUTATING}

# A third axis, and the one that actually matters for authority. MUTATING is
# true for adding a label and for replacing a source file alike; they are not
# remotely the same risk. CODE_WRITE_TOOLS is exactly the set that can alter the
# repository's contents, so "this agent may comment on issues but must not touch
# code" is expressible -- which it is not with READ_ONLY_TOOLS alone.
#
# Derived from the service tool name rather than a fourth class flag: the
# service split its tools on precisely this boundary (commit 6543689), so
# deriving from it means the two definitions of "can change code" cannot drift.
CODE_WRITE_TOOLS = {
    n: c for n, c in GITHUB_TOOLS.items()
    if c.SERVICE_TOOL == "github_code_tool"
}
SAFE_TOOLS = {n: c for n, c in GITHUB_TOOLS.items() if n not in CODE_WRITE_TOOLS}

__all__ = [
    "GITHUB_TOOLS", "READ_ONLY_TOOLS", "WRITE_TOOLS",
    "CODE_WRITE_TOOLS", "SAFE_TOOLS",
    "GitHubServiceTool", "GitHubToolConfigError",
]
