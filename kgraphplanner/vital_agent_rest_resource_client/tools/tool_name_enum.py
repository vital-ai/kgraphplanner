"""
Tool name enumeration for available tools.
"""

from enum import Enum


class ToolName(str, Enum):
    """Available tool names"""
    google_address_validation_tool = "google_address_validation_tool"
    google_web_search_tool = "google_web_search_tool"
    place_search_tool = "place_search_tool"
    serper_web_search_tool = "serper_web_search_tool"
    weather_tool = "weather_tool"

    # The GitHub tools are multi-operation: one service tool name serves many
    # operations, discriminated by the `operation` field on the input model. The
    # agent-facing LangChain tools are finer-grained than this enum -- see
    # planning/kg_tools/github_tools_plan.md section 2.
    github_issue_tool = "github_issue_tool"
    github_pr_tool = "github_pr_tool"
    github_actions_tool = "github_actions_tool"
    github_code_tool = "github_code_tool"
    github_repo_tool = "github_repo_tool"
