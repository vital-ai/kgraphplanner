"""
Autonomous Tools Agent — Deep Agent with external tools only (no sandbox).

A lightweight tool-calling agent built with LangChain Deep Agents
(``create_deep_agent``) that uses **only** KGraphPlanner-managed tools
(web search, weather, place search, etc.) without any sandbox, filesystem,
or shell capabilities.

This is the simplest Deep Agent configuration: an LLM with tools and
good-practice instructions.  It is ideal for:

  - **Q&A assistants** that answer factual questions via search/weather/place
    tools without needing file editing or code execution.
  - **Lightweight integrations** where Docker/AgentBox infrastructure is
    not available or not needed.
  - **Embedding as a subagent** inside a larger pipeline (Deep Agent or
    KGraphPlanner graph) for tool-calling subtasks.

What this agent does NOT have:

  - No sandbox filesystem (no ``edit``, ``ls``, ``glob``, ``grep``).
  - No shell execution (no ``execute``, no ``git``).
  - No ``write_todos`` (no backend to persist them).
  - No ``reportgen`` (no sandbox to run it in).
  - No skills or memory file loading (no backend filesystem).

Instead, the AGENTS.md best-practice guidelines are loaded from the
bundled ``memory/`` directory and prepended to the system prompt at
build time.

Flow::

    user message ──► Deep Agent (LLM + tool loop) ──► response
                         │
                         └── kgraph tools: web search, weather, place search

Prerequisites:
    - OPENAI_API_KEY (or other provider key) in environment
    - Tool server running for KGraphPlanner tools (optional — depends on
      tool configuration)

Usage::

    from kgraphplanner.sample.auto_tools import build_agent

    agent = build_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
    })
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from deepagents import create_deep_agent

from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.tool_name_enum import (
    ToolName as ToolNameEnum,
)

logger = logging.getLogger(__name__)

# ============================================================
# Paths to bundled memory files
# ============================================================

_PACKAGE_DIR = Path(__file__).resolve().parent

MEMORY_DIR = _PACKAGE_DIR / "memory"

# Default system prompt — tool-focused, no sandbox/file/shell references
DEFAULT_SYSTEM_PROMPT = """\
You are a research and conversational assistant with access to external
tools.  You answer questions by calling tools and synthesizing results
into clear, well-sourced responses.

# Available Tools

- **google_web_search_tool** — Search the web.  Returns titles, URLs,
  and snippets.  Use specific, targeted queries for best results.
- **weather_tool** — Get current weather conditions for a location.
- **place_search_tool** — Look up a place, restaurant, or business.
  Returns name, address, coordinates, ratings, and other details.

# Tool Usage Best Practices

1. **Batch calls:** If you need multiple independent pieces of
   information (e.g. weather in three cities), call the tools in
   parallel rather than sequentially.
2. **Be specific:** Use precise search queries.  Prefer
   "best Italian restaurants in downtown Chicago" over "food Chicago".
3. **Verify with tools:** Do not guess at factual data like weather,
   addresses, or business hours.  Always call the appropriate tool.
4. **Cite sources:** When presenting web search results, include the
   source URL so the user can verify.
5. **Combine results:** When multiple tool results are relevant,
   synthesize them into a single coherent answer rather than dumping
   raw output.
6. **Handle failures gracefully:** If a tool returns no useful results,
   say so clearly and suggest alternatives or rephrase the query.

# Response Guidelines

- Lead with the answer, then provide supporting details.
- Use structured formatting (headings, bullet lists, tables) when
  presenting complex or multi-part results.
- Be concise and direct.  Avoid filler.
- For multi-part questions, address each part clearly and separately.
- If you are unsure about something, say so rather than fabricating
  information.
"""


# ============================================================
# Helpers
# ============================================================

def _load_memory_content() -> str:
    """Load the AGENTS.md file and return its contents as a string.

    This is used to prepend persistent agent context to the system
    prompt, since there is no backend filesystem to load it from at
    runtime.
    """
    agents_md = MEMORY_DIR / "AGENTS.md"
    if agents_md.is_file():
        return agents_md.read_text(encoding="utf-8")
    return ""


def get_kgraph_tools(
    tool_manager: ToolManager,
    tool_ids: Optional[List[str]] = None,
) -> List:
    """Extract LangChain-compatible tool functions from a ToolManager.

    Args:
        tool_manager: Initialized ToolManager with tools loaded.
        tool_ids: Specific tool IDs to include.  If ``None``, uses the
            default set (web search, weather, place search).

    Returns:
        List of ``@tool``-decorated callables ready for
        ``create_deep_agent(tools=...)``.
    """
    if tool_ids is None:
        tool_ids = [
            ToolNameEnum.google_web_search_tool.value,
            ToolNameEnum.weather_tool.value,
            ToolNameEnum.place_search_tool.value,
        ]

    tool_functions = []
    for tid in tool_ids:
        fn = tool_manager.get_tool_function(tid)
        if fn is not None:
            tool_functions.append(fn)
        else:
            logger.warning(f"Tool '{tid}' not found in ToolManager — skipping")
    return tool_functions


# ============================================================
# Agent builder
# ============================================================

def build_agent(
    *,
    model: str | BaseChatModel = "openai:gpt-5",
    tool_manager: Optional[ToolManager] = None,
    agent_config: Optional[AgentConfig] = None,
    tool_ids: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    include_memory: bool = True,
) -> CompiledStateGraph:
    """Build a tool-only Deep Agent (no sandbox/filesystem/shell).

    Args:
        model: LLM model identifier (e.g. ``"openai:gpt-5"``,
            ``"anthropic:claude-sonnet-4-5-20250929"``) or a pre-configured
            ``BaseChatModel`` instance.
        tool_manager: Pre-configured ToolManager.  If ``None``, one is
            created from *agent_config* (or defaults).
        agent_config: AgentConfig for tool loading.  Ignored if
            *tool_manager* is provided.
        tool_ids: Which KGraphPlanner tools to include.  ``None`` uses
            the default set (web search, weather, place search).
        system_prompt: Override the default system prompt entirely.
        checkpointer: Optional LangGraph checkpointer for conversation
            persistence across invocations.
        include_memory: If ``True`` (default), the bundled
            ``memory/AGENTS.md`` content is appended to the system
            prompt as additional agent context.

    Returns:
        A compiled Deep Agent graph ready for ``.invoke()`` or
        ``.stream()`` calls.
    """
    logger.info("Building auto_tools agent (no backend)")

    # --- KGraphPlanner tools ---
    if tool_manager is None:
        config = agent_config or AgentConfig.from_env()
        tool_manager = ToolManager(config=config)
        tool_manager.load_tools_from_config()

    kgraph_tools = get_kgraph_tools(tool_manager, tool_ids)
    logger.info(f"Loaded {len(kgraph_tools)} KGraphPlanner tools for Deep Agent")

    # --- System prompt ---
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    if include_memory:
        memory_content = _load_memory_content()
        if memory_content:
            prompt = prompt.rstrip() + "\n\n" + memory_content

    # --- Build Deep Agent (no backend) ---
    agent = create_deep_agent(
        model=model,
        tools=kgraph_tools,
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    return agent
