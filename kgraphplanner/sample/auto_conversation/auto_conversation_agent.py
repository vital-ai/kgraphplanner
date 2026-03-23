"""
Autonomous Conversation Agent — Deep Agent with persistent git-backed sandbox.

A conversational agent designed for **multi-turn, ongoing conversations**
where knowledge accumulates across turns.  Built with LangChain Deep Agents
(``create_deep_agent``) and an ``AgentBoxSandbox`` git backend that persists
between turns.

Key differences from ``auto_general``:

  - **Knowledge persistence** — the agent actively maintains a knowledge base
    in the sandbox (``/workspace/knowledge/``) so facts learned in earlier
    turns are available in later turns, even after context summarization.
  - **Conversation continuity** — a single thread_id and sandbox (repo_id)
    are reused across all turns in a conversation.
  - **Git snapshots** — after each turn the agent commits accumulated
    knowledge so it can be recovered or inspected later.
  - **Reconnectable** — the caller supplies a ``repo_id`` and can reconnect
    to the same sandbox in a later session to continue the conversation.

Flow::

    turn 1 ──► Deep Agent ──► response + git commit (knowledge saved)
    turn 2 ──► Deep Agent ──► response + git commit (knowledge updated)
    ...
    turn N ──► Deep Agent ──► response + git commit

Prerequisites:
    - AgentBox orchestrator running (docker compose up --build -d)
    - OPENAI_API_KEY (or other provider key) in environment
    - Tool server running for KGraphPlanner tools (optional)

Usage::

    from kgraphplanner.sample.auto_conversation import build_agent

    repo_id = f"conversation-{uuid4().hex[:8]}"
    agent, backend = build_agent(
        orchestrator_url="http://localhost:8090",
        repo_id=repo_id,
    )
    # Turn 1
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Tell me about Tokyo."}]},
        config={"configurable": {"thread_id": repo_id}},
    )
    # Turn 2 — agent remembers Tokyo from turn 1
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather there?"}]},
        config={"configurable": {"thread_id": repo_id}},
    )
    backend.destroy()
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from deepagents import create_deep_agent
from agentbox.deepagents import AgentBoxSandbox

from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.tool_name_enum import (
    ToolName as ToolNameEnum,
)

logger = logging.getLogger(__name__)

# ============================================================
# Paths to bundled skill and memory files
# ============================================================

_PACKAGE_DIR = Path(__file__).resolve().parent

SKILLS_DIR = _PACKAGE_DIR / "skills"
MEMORY_DIR = _PACKAGE_DIR / "memory"

# Sandbox paths where skills/memory will be uploaded
_SANDBOX_SKILLS_ROOT = "/.deepagents/skills/"
_SANDBOX_MEMORY_ROOT = "/.deepagents/memory/"

# Default system prompt prepended before Deep Agents' base prompt
DEFAULT_SYSTEM_PROMPT = """\
You are a conversational assistant engaged in an ongoing, multi-turn
conversation.  You remember context from earlier turns and build on it.

# Sandbox Environment

You operate inside a secure sandboxed workspace at /workspace with
git-backed persistence.

- **CWD:** /workspace
- **Dirs:** /workspace, /data, /var, /etc, /tmp
- **No network:** curl/wget are blocked — use the provided tools
  (web search, weather, place search) instead of direct HTTP requests.

## File Editing
Use `edit` for precise, targeted file modifications:
- Replace: `edit <file> --old "exact old text" --new "replacement text"`
- View: `edit <file> --view` or `edit <file> --view --range 10:25`
- Insert: `edit <file> --insert <line> --text "new text"`
- Create: `edit <file> --create --content "file contents"`

For writing new files with lots of content, use heredoc:
```
cat > /workspace/file.md << 'EOF'
content here
EOF
```

## Reports (Markdown → PDF)
```
reportgen input.md -o output.pdf --title "Title" --author "Author" --toc
```

## Git
Pre-configured at /workspace.  No setup needed.
```
git add . && git commit -m "message"
git push
```
**IMPORTANT:** Do NOT use `git add -A` — the sandbox does not support
the -A flag.  Use `git add .` or `git add <specific-paths>` instead.

## Shell
Standard Linux commands: ls, cat, cp, mv, rm, mkdir, find, grep, sed,
awk, sort, head, tail, wc, tar, zip, diff, cut, tr, base64, date, seq.

# Knowledge Persistence

**After every turn** you MUST:
  1. Save any new facts, decisions, or research results to files under
     /workspace/knowledge/ so they survive context summarization.
  2. Run: git add . && git commit -m "<brief summary of this turn>" && git push

Use the knowledge directory as your long-term memory:
  - /workspace/knowledge/facts.md    — key facts learned during conversation
  - /workspace/knowledge/topics/     — one file per major topic explored
  - /workspace/knowledge/history.md  — chronological log of conversation turns

When the user asks a follow-up question, ALWAYS check /workspace/knowledge/
first to see if you already have relevant information before calling tools.

# Tools

You have access to web search, weather, and place-search tools in
addition to the built-in file and shell tools.  Use them to answer the
user's questions accurately.  When performing multi-step research, use
write_todos to plan and track your progress.
"""


# ============================================================
# Helpers
# ============================================================

def _collect_files_from_dir(local_dir: Path, sandbox_prefix: str) -> List[Tuple[str, bytes]]:
    """Walk a local directory and return (sandbox_path, content) tuples
    suitable for ``AgentBoxSandbox.upload_files``."""
    files: List[Tuple[str, bytes]] = []
    if not local_dir.is_dir():
        return files
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(local_dir)
            sandbox_path = sandbox_prefix + str(rel)
            files.append((sandbox_path, path.read_bytes()))
    return files


def get_kgraph_tools(
    tool_manager: ToolManager,
    tool_ids: Optional[List[str]] = None,
) -> List:
    """Extract LangChain-compatible tool functions from a ToolManager.

    Args:
        tool_manager: Initialized ToolManager with tools loaded.
        tool_ids: Specific tool IDs to include.  If ``None``, uses the
            default conversation set (web search, weather, place search).

    Returns:
        List of ``@tool``-decorated callables ready for ``create_deep_agent(tools=...)``.
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


def upload_skills_and_memory(backend: AgentBoxSandbox) -> None:
    """Upload the bundled SKILL.md and AGENTS.md files into the sandbox."""
    files: List[Tuple[str, bytes]] = []
    files.extend(_collect_files_from_dir(SKILLS_DIR, _SANDBOX_SKILLS_ROOT))
    files.extend(_collect_files_from_dir(MEMORY_DIR, _SANDBOX_MEMORY_ROOT))

    if files:
        logger.info(f"Uploading {len(files)} skill/memory files to sandbox")
        backend.upload_files(files)


# ============================================================
# Agent builder
# ============================================================

def build_agent(
    *,
    orchestrator_url: str = "http://localhost:8090",
    repo_id: Optional[str] = None,
    model: str | BaseChatModel = "openai:gpt-5",
    tool_manager: Optional[ToolManager] = None,
    agent_config: Optional[AgentConfig] = None,
    tool_ids: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    log_file: Optional[str] = None,
    sandbox_init_wait: float = 2.0,
) -> Tuple[CompiledStateGraph, AgentBoxSandbox, str]:
    """Build a ready-to-invoke Deep Agent with a git-backed sandbox.

    Args:
        orchestrator_url: URL of the AgentBox orchestrator service.
        repo_id: Git repo identifier for persistence.  Auto-generated
            if not provided.  **Save this value** to reconnect later.
        model: LLM model identifier or instance.
        tool_manager: Pre-configured ToolManager.  If ``None``, one is
            created from *agent_config* (or defaults).
        agent_config: AgentConfig for tool loading.  Ignored if
            *tool_manager* is provided.
        tool_ids: Which KGraphPlanner tools to include.  ``None`` uses
            the default set (web search, weather, place search).
        system_prompt: Override the default system prompt.
        checkpointer: Optional LangGraph checkpointer for conversation
            persistence across invocations.
        log_file: Optional path for AgentBox sandbox logs.
        sandbox_init_wait: Seconds to wait after sandbox creation for
            initialization to complete.

    Returns:
        Tuple of ``(compiled_agent, sandbox_backend, repo_id)``.
        The caller is responsible for calling ``backend.destroy()``
        when done.  The ``repo_id`` is returned so it can be stored
        and reused in future sessions.
    """
    # --- Repo ID ---
    if repo_id is None:
        repo_id = f"conversation-{uuid.uuid4().hex[:8]}"
    logger.info(f"Building auto_conversation agent (repo_id={repo_id})")

    # --- Sandbox ---
    create_kwargs: Dict[str, Any] = {
        "box_type": "git",
        "repo_id": repo_id,
    }
    if log_file:
        create_kwargs["log_file"] = log_file

    backend = AgentBoxSandbox.create(orchestrator_url, **create_kwargs)
    if sandbox_init_wait > 0:
        time.sleep(sandbox_init_wait)

    # --- Upload skills and memory ---
    upload_skills_and_memory(backend)

    # --- Initialize workspace dirs ---
    backend.execute(
        "mkdir -p /workspace/knowledge/topics /workspace/research /workspace/drafts /workspace/output"
    )

    # --- KGraphPlanner tools ---
    if tool_manager is None:
        config = agent_config or AgentConfig()
        tool_manager = ToolManager(config=config)
        tool_manager.load_tools_from_config()

    kgraph_tools = get_kgraph_tools(tool_manager, tool_ids)
    logger.info(f"Loaded {len(kgraph_tools)} KGraphPlanner tools for Deep Agent")

    # --- Build Deep Agent ---
    agent = create_deep_agent(
        model=model,
        tools=kgraph_tools,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        backend=backend,
        skills=[_SANDBOX_SKILLS_ROOT],
        memory=[_SANDBOX_MEMORY_ROOT + "AGENTS.md"],
        checkpointer=checkpointer,
    )

    return agent, backend, repo_id


def build_agent_with_existing_backend(
    backend: AgentBoxSandbox,
    *,
    model: str | BaseChatModel = "openai:gpt-5",
    tool_manager: Optional[ToolManager] = None,
    agent_config: Optional[AgentConfig] = None,
    tool_ids: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """Build a Deep Agent using a pre-existing sandbox backend.

    Use this when you want to manage the sandbox lifecycle yourself
    (e.g. reusing a sandbox across multiple agent invocations).

    Args:
        backend: An already-created AgentBoxSandbox instance.
        model: LLM model identifier or instance.
        tool_manager: Pre-configured ToolManager.
        agent_config: AgentConfig for tool loading.
        tool_ids: Which KGraphPlanner tools to include.
        system_prompt: Override the default system prompt.
        checkpointer: Optional LangGraph checkpointer.

    Returns:
        A compiled Deep Agent graph.
    """
    # --- Upload skills and memory ---
    upload_skills_and_memory(backend)

    # --- KGraphPlanner tools ---
    if tool_manager is None:
        config = agent_config or AgentConfig()
        tool_manager = ToolManager(config=config)
        tool_manager.load_tools_from_config()

    kgraph_tools = get_kgraph_tools(tool_manager, tool_ids)

    # --- Build Deep Agent ---
    return create_deep_agent(
        model=model,
        tools=kgraph_tools,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        backend=backend,
        skills=[_SANDBOX_SKILLS_ROOT],
        memory=[_SANDBOX_MEMORY_ROOT + "AGENTS.md"],
        checkpointer=checkpointer,
    )
