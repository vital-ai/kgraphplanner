"""
Case: Autonomous Conversation Agent — multi-turn conversation with
persistent knowledge in a git-backed sandbox.

Uses the agent implementation from ``kgraphplanner.sample.auto_conversation``.
This test explores a multi-turn conversation where the agent accumulates
knowledge across turns, saves it to the sandbox filesystem, and uses it
to answer follow-up questions.

The same ``repo_id``, ``thread_id``, and sandbox backend persist across
all turns so the agent can recall context from earlier in the conversation.

Prerequisites:
    - AgentBox orchestrator running  (docker compose up --build -d)
    - Tool server running for KGraphPlanner tools
    - OPENAI_API_KEY in environment
"""

from __future__ import annotations

import io
import logging as _logging
import os
import sys
import asyncio
import time as _time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

from kgraphplanner.sample.auto_conversation import build_agent
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import (
    create_tool_manager, check_tools_available,
    log, write_log, _Tee, OUTPUT_DIR,
)

# Local directory where sandbox files are downloaded
DOWNLOAD_DIR = os.path.join(project_root, "test_output", "auto_conversation")


# ============================================================
# Configuration
# ============================================================

ORCHESTRATOR_URL = os.getenv("AGENTBOX_ORCHESTRATOR_URL", "http://localhost:8090")
MODEL = os.getenv("AUTO_CONVERSATION_MODEL", "anthropic:claude-sonnet-4-5-20250929")


# ============================================================
# Conversation turns — a coherent multi-turn dialogue where
# later turns depend on knowledge from earlier turns.
# ============================================================

CONVERSATION_TURNS = [
    {
        "input": (
            "I'm planning a trip to Tokyo next month. "
            "Can you research the top 5 must-visit attractions there? "
            "Save what you learn to your knowledge files."
        ),
        "description": "Turn 1: Initial research — Tokyo attractions",
    },
    {
        "input": (
            "What's the current weather in Tokyo? "
            "Based on what you already know about the attractions, "
            "which ones would be best to visit in this weather?"
        ),
        "description": "Turn 2: Follow-up — weather + cross-reference with turn 1 knowledge",
    },
    {
        "input": (
            "Now research the best restaurants near the top attraction "
            "you recommended. Find at least 3 options."
        ),
        "description": "Turn 3: Deeper follow-up — restaurants near previously recommended attraction",
    },
    {
        "input": (
            "Summarize everything you know about my Tokyo trip so far. "
            "Check your knowledge files and give me a complete overview "
            "of attractions, weather, and restaurant recommendations."
        ),
        "description": "Turn 4: Knowledge recall — summarize all accumulated knowledge",
    },
    {
        "input": (
            "Write a one-page travel brief as a Markdown file to "
            "/workspace/output/tokyo_travel_brief.md with all the "
            "information you've gathered. Include sections for "
            "Attractions, Weather, Dining, and Tips. "
            "Then git add, commit, and push."
        ),
        "description": "Turn 5: Synthesis — compile knowledge into deliverable + git",
    },
    {
        "input": (
            "Now generate a polished PDF report from the travel brief you just wrote.\n\n"
            "1. First get a timestamp: execute python3 -c \"from datetime import datetime; "
            "print(datetime.now().strftime('%Y%m%d_%H%M%S'))\"\n\n"
            "2. Generate the PDF using the reportgen shell command:\n"
            "   reportgen /workspace/output/tokyo_travel_brief.md "
            "-o /workspace/output/tokyo_travel_report_<TIMESTAMP>.pdf "
            "--title 'Tokyo Trip Planning Report' "
            "--author 'Auto Conversation Agent' --toc\n\n"
            "3. Git add, commit, and push all files in /workspace/output/.\n"
            "4. Confirm both the .md and .pdf files exist by running: ls -la /workspace/output/"
        ),
        "description": "Turn 6: Report generation — PDF from accumulated knowledge + download",
        "download_files": True,
    },
]


# ============================================================
# Helpers
# ============================================================

def _msg_text(content) -> str:
    """Extract plain text from an AI message content (str or list of blocks)."""
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content)


async def _astream_with_commentary(agent, input_dict, config, buf: io.StringIO):
    """Stream agent execution, printing [AGENT] AI commentary in real time.

    Uses ``stream_mode="updates"`` which yields ``{node_name: update}``
    dicts.  AI commentary comes from the ``"model"`` node updates which
    contain ``{"messages": [AIMessage(...)]}``; sandbox commands are
    printed to stdout by the agentbox backend and captured via a
    stdout tee.

    Returns a dict with ``{"messages": all_collected_messages}``.
    """
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, buf)

    log_handler = _logging.StreamHandler(buf)
    log_handler.setLevel(_logging.INFO)
    log_handler.setFormatter(_logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    _logging.getLogger().addHandler(log_handler)

    all_messages: list = []
    try:
        async for chunk in agent.astream(
            input_dict, config=config, stream_mode="updates"
        ):
            if not isinstance(chunk, dict):
                continue
            for node_name, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                msgs = update.get("messages", [])
                if not isinstance(msgs, list):
                    continue
                for msg in msgs:
                    all_messages.append(msg)
                    role = getattr(msg, "type", "unknown")
                    content = getattr(msg, "content", "")
                    if role == "ai" and content:
                        text = _msg_text(content)
                        if text.strip():
                            print(f"  [AGENT] {text.strip()}")
    finally:
        sys.stdout = old_stdout
        _logging.getLogger().removeHandler(log_handler)

    return {"messages": all_messages}


def _extract_final_response(result: dict) -> str:
    """Extract the last AI message content from a Deep Agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if role == "ai" and content:
            text = _msg_text(content)
            if text.strip():
                return text.strip()
    return ""


def _download_sandbox_files(backend, buf: io.StringIO) -> bool:
    """Download all files from /workspace/output/ in the sandbox to DOWNLOAD_DIR.

    Lists the sandbox output directory, downloads every file found, and
    writes them to the local ``DOWNLOAD_DIR``.  Returns True if at least
    one file was downloaded successfully.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    log(buf, "\n  === Download: fetching files from sandbox /workspace/output/ ===")

    # List files in the sandbox output dir
    try:
        ls_result = backend.execute("ls -la /workspace/output/")
        log(buf, f"  Sandbox /workspace/output/:\n{ls_result.output}")
    except Exception as e:
        log(buf, f"  Warning: could not list sandbox output dir: {e}")

    # Discover file paths via find
    try:
        glob_result = backend.execute("find /workspace/output/ -type f")
        sandbox_paths = [
            p.strip() for p in glob_result.output.strip().split("\n")
            if p.strip() and not p.strip().startswith("find:")
        ]
    except Exception as e:
        log(buf, f"  Warning: could not glob sandbox output: {e}")
        sandbox_paths = []

    if not sandbox_paths:
        log(buf, "  No files found in /workspace/output/")
        return False

    log(buf, f"  Found {len(sandbox_paths)} file(s) to download")

    downloaded = 0
    for sp in sandbox_paths:
        try:
            downloads = backend.download_files([sp])
            for dl in downloads:
                if dl.error:
                    log(buf, f"  ✗ {dl.path}: {dl.error}")
                    continue
                filename = os.path.basename(dl.path)
                local_path = os.path.join(DOWNLOAD_DIR, filename)
                with open(local_path, "wb") as f:
                    f.write(dl.content)
                log(buf, f"  ✓ {dl.path} → {local_path} ({len(dl.content):,} bytes)")
                downloaded += 1
        except Exception as e:
            log(buf, f"  ✗ {sp}: download error: {e}")

    log(buf, f"  Downloaded {downloaded}/{len(sandbox_paths)} file(s) to {DOWNLOAD_DIR}")
    return downloaded > 0


def _dump_knowledge_files(backend, buf: io.StringIO):
    """List and display the contents of /workspace/knowledge/ for verification."""
    log(buf, "\n  === Knowledge Files Snapshot ===")
    try:
        tree_result = backend.execute("find /workspace/knowledge/ -type f 2>/dev/null | sort")
        files = [p.strip() for p in tree_result.output.strip().split("\n") if p.strip()]
        if not files:
            log(buf, "  (no knowledge files found)")
            return
        log(buf, f"  Knowledge files ({len(files)}):")
        for f in files:
            log(buf, f"    {f}")
        # Show contents of key files
        for key_file in ["/workspace/knowledge/facts.md", "/workspace/knowledge/history.md"]:
            if key_file in files:
                try:
                    cat_result = backend.execute(f"cat {key_file}")
                    log(buf, f"\n  --- {key_file} ---")
                    log(buf, cat_result.output)
                except Exception:
                    pass
    except Exception as e:
        log(buf, f"  Warning: could not list knowledge files: {e}")


# ============================================================
# Main entry point
# ============================================================

async def run(request_filter: list[int] | None = None) -> TestResult:
    """Run the autonomous conversation agent test.

    Args:
        request_filter: Optional list of 1-based turn numbers to run.
                        If None, all turns are run.
    """
    load_dotenv()
    buf = io.StringIO()

    # --- Setup ---
    log(buf, "  === Step 1: Setup ===")
    tm = create_tool_manager()
    if not check_tools_available(tm):
        return TestResult(
            name="Auto Conversation Agent (deep agent)",
            passed=True,
            details={"skipped": True, "reason": "No tools available (tool server not running?)"},
        )

    log(buf, f"  Model: {MODEL}")
    log(buf, f"  Orchestrator: {ORCHESTRATOR_URL}")

    # --- Build agent + sandbox ---
    log(buf, "\n  === Step 2: Build Agent ===")
    log_dir = os.path.join(project_root, "test_output")
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sandbox_log_file = os.path.join(log_dir, f"auto_conversation_sandbox_{ts}.log")

    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)

    try:
        agent, backend, repo_id = build_agent(
            orchestrator_url=ORCHESTRATOR_URL,
            model=MODEL,
            tool_manager=tm,
            log_file=sandbox_log_file,
            checkpointer=checkpointer,
        )
    except Exception as e:
        log(buf, f"  ❌ Failed to build agent: {e}")
        import traceback
        log(buf, traceback.format_exc())
        write_log(buf, "auto_conversation_agent_run.log")
        return TestResult(
            name="Auto Conversation Agent (deep agent)",
            passed=False,
            details={"error": str(e), "stage": "build"},
        )

    log(buf, f"  Agent built successfully")
    log(buf, f"  Repo ID: {repo_id}  (save this to reconnect later)")
    log(buf, f"  Sandbox log: {sandbox_log_file}")

    # --- Determine which turns to run ---
    if request_filter:
        turns_to_run = [(i, CONVERSATION_TURNS[i - 1]) for i in request_filter if 1 <= i <= len(CONVERSATION_TURNS)]
    else:
        turns_to_run = list(enumerate(CONVERSATION_TURNS, 1))

    log(buf, f"\n  === Step 3: Execute {len(turns_to_run)}-Turn Conversation ===")
    log(buf, f"  Thread ID (shared across all turns): {repo_id}")
    sub_results = []

    # Use the SAME thread_id for all turns so the agent sees
    # the full conversation history (or its summarized form).
    thread_id = repo_id

    try:
        for i, tc in turns_to_run:
            user_input = tc["input"]
            desc = tc["description"]
            log(buf, f"\n  {'=' * 50}")
            log(buf, f"  --- Turn {i}: {desc} ---")
            log(buf, f"  {'=' * 50}")
            log(buf, f"  User: {user_input}")

            try:
                _req_t0 = _time.time()

                # Stream with real-time [AGENT] commentary logging.
                # Same thread_id for all turns — conversation continuity.
                config = {"configurable": {"thread_id": thread_id}}
                result = await _astream_with_commentary(
                    agent,
                    {"messages": [{"role": "user", "content": user_input}]},
                    config,
                    buf,
                )

                final_text = _extract_final_response(result)
                _req_elapsed = _time.time() - _req_t0

                log(buf, f"\n  Final response: {final_text}")
                log(buf, f"  ⏱️  Turn {i} took {_req_elapsed:.1f}s")

                ok = bool(final_text and len(final_text) > 20)

                # Dump knowledge files after each turn to show accumulation
                _dump_knowledge_files(backend, buf)

                # Download files from sandbox if flagged
                if tc.get("download_files") and backend is not None:
                    ok = _download_sandbox_files(backend, buf) and ok

                sub_results.append({
                    "turn": i, "desc": desc, "ok": ok, "elapsed": _req_elapsed,
                })

            except Exception as e:
                _req_elapsed = _time.time() - _req_t0
                log(buf, f"  ❌ Error after {_req_elapsed:.1f}s: {e}")
                import traceback
                log(buf, traceback.format_exc())
                sub_results.append({
                    "turn": i, "desc": desc, "ok": False, "error": str(e),
                    "elapsed": _req_elapsed,
                })

            await asyncio.sleep(0.5)

    finally:
        # Always destroy the sandbox
        log(buf, "\n  === Cleanup: Destroying sandbox ===")
        log(buf, f"  Repo ID was: {repo_id}")
        try:
            backend.destroy()
            log(buf, "  Sandbox destroyed")
        except Exception as e:
            log(buf, f"  Warning: sandbox destroy failed: {e}")

    # --- Summary ---
    passed_count = sum(1 for r in sub_results if r["ok"])
    total_count = len(sub_results)

    log(buf, f"\n  === Results: {passed_count}/{total_count} turns succeeded ===")
    for r in sub_results:
        status = "✅" if r["ok"] else "❌"
        elapsed = r.get("elapsed", 0)
        log(buf, f"    {status} Turn {r['turn']}: {r['desc']} ({elapsed:.1f}s)")

    # --- Write log ---
    write_log(buf, f"auto_conversation_agent_run_{ts}.log")

    return TestResult(
        name="Auto Conversation Agent (deep agent)",
        passed=passed_count == total_count,
        details={
            "turns_total": total_count,
            "turns_passed": passed_count,
            "repo_id": repo_id,
            "model": MODEL,
            "orchestrator_url": ORCHESTRATOR_URL,
            "output_dir": OUTPUT_DIR,
        },
    )
