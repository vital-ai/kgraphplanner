"""
Case: Autonomous General Agent — Deep Agent with git-backed sandbox.

Uses the agent implementation from ``kgraphplanner.sample.auto_general``.
This is the *autonomous agent* counterpart of ``case_general_agent.py``
which uses the *structured orchestration* pattern.

The Deep Agent uses an implicit tool loop (no explicit action JSON or
conditional routing).  KGraphPlanner tools (web search, weather, place
search) are provided alongside the built-in file/shell/git tools.

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

from kgraphplanner.sample.auto_general import build_agent
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import (
    create_tool_manager, check_tools_available,
    log, write_log, _Tee, OUTPUT_DIR,
)

# Local directory where sandbox files are downloaded
DOWNLOAD_DIR = os.path.join(project_root, "test_output", "auto_general")


# ============================================================
# Configuration
# ============================================================

ORCHESTRATOR_URL = os.getenv("AGENTBOX_ORCHESTRATOR_URL", "http://localhost:8090")
MODEL = os.getenv("AUTO_GENERAL_MODEL", "anthropic:claude-sonnet-4-5-20250929")


# ============================================================
# Test requests (same set as the structured general agent)
# ============================================================

TEST_REQUESTS = [
    {
        "input": "What's the weather like in New York City right now?",
        "description": "Weather lookup (tool call expected)",
    },
    {
        "input": "Search the web for the latest news about artificial intelligence.",
        "description": "Web search (tool call expected)",
    },
    {
        "input": "Where is the Statue of Liberty located?",
        "description": "Place search (tool call expected)",
    },
    {
        "input": "Hello! How are you doing today?",
        "description": "General greeting (direct response expected)",
    },
    {
        "input": "What's the current temperature in San Francisco?",
        "description": "Weather lookup #2 (tool call expected)",
    },
    {
        "input": (
            "Find the top 5 rated restaurants in the world and then find the "
            "current weather in those places.\n\n"
            "After gathering all the data:\n\n"
            "1. Include a timestamp in all output file names (YYYYMMDD_HHMMSS format). "
            "   You can get the current time via: execute python3 -c \"from datetime import datetime; print(datetime.now().strftime('%Y%m%d_%H%M%S'))\"\n\n"
            "2. Write a polished Markdown report to /workspace/output/restaurants_weather_<TIMESTAMP>.md. "
            "   The report should include a title, date, a summary paragraph, and a table "
            "   with columns: Rank, Restaurant, City/Country, Cuisine, Temperature, Conditions.\n\n"
            "3. Generate a PDF from the Markdown report using the reportgen shell command:\n"
            "   reportgen /workspace/output/restaurants_weather_<TIMESTAMP>.md "
            "-o /workspace/output/restaurants_weather_<TIMESTAMP>.pdf "
            "--title 'Top 5 Restaurants & Weather Report' "
            "--author 'Auto General Agent' --toc\n\n"
            "4. Git add, commit, and push all files in /workspace/output/.\n"
            "5. Confirm both the .md and .pdf files exist by running: ls -la /workspace/output/"
        ),
        "description": "Multi-step: research → markdown report → PDF → git (with download)",
        "download_files": True,
    },
    {
        "input": "What are the top 5 largest music festivals in Europe and what is the weather like in those cities right now?",
        "description": "Multi-step planning #2 (web search + weather — non-restaurant domain)",
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

    # Discover file paths via glob
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


# ============================================================
# Main entry point
# ============================================================

async def run(request_filter: list[int] | None = None) -> TestResult:
    """Run the autonomous general agent test.

    Args:
        request_filter: Optional list of 1-based request numbers to run.
                        If None, all requests are run.
    """
    load_dotenv()
    buf = io.StringIO()

    # --- Setup ---
    log(buf, "  === Step 1: Setup ===")
    tm = create_tool_manager()
    if not check_tools_available(tm):
        return TestResult(
            name="Auto General Agent (deep agent)",
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
    sandbox_log_file = os.path.join(log_dir, f"auto_general_sandbox_{ts}.log")

    serializer = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serializer)

    try:
        agent, backend = build_agent(
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
        write_log(buf, "auto_general_agent_run.log")
        return TestResult(
            name="Auto General Agent (deep agent)",
            passed=False,
            details={"error": str(e), "stage": "build"},
        )

    log(buf, f"  Agent built successfully")
    log(buf, f"  Sandbox log: {sandbox_log_file}")

    # --- Execute test requests ---
    if request_filter:
        requests_to_run = [(i, TEST_REQUESTS[i - 1]) for i in request_filter if 1 <= i <= len(TEST_REQUESTS)]
    else:
        requests_to_run = list(enumerate(TEST_REQUESTS, 1))

    log(buf, f"\n  === Step 3: Execute {len(requests_to_run)} Requests ===")
    sub_results = []

    try:
        for i, tc in requests_to_run:
            user_input = tc["input"]
            desc = tc["description"]
            log(buf, f"\n  --- Request {i}: {desc} ---")
            log(buf, f"  Input: {user_input}")

            try:
                _req_t0 = _time.time()

                # Stream with real-time [AGENT] commentary logging.
                config = {"configurable": {"thread_id": f"auto-general-test-{i}"}}
                result = await _astream_with_commentary(
                    agent,
                    {"messages": [{"role": "user", "content": user_input}]},
                    config,
                    buf,
                )

                final_text = _extract_final_response(result)
                _req_elapsed = _time.time() - _req_t0

                log(buf, f"  Final response: {final_text}")
                log(buf, f"  ⏱️  Request {i} took {_req_elapsed:.1f}s")

                ok = bool(final_text and len(final_text) > 20)

                # Download files from sandbox if flagged
                if tc.get("download_files") and backend is not None:
                    ok = _download_sandbox_files(backend, buf) and ok

                sub_results.append({
                    "test": i, "desc": desc, "ok": ok, "elapsed": _req_elapsed,
                })

            except Exception as e:
                _req_elapsed = _time.time() - _req_t0
                log(buf, f"  ❌ Error after {_req_elapsed:.1f}s: {e}")
                import traceback
                log(buf, traceback.format_exc())
                sub_results.append({
                    "test": i, "desc": desc, "ok": False, "error": str(e),
                    "elapsed": _req_elapsed,
                })

            await asyncio.sleep(0.5)

    finally:
        # Always destroy the sandbox
        log(buf, "\n  === Cleanup: Destroying sandbox ===")
        try:
            backend.destroy()
            log(buf, "  Sandbox destroyed")
        except Exception as e:
            log(buf, f"  Warning: sandbox destroy failed: {e}")

    # --- Summary ---
    passed_count = sum(1 for r in sub_results if r["ok"])
    total_count = len(sub_results)

    log(buf, f"\n  === Results: {passed_count}/{total_count} requests succeeded ===")
    for r in sub_results:
        status = "✅" if r["ok"] else "❌"
        elapsed = r.get("elapsed", 0)
        log(buf, f"    {status} Test {r['test']}: {r['desc']} ({elapsed:.1f}s)")

    # --- Write log ---
    write_log(buf, f"auto_general_agent_run_{ts}.log")

    return TestResult(
        name="Auto General Agent (deep agent)",
        passed=passed_count == total_count,
        details={
            "requests_total": total_count,
            "requests_passed": passed_count,
            "model": MODEL,
            "orchestrator_url": ORCHESTRATOR_URL,
            "output_dir": OUTPUT_DIR,
        },
    )
