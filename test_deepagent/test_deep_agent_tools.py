"""Test: Deep Agent with tool-only configuration (no sandbox).

Uses the ``auto_tools`` sample agent with weather and place-search tools.
No Docker, no AgentBox, no filesystem — just an LLM with KGraphPlanner tools.

Prerequisites:
    - OPENAI_API_KEY in .env (or environment)
    - Tool server running at the configured endpoint (default http://localhost:8008)

Usage:
    python test_deepagent/test_deep_agent_tools.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message=".*NotRequired.*", category=UserWarning)

import httpx
from dotenv import load_dotenv

load_dotenv()

from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.sample.auto_tools import build_agent


class TeeWriter:
    """Duplicate writes to both a file and the original stream."""

    def __init__(self, log_path: str, original):
        self._file = open(log_path, "w", encoding="utf-8")
        self._original = original

    def write(self, data):
        self._original.write(data)
        self._original.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ============================================================
# Auth helper
# ============================================================

async def get_keycloak_token():
    """Get JWT token from Keycloak using env credentials."""
    username = os.getenv('KEYCLOAK_USER')
    password = os.getenv('KEYCLOAK_PASSWORD')
    realm = os.getenv('KEYCLOAK_REALM')
    client_id = os.getenv('KEYCLOAK_CLIENT_ID')
    client_secret = os.getenv('KEYCLOAK_CLIENT_SECRET')

    if not username or not password:
        return None, "KEYCLOAK_USER/KEYCLOAK_PASSWORD not set"

    token_url = f"http://localhost:8085/realms/{realm}/protocol/openid-connect/token"
    data = {
        'grant_type': 'password',
        'client_id': client_id,
        'username': username,
        'password': password,
        'scope': 'openid profile email',
    }
    if client_secret:
        data['client_secret'] = client_secret

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(token_url, data=data, timeout=5)
        resp.raise_for_status()
        token = resp.json().get('access_token')
        if not token:
            return None, "No access_token in Keycloak response"
        return token, None
    except Exception as e:
        return None, f"Keycloak token request failed: {e}"


async def create_authenticated_tool_manager() -> ToolManager:
    """Create ToolManager with JWT auth from Keycloak."""
    print_header("Setting up ToolManager with JWT auth")
    config = AgentConfig.from_env()
    tm = ToolManager(config=config)
    tm.load_tools_from_config()

    token, err = await get_keycloak_token()
    if err:
        print(f"  JWT auth: {err}")
    else:
        tm.set_jwt_token(token)
        print(f"  JWT token set")

    tools = tm.list_available_tools()
    print(f"  Available tools: {tools}")
    return tm


# ============================================================
# Test cases
# ============================================================

TEST_CASES = [
    {
        "name": "weather_single_city",
        "query": "What is the current weather in San Francisco?",
        "expect_keywords": ["san francisco", "temperature", "weather"],
        "description": "Single city weather lookup",
    },
    {
        "name": "place_search_restaurant",
        "query": "Find me the best pizza restaurant near Times Square in New York City.",
        "expect_keywords": ["pizza", "new york"],
        "description": "Place search for a restaurant near a landmark",
    },
    {
        "name": "weather_and_place_combined",
        "query": (
            "I'm planning a trip to Chicago. What's the weather like there right now, "
            "and can you find a good coffee shop downtown?"
        ),
        "expect_keywords": ["chicago", "weather"],
        "description": "Combined weather + place search in a single query",
    },
]

# ============================================================
# Helpers
# ============================================================

def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def extract_response_text(result: dict) -> str:
    """Extract the final AI response text from a Deep Agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if role == "ai" and content:
            if isinstance(content, list):
                text = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            else:
                text = str(content)
            if text.strip():
                return text.strip()
    return ""


def check_keywords(text: str, keywords: list[str]) -> tuple[bool, list[str]]:
    """Check if all expected keywords appear in the response (case-insensitive)."""
    lower_text = text.lower()
    missing = [kw for kw in keywords if kw.lower() not in lower_text]
    return len(missing) == 0, missing


def count_tool_calls(result: dict) -> int:
    """Count how many tool calls were made during the agent run."""
    count = 0
    messages = result.get("messages", [])
    for msg in messages:
        if getattr(msg, "type", "") == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                count += len(tool_calls)
    return count


# Max graph steps per invocation (prevents infinite tool-retry loops)
RECURSION_LIMIT = 25
# Max wall-clock seconds per test case
TEST_TIMEOUT_SECONDS = 60


# ============================================================
# Main
# ============================================================

async def main():
    # Set up log file
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_deep_agent_tools_{ts}.log")

    # Tee stdout to log file (same pattern as editorial/git tests)
    tee = TeeWriter(log_file, sys.stdout)
    sys.stdout = tee

    print(f"  Logging to {os.path.abspath(log_file)}")

    print_header("DEEP AGENT TOOLS-ONLY TEST")

    # --- Check prerequisites ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  ERROR: OPENAI_API_KEY not set. Exiting.")
        return False

    # --- Auth: get Keycloak JWT for tool server ---
    tool_manager = await create_authenticated_tool_manager()

    # --- Build agent ---
    print_header("Building tool-only Deep Agent")
    t0 = time.time()
    agent = build_agent(model="openai:gpt-4o-mini", tool_manager=tool_manager)
    build_time = time.time() - t0
    print(f"  Agent built in {build_time:.1f}s")

    # --- Run test cases ---
    results = []
    total_start = time.time()

    for i, tc in enumerate(TEST_CASES, 1):
        print_header(f"TEST {i}/{len(TEST_CASES)}: {tc['name']} — {tc['description']}")
        print(f"  Query: {tc['query']}\n")

        t_start = time.time()
        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [{"role": "user", "content": tc["query"]}]},
                    config={"recursion_limit": RECURSION_LIMIT},
                ),
                timeout=TEST_TIMEOUT_SECONDS,
            )
            elapsed = time.time() - t_start

            response_text = extract_response_text(result)
            tool_call_count = count_tool_calls(result)
            kw_pass, kw_missing = check_keywords(response_text, tc["expect_keywords"])

            print(f"  Time: {elapsed:.1f}s")
            print(f"  Tool calls: {tool_call_count}")
            print(f"  Keywords pass: {kw_pass}")
            if kw_missing:
                print(f"  Missing keywords: {kw_missing}")
            print(f"\n  [Response]\n")
            for line in response_text.split("\n"):
                print(f"    {line}")

            passed = kw_pass and tool_call_count > 0
            results.append({
                "name": tc["name"],
                "passed": passed,
                "elapsed": elapsed,
                "tool_calls": tool_call_count,
                "kw_pass": kw_pass,
                "kw_missing": kw_missing,
                "error": None,
            })

        except asyncio.TimeoutError:
            elapsed = time.time() - t_start
            print(f"  TIMEOUT after {elapsed:.1f}s (limit={TEST_TIMEOUT_SECONDS}s)")
            results.append({
                "name": tc["name"],
                "passed": False,
                "elapsed": elapsed,
                "tool_calls": 0,
                "kw_pass": False,
                "kw_missing": tc["expect_keywords"],
                "error": f"Timeout after {TEST_TIMEOUT_SECONDS}s",
            })
            continue

        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  ERROR: {e}")
            results.append({
                "name": tc["name"],
                "passed": False,
                "elapsed": elapsed,
                "tool_calls": 0,
                "kw_pass": False,
                "kw_missing": tc["expect_keywords"],
                "error": str(e),
            })

    total_elapsed = time.time() - total_start

    # --- Summary ---
    print_header("TEST SUMMARY")
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        error_info = f"  error={r['error']}" if r["error"] else ""
        print(
            f"  [{status}]  {r['name']:<30s}  "
            f"{r['elapsed']:5.1f}s  "
            f"tools={r['tool_calls']}{error_info}"
        )

    print(f"\n  Total: {passed_count} passed, {failed_count} failed, {total_elapsed:.1f}s elapsed")
    print(f"  Log:   {os.path.abspath(log_file)}")

    tee.close()
    sys.stdout = tee._original
    return failed_count == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
