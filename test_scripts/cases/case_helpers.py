"""
Shared helpers for test_scripts test cases.

Provides common infrastructure: Keycloak auth, ToolManager setup,
logging, PNG generation, and execution with timing capture.
"""
from __future__ import annotations

import os
import sys
import io
import logging as _logging
from typing import Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import requests
from dotenv import load_dotenv
from langchain_core.runnables.graph import MermaidDrawMethod

from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager

OUTPUT_DIR = os.path.join(project_root, "test_output")


# --- Auth & tool setup ---

def get_keycloak_token():
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
        'scope': 'openid profile email'
    }
    if client_secret:
        data['client_secret'] = client_secret

    try:
        resp = requests.post(token_url, data=data, timeout=5)
        resp.raise_for_status()
        token = resp.json().get('access_token')
        if not token:
            return None, "No access_token in Keycloak response"
        return token, None
    except Exception as e:
        return None, f"Keycloak token request failed: {e}"


def create_tool_manager() -> ToolManager:
    """Create ToolManager with config from KGPLAN__ env vars, loaded tools, and JWT auth."""
    config = AgentConfig.from_env()
    tm = ToolManager(config=config)
    tm.load_tools_from_config()

    token, err = get_keycloak_token()
    if err:
        print(f"  JWT auth: {err}")
    else:
        tm.set_jwt_token(token)
        print(f"  JWT token set")

    return tm


def check_tools_available(tool_manager: ToolManager) -> bool:
    """Check if any tools are available (tool server running)."""
    try:
        available = tool_manager.list_available_tools()
        return len(available) > 0
    except Exception:
        return False


# --- Logging ---

def log(buf: io.StringIO, msg: str):
    """Print and write to log buffer."""
    print(msg)
    buf.write(msg + "\n")


def write_log(buf: io.StringIO, filename: str):
    """Flush log buffer to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(buf.getvalue())
    print(f"  Log written to {path}")


# --- PNG diagrams ---

async def save_png(compiled_graph, filename: str, buf: io.StringIO):
    """Generate and save a Mermaid PNG diagram."""
    import asyncio
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    try:
        graph_drawable = compiled_graph.get_graph()
        # Use API method to avoid pyppeteer Chrome atexit cleanup errors
        image_bytes = await asyncio.to_thread(
            graph_drawable.draw_mermaid_png,
            draw_method=MermaidDrawMethod.API
        )
        with open(path, "wb") as f:
            f.write(image_bytes)
        log(buf, f"  Diagram saved: {path} ({len(image_bytes)} bytes)")
    except Exception as e:
        log(buf, f"  Diagram generation failed ({filename}): {e}")


# --- Execution with timing capture ---

class _Tee:
    """Tee stdout to also write into a StringIO buffer."""
    def __init__(self, original, buf):
        self._orig = original
        self._buf = buf
    def write(self, data):
        self._orig.write(data)
        self._buf.write(data)
    def flush(self):
        self._orig.flush()


async def execute_with_logging(coro, buf: io.StringIO):
    """
    Run an awaitable while capturing both stdout and logger output
    into the log buffer for offline inspection.
    """
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, buf)

    log_handler = _logging.StreamHandler(buf)
    log_handler.setLevel(_logging.INFO)
    log_handler.setFormatter(_logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    _logging.getLogger().addHandler(log_handler)

    try:
        result = await coro
    finally:
        sys.stdout = old_stdout
        _logging.getLogger().removeHandler(log_handler)

    return result
