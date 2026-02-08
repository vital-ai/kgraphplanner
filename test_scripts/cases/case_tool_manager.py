"""
Case: ToolManager — registration, config loading, tool functions, integration.

Tests the ToolManager's ability to:
  1. Register tools and retrieve them by name
  2. Load configuration from YAML and from environment variables
  3. Create tool functions for LangChain
  4. Full integration with config-driven tool loading
"""

from __future__ import annotations

import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.tools.websearch.web_search_tool import WebSearchTool

from test_scripts.cases.test_result import TestResult
from test_scripts.cases.case_helpers import log, write_log, OUTPUT_DIR

TOOL_CONFIG = {"tool_endpoint": "http://localhost:8008"}


# --- Sub-tests ---

def _test_basic(buf: io.StringIO) -> bool:
    """Test basic tool manager functionality."""
    log(buf, "\n  --- Basic ToolManager ---")

    tm = ToolManager()
    websearch_tool = WebSearchTool(TOOL_CONFIG, tm)

    assert "google_web_search_tool" in tm.list_available_tools(), "Tool not registered"
    log(buf, "  ✓ Tool registration")

    retrieved = tm.get_tool("google_web_search_tool")
    assert retrieved is not None and retrieved.get_tool_name() == "google_web_search_tool"
    log(buf, "  ✓ Tool retrieval")

    tool_fn = tm.get_tool_function("google_web_search_tool")
    assert tool_fn is not None
    log(buf, "  ✓ Tool function retrieval")

    descriptions = tm.get_tool_descriptions()
    assert "google_web_search_tool" in descriptions
    log(buf, "  ✓ Tool descriptions")

    log(buf, "  ✅ Basic: passed")
    return True


def _test_config_loading(buf: io.StringIO) -> bool:
    """Test configuration loading."""
    log(buf, "\n  --- Config Loading ---")

    test_config_path = "/tmp/test_agent_config_runner.yaml"
    test_config_content = """
tools:
  endpoint: "http://localhost:8008"
  enabled:
    - google_web_search_tool
  web_search:
    default_num_results: 10

agent:
  name: "Test Agent"
"""
    with open(test_config_path, 'w') as f:
        f.write(test_config_content)

    try:
        tm = ToolManager(config_path=test_config_path)

        assert tm.config.get_tool_endpoint() == "http://localhost:8008"
        log(buf, "  ✓ Endpoint config")

        assert tm.config.get_enabled_tools() == ["google_web_search_tool"]
        log(buf, "  ✓ Enabled tools config")

        assert tm.config.name == "Test Agent"
        log(buf, "  ✓ Agent name config")

        tm.load_tools_from_config()
        assert "google_web_search_tool" in tm.list_available_tools()
        log(buf, "  ✓ Tools loaded from config")

    finally:
        os.unlink(test_config_path)

    log(buf, "  ✅ Config Loading: passed")
    return True


def _test_env_loading(buf: io.StringIO) -> bool:
    """Test configuration loading from environment variables."""
    log(buf, "\n  --- Env Loading (from_env) ---")

    # Inject test env vars with a unique prefix to avoid collisions
    prefix = "KGTEST"
    test_vars = {
        f"{prefix}__TOOLS__ENDPOINT": "http://localhost:9999",
        f"{prefix}__TOOLS__ENABLED": "weather_tool,place_search_tool",
        f"{prefix}__AGENT__NAME": "Env Test Agent",
        f"{prefix}__AGENT__MODEL__TEMPERATURE": "0.3",
        f"{prefix}__CHECKPOINTING__ENABLED": "false",
        f"{prefix}__MEMORY__MAX_HISTORY": "42",
    }
    for k, v in test_vars.items():
        os.environ[k] = v

    try:
        config = AgentConfig.from_env(prefix=prefix)

        assert config.tools.endpoint == "http://localhost:9999"
        log(buf, "  ✓ Endpoint from env")

        assert config.tools.enabled == ["weather_tool", "place_search_tool"]
        log(buf, "  ✓ Enabled tools from env (comma-separated)")

        assert config.name == "Env Test Agent"
        log(buf, "  ✓ Agent name from env")

        assert config.model.temperature == 0.3
        log(buf, f"  ✓ Temperature coerced to float: {config.model.temperature}")

        assert config.checkpointing.enabled is False
        log(buf, "  ✓ Checkpointing.enabled coerced to bool")

        assert config.memory.max_history == 42
        log(buf, f"  ✓ max_history coerced to int: {config.memory.max_history}")

        # Verify ToolManager accepts it
        tm = ToolManager(config=config)
        assert tm.config.tools.endpoint == "http://localhost:9999"
        log(buf, "  ✓ ToolManager accepts AgentConfig from env")

    finally:
        for k in test_vars:
            os.environ.pop(k, None)

    log(buf, "  ✅ Env Loading: passed")
    return True


def _test_websearch_props(buf: io.StringIO) -> bool:
    """Test websearch tool properties via manager."""
    log(buf, "\n  --- WebSearch Tool Properties ---")

    tool = WebSearchTool(TOOL_CONFIG)

    assert tool.get_tool_name() == "google_web_search_tool"
    log(buf, "  ✓ Tool name")

    assert "Search the web" in tool.get_tool_description()
    log(buf, "  ✓ Tool description")

    schema = tool.get_tool_schema()
    assert schema is not None
    log(buf, "  ✓ Tool schema")

    tool_fn = tool.get_tool_function()
    assert tool_fn is not None, "Tool function is None"
    log(buf, f"  ✓ Tool function OK ({type(tool_fn).__name__})")

    log(buf, "  ✅ WebSearch Properties: passed")
    return True


def _test_integration(buf: io.StringIO) -> bool:
    """Test full integration with config template."""
    log(buf, "\n  --- Integration ---")

    template_path = os.path.join(project_root, "agent_config.yaml.template")
    test_config_path = "/tmp/integration_test_config_runner.yaml"

    with open(template_path, 'r') as f:
        config_content = f.read()
    with open(test_config_path, 'w') as f:
        f.write(config_content)

    try:
        tm = ToolManager(config_path=test_config_path)
        tm.load_tools_from_config()

        available = tm.list_available_tools()
        enabled = tm.get_enabled_tools()
        log(buf, f"  Available: {available}")
        log(buf, f"  Enabled: {[t.get_tool_name() for t in enabled]}")

        tool_fns = tm.get_enabled_tool_functions()
        assert len(tool_fns) > 0, "No tool functions"
        log(buf, f"  ✓ {len(tool_fns)} tool functions ready for LangChain")

    finally:
        os.unlink(test_config_path)

    log(buf, "  ✅ Integration: passed")
    return True


# --- Main entry point ---

SUB_TESTS = [
    ("Basic", _test_basic),
    ("Config Loading", _test_config_loading),
    ("Env Loading", _test_env_loading),
    ("WebSearch Properties", _test_websearch_props),
    ("Integration", _test_integration),
]


async def run() -> TestResult:
    """Run all tool manager tests."""
    load_dotenv()
    buf = io.StringIO()

    log(buf, "  === Tool Manager Tests ===")
    passed = []
    failed = []

    for name, fn in SUB_TESTS:
        try:
            fn(buf)
            passed.append(name)
        except Exception as e:
            log(buf, f"  ❌ {name}: {e}")
            failed.append(name)

    log(buf, f"\n  === Results: {len(passed)}/{len(SUB_TESTS)} passed ===")
    for name in passed:
        log(buf, f"    ✅ {name}")
    for name in failed:
        log(buf, f"    ❌ {name}")

    write_log(buf, "tool_manager_run.log")

    assert not failed, f"Failed: {failed}"

    return TestResult(
        name="Tool Manager",
        passed=True,
        details={
            "sub_tests": len(SUB_TESTS),
            "passed": len(passed),
        },
    )
