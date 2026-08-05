"""Per-tool configuration delivery.

Before this, `ToolManager.load_tools_from_config()` built one
`{"tool_endpoint": ...}` dict and handed the same one to every tool, so there was
no route for per-tool settings to reach a constructor. Nothing was broken --
every existing tool reads only `tool_endpoint` -- but the GitHub tools need a
per-instance repo list, so the route had to exist.
"""

import pytest

from kgraphplanner.config.agent_config import AgentConfig, ToolConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager


ENDPOINT = "http://localhost:8008"


def _config(**tools):
    return AgentConfig.from_dict({"tools": {"endpoint": ENDPOINT, **tools}})


class TestPerToolConfig:

    def test_block_under_tool_configs_is_delivered(self):
        c = _config(tool_configs={"github_list_issues": {"repos": ["vital-ai/agent-b"]}})
        assert c.get_config_for_tool("github_list_issues") == {
            "tool_endpoint": ENDPOINT,
            "repos": ["vital-ai/agent-b"],
        }

    def test_shorthand_block_directly_under_tools(self):
        """`tools: {weather_tool: {...}}` works without the tool_configs nesting."""
        c = _config(weather_tool={"units": "metric"})
        assert c.get_config_for_tool("weather_tool") == {
            "tool_endpoint": ENDPOINT,
            "units": "metric",
        }

    def test_tool_configs_wins_over_shorthand(self):
        c = _config(weather_tool={"units": "metric"},
                    tool_configs={"weather_tool": {"units": "imperial"}})
        assert c.get_config_for_tool("weather_tool")["units"] == "imperial"

    def test_tool_without_a_block_gets_only_shared_settings(self):
        c = _config(tool_configs={"other_tool": {"x": 1}})
        assert c.get_config_for_tool("weather_tool") == {"tool_endpoint": ENDPOINT}

    def test_reserved_keys_are_not_treated_as_tool_blocks(self):
        c = _config(enabled=["weather_tool"], web_search={"num_results": 4})
        assert "enabled" not in c.tools.tool_configs
        assert "web_search" not in c.tools.tool_configs
        assert "endpoint" not in c.tools.tool_configs

    def test_non_dict_values_are_not_tool_blocks(self):
        c = _config(enabled=["weather_tool"], some_flag=True)
        assert "some_flag" not in c.tools.tool_configs

    def test_mutating_a_delivered_config_does_not_corrupt_the_source(self):
        """A tool mutating what it was handed must not affect the config or the
        next tool's view of it. Nested values matter here -- a repo list is
        exactly the shape a shallow copy would share."""
        c = _config(tool_configs={"a": {"repos": ["x"], "nested": {"k": "v"}}})

        first = c.get_config_for_tool("a")
        first["tool_endpoint"] = "mutated"
        first["repos"].append("y")
        first["nested"]["k"] = "changed"

        second = c.get_config_for_tool("a")
        assert second["tool_endpoint"] == ENDPOINT
        assert second["repos"] == ["x"]
        assert second["nested"] == {"k": "v"}
        assert c.tools.tool_configs["a"]["repos"] == ["x"]

    def test_round_trips_through_get_tool_config(self):
        c = _config(tool_configs={"github_list_issues": {"repos": ["vital-ai/agent-b"]}})
        rt = AgentConfig.from_dict({"tools": c.get_tool_config()})
        assert rt.get_config_for_tool("github_list_issues") == c.get_config_for_tool("github_list_issues")

    def test_defaults_have_no_tool_configs(self):
        assert AgentConfig().tools.tool_configs == {}
        assert AgentConfig().get_config_for_tool("anything") == {
            "tool_endpoint": ToolConfig.endpoint
        }


class TestWebSearchBlockUnchanged:
    """`web_search` has no consumer -- the web search tools take their parameters
    from the tool call, not from config. It is preserved as-is rather than being
    remapped onto tool names it was never declared to belong to."""

    def test_web_search_field_is_preserved(self):
        c = _config(web_search={"num_results": 4})
        assert c.tools.web_search == {"num_results": 4}

    @pytest.mark.parametrize("tool", ["google_web_search_tool", "serper_web_search_tool"])
    def test_web_search_is_not_injected_into_the_search_tools(self, tool):
        c = _config(web_search={"num_results": 4})
        assert c.get_config_for_tool(tool) == {"tool_endpoint": ENDPOINT}


class TestToolManagerDelivery:
    """The tools registered here reach the network only when called, so
    constructing them is offline."""

    ALL_TOOLS = ["weather_tool", "serper_web_search_tool", "google_web_search_tool",
                 "place_search_tool", "google_address_validation_tool"]

    def test_existing_tools_still_load(self):
        tm = ToolManager(config=_config(enabled=self.ALL_TOOLS))
        tm.load_tools_from_config()
        assert sorted(tm.get_tool_names()) == sorted(self.ALL_TOOLS)
        assert len(tm.get_enabled_tool_functions()) == len(self.ALL_TOOLS)

    def test_existing_tools_receive_only_the_endpoint(self):
        """Regression: no tool's config changed shape when per-tool blocks landed."""
        tm = ToolManager(config=_config(enabled=self.ALL_TOOLS, web_search={"num_results": 4}))
        tm.load_tools_from_config()
        for name in self.ALL_TOOLS:
            assert tm.get_tool(name).config == {"tool_endpoint": ENDPOINT}, name

    def test_a_tool_with_a_block_receives_it(self):
        tm = ToolManager(config=_config(
            enabled=["weather_tool"],
            tool_configs={"weather_tool": {"units": "metric"}},
        ))
        tm.load_tools_from_config()
        assert tm.get_tool("weather_tool").config == {
            "tool_endpoint": ENDPOINT, "units": "metric",
        }

    def test_unknown_enabled_tool_is_skipped_not_fatal(self):
        tm = ToolManager(config=_config(enabled=["weather_tool", "no_such_tool"]))
        tm.load_tools_from_config()
        assert tm.get_tool_names() == ["weather_tool"]
