from __future__ import annotations

import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

_DEFAULT_ENV_PREFIX = "KGPLAN"


def _coerce_value(raw: str) -> Any:
    """Best-effort coercion of an env-var string to a Python scalar.

    Rules (applied in order):
      - JSON-parseable → parsed value (handles lists, nested dicts, booleans,
        numbers, and quoted strings)
      - Comma-separated string (contains ',') → list of stripped strings
      - Otherwise → the raw string unchanged
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    if "," in raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


def _env_to_dict(prefix: str = _DEFAULT_ENV_PREFIX) -> Dict[str, Any]:
    """Parse environment variables with *prefix* into a nested dict.

    Variable names use ``__`` (double underscore) as the hierarchy separator.
    The prefix itself is stripped and is **not** part of the resulting keys.
    Keys are lower-cased so that ``KGPLAN__TOOLS__ENDPOINT`` becomes
    ``{"tools": {"endpoint": "..."}}``.

    Scalar values are auto-coerced via :func:`_coerce_value`.

    Example environment::

        KGPLAN__TOOLS__ENDPOINT=http://localhost:9000
        KGPLAN__TOOLS__ENABLED=weather_tool,place_search_tool
        KGPLAN__AGENT__MODEL__TEMPERATURE=0.5
        KGPLAN__CHECKPOINTING__ENABLED=true

    Produces::

        {
            "tools": {"endpoint": "http://localhost:9000",
                      "enabled": ["weather_tool", "place_search_tool"]},
            "agent": {"model": {"temperature": 0.5}},
            "checkpointing": {"enabled": True},
        }
    """
    full_prefix = prefix + "__"
    result: Dict[str, Any] = {}

    for key, value in os.environ.items():
        if not key.startswith(full_prefix):
            continue
        parts = key[len(full_prefix):].lower().split("__")
        if not parts or not parts[0]:
            continue

        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce_value(value)

    return result


@dataclass(frozen=True)
class ToolConfig:
    """Configuration for the tool subsystem."""
    endpoint: str = "http://localhost:8008"
    enabled: List[str] = field(default_factory=list)
    web_search: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the LLM model."""
    provider: str = "openai"
    name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for checkpointing."""
    enabled: bool = True
    backend: str = "memory"


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for conversation memory."""
    enabled: bool = True
    max_history: int = 100


@dataclass(frozen=True)
class AgentConfig:
    """Typed, immutable configuration for KGraphPlanner agents.

    Construct programmatically::

        config = AgentConfig(
            tools=ToolConfig(endpoint="http://localhost:9000", enabled=["weather_tool"]),
        )

    Or from a YAML file::

        config = AgentConfig.from_yaml("agent_config.yaml")

    Or from a plain dict (e.g. loaded from a database)::

        config = AgentConfig.from_dict({"tools": {"endpoint": "..."}})
    """
    name: str = "KGraphPlanner Agent"
    description: str = "AI agent with planning and tool capabilities"
    tools: ToolConfig = field(default_factory=ToolConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging_level: str = "INFO"
    logging_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # --- Factory classmethods ------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str) -> AgentConfig:
        """Load configuration from a YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_file, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_env(cls, prefix: str = _DEFAULT_ENV_PREFIX) -> AgentConfig:
        """Build configuration from environment variables.

        Variables are expected in the form ``{PREFIX}__{SECTION}__{KEY}``
        using ``__`` as the hierarchy separator.  The default prefix is
        ``KGPLAN``, so for example::

            KGPLAN__TOOLS__ENDPOINT=http://localhost:9000
            KGPLAN__TOOLS__ENABLED=weather_tool,place_search_tool
            KGPLAN__AGENT__MODEL__TEMPERATURE=0.5
            KGPLAN__CHECKPOINTING__ENABLED=true

        Comma-separated values are split into lists.  JSON literals
        (numbers, booleans, arrays, objects) are parsed automatically.
        """
        return cls.from_dict(_env_to_dict(prefix))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        """Construct from a plain dictionary (e.g. parsed YAML or JSON)."""
        tools_data = data.get("tools", {})
        tool_config = ToolConfig(
            endpoint=tools_data.get("endpoint", ToolConfig.endpoint),
            enabled=tools_data.get("enabled", []),
            web_search=tools_data.get("web_search", {}),
        )

        model_data = data.get("agent", {}).get("model", {})
        model_config = ModelConfig(
            provider=model_data.get("provider", ModelConfig.provider),
            name=model_data.get("name", ModelConfig.name),
            temperature=model_data.get("temperature", ModelConfig.temperature),
            max_tokens=model_data.get("max_tokens", ModelConfig.max_tokens),
        )

        cp_data = data.get("checkpointing", {})
        cp_config = CheckpointConfig(
            enabled=cp_data.get("enabled", CheckpointConfig.enabled),
            backend=cp_data.get("backend", CheckpointConfig.backend),
        )

        mem_data = data.get("memory", {})
        mem_config = MemoryConfig(
            enabled=mem_data.get("enabled", MemoryConfig.enabled),
            max_history=mem_data.get("max_history", MemoryConfig.max_history),
        )

        agent_data = data.get("agent", {})
        log_data = data.get("logging", {})

        return cls(
            name=agent_data.get("name", cls.name),
            description=agent_data.get("description", cls.description),
            tools=tool_config,
            model=model_config,
            checkpointing=cp_config,
            memory=mem_config,
            logging_level=log_data.get("level", cls.logging_level),
            logging_format=log_data.get("format", cls.logging_format),
        )

    # --- Convenience accessors (backward-compatible) -------------------------

    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration as a plain dict."""
        return {
            "endpoint": self.tools.endpoint,
            "enabled": list(self.tools.enabled),
            "web_search": dict(self.tools.web_search),
        }

    def get_tool_endpoint(self) -> str:
        """Get the tool server endpoint URL."""
        return self.tools.endpoint

    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tool name strings."""
        return list(self.tools.enabled)

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation lookup for backward compatibility.

        Prefer direct attribute access (e.g. ``config.tools.endpoint``)
        over this method in new code.
        """
        parts = key.split(".")
        obj: Any = self
        for p in parts:
            if isinstance(obj, dict):
                obj = obj.get(p)
            elif hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return default
            if obj is None:
                return default
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full configuration to a plain dictionary."""
        from dataclasses import asdict
        return asdict(self)