"""
Prompt loader for the general agent sample.

Loads prompt templates from the ``prompts/`` directory (sibling to this
module) and renders them with ``{{variable}}`` substitution.

Usage::

    from kgraphplanner.sample.general.prompt_loader import load_prompt

    # Load a raw template
    raw = load_prompt("orchestrator")

    # Load and render with variables
    rendered = load_prompt("orchestrator", tool_list="weather, search")
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


@lru_cache(maxsize=32)
def _read_template(name: str) -> str:
    """Read a prompt template file by name (without extension)."""
    path = os.path.join(_PROMPTS_DIR, f"{name}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompt(name: str, **variables: Any) -> str:
    """Load a prompt template and optionally render ``{{key}}`` placeholders.

    Parameters
    ----------
    name:
        Template file name without ``.txt`` extension
        (e.g. ``"orchestrator"``).
    **variables:
        Key-value pairs to substitute into ``{{key}}`` placeholders.
        Unmatched placeholders are left as-is so partial rendering is
        safe.

    Returns
    -------
    str
        The (optionally rendered) prompt text.
    """
    template = _read_template(name)
    if not variables:
        return template

    def _replace(match: re.Match) -> str:
        key = match.group(1).strip()
        return str(variables[key]) if key in variables else match.group(0)

    return re.sub(r"\{\{(.+?)\}\}", _replace, template)
