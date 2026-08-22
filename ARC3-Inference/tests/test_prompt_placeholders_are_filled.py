"""Every ``{placeholder}`` in the agent's prompt templates must be supplied by the agent.

A half-revert shipped to Kaggle on 2026-08-21 (kernel taaf-qwen38-nvfp4-vllm v4):
``tool_agent.py`` was restored from upstream-baseline but ``prompts.py`` was not, so the
template still carried ``{tool_inventory}`` and every one of the 25 games crashed in under
eight seconds with ``KeyError: 'tool_inventory'`` -- after a ten-minute server start that
passed every smoke check. The two files are one unit; this pins that.
"""
from __future__ import annotations

import re
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parents[1] / "inference" / "agent"
_PLACEHOLDER = re.compile(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}(?!\})")


def _placeholders(source: str) -> set[str]:
    return set(_PLACEHOLDER.findall(source))


def _format_kwargs(source: str) -> set[str]:
    supplied: set[str] = set()
    for call in re.finditer(r"\.format\(([^)]*)\)", source, re.S):
        supplied.update(re.findall(r"(\w+)\s*=", call.group(1)))
    return supplied


def test_every_prompt_placeholder_is_supplied_by_the_agent() -> None:
    prompts = (_AGENT_DIR / "prompts.py").read_text(encoding="utf-8")
    agent = (_AGENT_DIR / "tool_agent.py").read_text(encoding="utf-8")
    missing = _placeholders(prompts) - _format_kwargs(agent)
    assert not missing, (
        f"prompts.py uses {sorted(missing)} but tool_agent.py never passes them to .format(); "
        f"every game would crash with KeyError at its first turn."
    )
