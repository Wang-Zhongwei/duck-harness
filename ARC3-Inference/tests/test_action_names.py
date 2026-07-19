"""Tests for the model-facing action-name mapping."""
from inference.agent.action_names import (
    to_engine_action,
    to_model_action,
    to_model_actions,
)
from inference.agent.tool_agent import (
    _format_valid_action_line,
    _normalize_valid_actions,
)


def test_mapped_engine_actions_get_model_labels() -> None:
    assert to_model_actions(["ACTION1", "ACTION2", "ACTION6", "RESET"]) == [
        "UP",
        "DOWN",
        "MOUSE",
        "RESET",
    ]


def test_unmapped_engine_actions_are_hidden_from_the_model() -> None:
    # ACTION7 (undo) is deliberately unsupported; it must not leak into the
    # valid-actions list under its raw engine name.
    assert to_model_actions(["ACTION3", "ACTION4", "ACTION6", "ACTION7"]) == [
        "LEFT",
        "RIGHT",
        "MOUSE",
    ]
    assert to_engine_action("ACTION7") is None


def test_single_action_display_passthrough_is_preserved() -> None:
    assert to_model_action("ACTION5") == "SPACE"
    assert to_model_action("SOMETHING_ELSE") == "SOMETHING_ELSE"


def test_normalize_valid_actions_drops_unmapped_engine_actions() -> None:
    # Regression: `to_model_action` falls back to the raw name, so normalizing
    # the engine list without an explicit guard leaked ACTION7 into both the
    # "Valid actions right now:" prompt line and the sandbox `valid_actions`
    # variable, while the executor still rejected it.
    assert _normalize_valid_actions(
        ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
    ) == ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]


def test_valid_action_prompt_line_never_mentions_unmapped_actions() -> None:
    line = _format_valid_action_line(["ACTION1", "ACTION6", "ACTION7"])
    assert "ACTION7" not in line
    assert line == "UP, MOUSE"
