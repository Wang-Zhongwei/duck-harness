from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent, _format_level_up_line, _level_up_events


def _agent() -> ToolAgent:
    return ToolAgent(
        model="test-model",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )


def _entry(action: str, *, step: int, level: int) -> HistoryEntry:
    return HistoryEntry(
        action=action,
        frame=Frame(grid=((0, 0), (0, 0)), step=step, level=level),
    )


def test_level_up_events_capture_each_increase_with_step_and_action():
    history = [
        _entry("", step=0, level=1),
        _entry("LEFT", step=1, level=1),
        _entry("SPACE", step=2, level=2),
        _entry("LEFT", step=3, level=2),
        _entry("MOUSE(row=4, col=7)", step=4, level=3),
    ]

    assert _level_up_events(history) == [
        (1, 2, 2, "SPACE"),
        (2, 3, 4, "MOUSE(row=4, col=7)"),
    ]


def test_level_up_events_ignore_level_resets():
    history = [
        _entry("", step=0, level=1),
        _entry("SPACE", step=1, level=2),
        _entry("RESET", step=2, level=1),
        _entry("SPACE", step=3, level=2),
    ]

    assert _level_up_events(history) == [
        (1, 2, 1, "SPACE"),
        (1, 2, 3, "SPACE"),
    ]


def test_user_prompt_includes_level_up_line_only_after_a_level_up():
    agent = _agent()
    flat_history = [
        _entry("", step=0, level=1),
        _entry("LEFT", step=1, level=1),
    ]
    prompt = agent._build_user_prompt(
        1,
        valid_actions=["LEFT"],
        current_frame=flat_history[-1].frame,
        history_entries=flat_history,
    )
    assert "Level-up key steps" not in prompt

    leveled_history = [
        *flat_history,
        _entry("SPACE", step=2, level=2),
    ]
    prompt = agent._build_user_prompt(
        2,
        valid_actions=["LEFT"],
        current_frame=leveled_history[-1].frame,
        history_entries=leveled_history,
    )
    assert "Level-up key steps so far: level 1 -> 2 at step 2 (action: SPACE)." in prompt
    assert "transitions.at_step(step)" in prompt


def test_format_level_up_line_renders_multiple_events():
    line = _format_level_up_line([(1, 2, 2, "SPACE"), (2, 3, 4, "LEFT")])
    assert "level 1 -> 2 at step 2 (action: SPACE); level 2 -> 3 at step 4 (action: LEFT)" in line
