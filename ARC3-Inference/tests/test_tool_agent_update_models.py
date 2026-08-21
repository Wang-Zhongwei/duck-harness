import json
from pathlib import Path

from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import (
    ToolAgent,
    _build_system_prompt,
    _empty_world_model,
    _normalize_model_update_mode,
)


def _agent(mode: str) -> ToolAgent:
    return ToolAgent(
        model="test-model",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
        model_update_mode=mode,
    )


def _state_path(tmp_path: Path) -> Path:
    frame = Frame(grid=((0, 0), (0, 0)), step=0, level=1)
    state_path = tmp_path / "tool_runtime_state.json"
    write_runtime_state(
        state_path,
        current_frame=frame,
        history=[HistoryEntry(action="", frame=frame)],
    )
    return state_path


def test_model_update_mode_is_explicit_and_validated() -> None:
    assert _normalize_model_update_mode(None) == "assistant"
    assert _normalize_model_update_mode(" TOOL ") == "tool"

    try:
        _normalize_model_update_mode("automatic")
    except ValueError as exc:
        assert "assistant" in str(exc)
        assert "tool" in str(exc)
    else:
        raise AssertionError("invalid model-update mode was accepted")


def test_tool_mode_exposes_single_python_tool_with_update_memory_runtime(tmp_path: Path) -> None:
    agent = _agent("tool")
    tools = agent._tools(_state_path(tmp_path))

    assert [tool["function"]["name"] for tool in tools] == ["python"]
    description = tools[0]["function"]["description"]
    assert "update_memory(...)" in description
    assert set(tools[0]["function"]["parameters"]["properties"]) == {"code"}


def test_assistant_mode_keeps_original_single_tool_and_prose_parser(tmp_path: Path) -> None:
    agent = _agent("assistant")
    tools = agent._tools(_state_path(tmp_path))
    assert [tool["function"]["name"] for tool in tools] == ["python"]

    agent._update_summarized_knowledge_from_assistant(
        "World model: A token moves on a grid.\n"
        "Goal model: Reach the target.\n"
        "Cross-level notes: Targets retain their shape."
    )
    assert agent._summarized_knowledge["world_model"] == "A token moves on a grid."
    assert agent._summarized_knowledge["goal_model"] == "Reach the target."
    assert agent._summarized_knowledge["cross_level_notes"] == "Targets retain their shape."


def test_tool_mode_replaces_prose_parsing_without_losing_any_model_fields() -> None:
    agent = _agent("tool")
    agent._update_summarized_knowledge_from_assistant(
        "World model: This prose must not be parsed in tool mode."
    )
    assert agent._summarized_knowledge == _empty_world_model()

    payload = agent._apply_memory_update(
        {
            "world_model": "  A token moves.  ",
            "goal_model": "Reach the target.",
            "action_model": "RIGHT shifts the token.",
            "recent_findings": "The last move changed the token only.",
            "open_questions": "Whether walls wrap.",
            "plan": "Probe the right edge.",
            "cross_level_notes": "Target colors transfer.",
        }
    )

    assert payload["updated"] == [
        "world_model",
        "goal_model",
        "action_model",
        "recent_findings",
        "open_questions",
        "plan",
        "cross_level_notes",
    ]
    assert agent._summarized_knowledge == {
        "world_model": "A token moves.",
        "goal_model": "Reach the target.",
        "action_model": "RIGHT shifts the token.",
        "recent_findings": "The last move changed the token only.",
        "open_questions": "Whether walls wrap.",
        "current_plan": "Probe the right edge.",
        "cross_level_notes": "Target colors transfer.",
    }

    second = agent._apply_memory_update(
        {"world_model": "A token moves horizontally.", "goal_model": ""}
    )
    assert second == {"updated": ["world_model"]}
    assert agent._summarized_knowledge["world_model"] == "A token moves horizontally."
    assert agent._summarized_knowledge["goal_model"] == "Reach the target."


def test_level_transition_omits_completed_level_transfer_prompt() -> None:
    agent = _agent("tool")
    agent._summarized_knowledge.update(
        {
            "world_model": "Red advances the cycle; green reverses it.",
            "goal_model": "Find the cycle offset that completes the board.",
            "action_model": "Button clicks rotate colored blocks.",
            "recent_findings": "Five red clicks completed level 1.",
            "open_questions": "Whether later levels have independent rows.",
            "current_plan": "Inspect the new layout before clicking.",
            "cross_level_notes": "Colored blocks occupy a fixed cycle.",
        }
    )
    summary = {
        "level_transition": True,
        "level": 2,
        "executed_count": 2,
        "executed_actions": ["MOUSE(row=32, col=4)"] * 2,
    }
    agent._last_step_summary = summary

    agent._update_summarized_knowledge_from_step_summary()

    assert agent._summarized_knowledge["cross_level_notes"] == (
        "Colored blocks occupy a fixed cycle."
    )
    assert all(
        agent._summarized_knowledge[key] == ""
        for key in (
            "world_model",
            "goal_model",
            "action_model",
            "recent_findings",
            "open_questions",
            "current_plan",
        )
    )
    intro, rest = agent._build_user_prompt(
        9,
        valid_actions=["MOUSE"],
        current_frame=Frame(grid=((0, 0), (0, 0)), step=9, level=2),
        previous_step_summary=summary,
    )
    prompt = f"{intro}\n{rest}"
    assert "- You have progressed to a new level!\n\nCurrent state:\n- Step 10, level 2." in prompt
    assert "- Cross-level notes: Colored blocks occupy a fixed cycle." in prompt
    assert "Completed-level models are pasted below" not in prompt
    assert "temporary snapshot" not in prompt
    assert "REQUIRED before executing any environment action" in prompt
    assert "call `update_memory(cross_level_notes=...)`" in prompt


def test_assistant_mode_omits_completed_level_transfer_prompt() -> None:
    agent = _agent("assistant")
    agent._summarized_knowledge["cross_level_notes"] = "Buttons rotate a fixed cycle."
    summary = {"level_transition": True, "level": 2, "executed_count": 1}

    intro, rest = agent._build_user_prompt(
        1,
        valid_actions=["MOUSE"],
        current_frame=Frame(grid=((0, 0), (0, 0)), step=1, level=2),
        previous_step_summary=summary,
    )
    prompt = f"{intro}\n{rest}"

    assert "- Cross-level notes: Buttons rotate a fixed cycle." in prompt
    assert "Completed-level models are pasted below" not in prompt
    assert "REQUIRED before executing any new action" in prompt
    assert "write a `Cross-level notes:` section" in prompt


def test_update_memory_has_no_predictor_backtest_or_action_gate(tmp_path: Path) -> None:
    agent = _agent("tool")
    state_path = _state_path(tmp_path)
    callback_calls = []

    def step_env(payload):
        callback_calls.append(payload)
        return {
            "executed": True,
            "action_num": 1,
            "level": 1,
            "score": 0,
            "reward": 0,
            "state": "NOT_FINISHED",
            "valid_actions": ["ACTION1"],
            "board_changed": False,
            "done": False,
            "level_completed": False,
            "game_over": False,
            "run_complete": False,
            "action_display": "RIGHT",
        }

    agent._step_env_callback = step_env
    dispatch = agent._run_python_tool(state_path, {"code": "action('RIGHT')"})

    assert dispatch.step_executed is True
    assert callback_calls == [{"actions": [{"action": "RIGHT"}]}]
    tools = agent._tools(state_path)
    assert [tool["function"]["name"] for tool in tools] == ["python"]
    assert "predictor_code" not in tools[0]["function"]["parameters"]["properties"]


def test_prompts_are_mode_specific() -> None:
    assistant_prompt = _build_system_prompt(
        tool_output_tokens=128, model_update_mode="assistant"
    )
    tool_prompt = _build_system_prompt(
        tool_output_tokens=128, model_update_mode="tool"
    )

    assert "exactly one tool: `python`" in assistant_prompt
    assert "`update_memory`" not in assistant_prompt
    assert "exactly one tool: `python`" in tool_prompt
    assert "`update_memory(...)` is a function available inside every `python` tool call" in tool_prompt
    assert "`previous_frame`" in tool_prompt
    assert "`current_frame`" in tool_prompt
    assert "`last_action`" in tool_prompt


def test_tool_mode_prompts_use_prediction_check_not_parsing_history(tmp_path: Path) -> None:
    agent = _agent("tool")

    description = {
        tool["function"]["name"]: tool["function"]["description"]
        for tool in agent._tools(_state_path(tmp_path))
    }["python"]
    assert "assistant-text parsing" not in description
    assert "previous_frame" in description
    assert "update_memory" in description

    intro, rest = agent._build_user_prompt(
        3,
        valid_actions=["RIGHT"],
        current_frame=Frame(grid=((0, 0), (0, 0)), step=3, level=1),
        previous_step_summary={"executed_count": 1, "level": 1},
    )
    prompt = f"{intro}\n{rest}"
    assert "A useful check: would these models have predicted" in prompt
    assert "put `update_memory` first" not in prompt
    assert "Persistent memory (carried from your previous turns):" in prompt


def test_python_tool_update_memory_updates_agent_state_end_to_end(tmp_path: Path) -> None:
    agent = _agent("tool")
    state_path = _state_path(tmp_path)

    dispatch = agent._run_python_tool(
        state_path,
        {
            "code": (
                "r = update_memory(world_model='A token moves.',\n"
                "                  cross_level_notes='Targets keep their shape.')\n"
                "print(r['updated'])"
            )
        },
    )

    assert agent._summarized_knowledge["world_model"] == "A token moves."
    assert agent._summarized_knowledge["cross_level_notes"] == "Targets keep their shape."
    assert "error" not in json.loads(dispatch.content)
    assert "['world_model', 'cross_level_notes']" in dispatch.content
