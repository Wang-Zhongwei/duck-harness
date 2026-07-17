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


def test_tool_mode_adds_complete_optional_update_schema(tmp_path: Path) -> None:
    agent = _agent("tool")
    functions = {
        tool["function"]["name"]: tool["function"]
        for tool in agent._tools(_state_path(tmp_path))
    }

    assert set(functions) == {"python", "update_memory"}
    parameters = functions["update_memory"]["parameters"]
    assert set(parameters["properties"]) == {
        "world_model",
        "goal_model",
        "action_model",
        "recent_findings",
        "open_questions",
        "plan",
        "cross_level_notes",
    }
    assert "required" not in parameters
    assert parameters["additionalProperties"] is False


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

    payload = json.loads(
        agent._run_update_memory_tool(
            {
                "world_model": "  A token moves.  ",
                "goal_model": "Reach the target.",
                "action_model": "RIGHT shifts the token.",
                "recent_findings": "The last move changed the token only.",
                "open_questions": "Whether walls wrap.",
                "plan": "Probe the right edge.",
                "cross_level_notes": "Target colors transfer.",
            }
        ).content
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

    second = json.loads(
        agent._run_update_memory_tool(
            {"world_model": "A token moves horizontally.", "goal_model": ""}
        ).content
    )
    assert second == {"updated": ["world_model"]}
    assert agent._summarized_knowledge["world_model"] == "A token moves horizontally."
    assert agent._summarized_knowledge["goal_model"] == "Reach the target."


def test_level_transition_requires_model_authored_cross_level_summary() -> None:
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
    prompt = agent._build_user_prompt(
        9,
        valid_actions=["MOUSE"],
        current_frame=Frame(grid=((0, 0), (0, 0)), step=9, level=2),
        previous_step_summary=summary,
    )
    assert "You have progressed to a new level!\nCurrent state: step 10, level 2." in prompt
    assert "Completed-level models are pasted below" in prompt
    assert "- World model: Red advances the cycle; green reverses it." in prompt
    assert "- Goal model: Find the cycle offset that completes the board." in prompt
    assert "this temporary snapshot will not be shown after the next environment action" in prompt
    assert "REQUIRED before executing any environment action" in prompt
    assert "call `update_memory` with `cross_level_notes`" in prompt


def test_assistant_mode_requires_cross_level_notes_section_after_transition() -> None:
    agent = _agent("assistant")
    agent._completed_level_model_snapshot = {
        "world_model": "Buttons rotate a fixed cycle."
    }
    summary = {"level_transition": True, "level": 2, "executed_count": 1}

    prompt = agent._build_user_prompt(
        1,
        valid_actions=["MOUSE"],
        current_frame=Frame(grid=((0, 0), (0, 0)), step=1, level=2),
        previous_step_summary=summary,
    )

    assert "- World model: Buttons rotate a fixed cycle." in prompt
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
    update_schema = agent._tools(state_path)[1]["function"]["parameters"]
    assert "predictor_code" not in update_schema["properties"]


def test_prompts_are_mode_specific() -> None:
    assistant_prompt = _build_system_prompt(
        tool_output_tokens=128, model_update_mode="assistant"
    )
    tool_prompt = _build_system_prompt(
        tool_output_tokens=128, model_update_mode="tool"
    )

    assert "exactly one tool: `python`" in assistant_prompt
    assert "`update_memory`" not in assistant_prompt
    assert "two tools: `python` and `update_memory`" in tool_prompt
    assert "`update_memory` saves your models" in tool_prompt
    assert "`previous_frame`" in tool_prompt
    assert "`current_frame`" in tool_prompt
    assert "`last_action`" in tool_prompt


def test_tool_mode_prompts_use_prediction_check_not_parsing_history(tmp_path: Path) -> None:
    agent = _agent("tool")

    description = {
        tool["function"]["name"]: tool["function"]["description"]
        for tool in agent._tools(_state_path(tmp_path))
    }["update_memory"]
    assert "assistant-text parsing" not in description
    assert "previous_frame" in description
    assert "last_action" in description

    prompt = agent._build_user_prompt(
        3,
        valid_actions=["RIGHT"],
        current_frame=Frame(grid=((0, 0), (0, 0)), step=3, level=1),
        previous_step_summary={"executed_count": 1, "level": 1},
    )
    assert "A useful check: would these models have predicted" in prompt
    assert "put `update_memory` first" not in prompt
    assert "Below are the persistent memory carried" in prompt
