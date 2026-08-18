import json
from pathlib import Path
from typing import Any

from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import (
    AnalyzerModelConfig,
    ToolAgent,
    _ChatCompletionResult,
    _PendingLevelTransition,
    _empty_world_model,
)
from viewer.data import _split_labeled_sections


def _frame(value: int, *, step: int, level: int) -> Frame:
    return Frame(grid=((value,),), step=step, level=level)


def _bare_agent(runtime_dir: Path | None = None) -> ToolAgent:
    agent = object.__new__(ToolAgent)
    agent._summarized_knowledge = _empty_world_model()
    agent._pending_level_transition = None
    agent._history_messages = []
    agent._session_runtime_dir = runtime_dir
    agent._session_total_tokens = 0
    agent._session_generated_tokens = 0
    agent._last_step_summary = None
    agent._last_action_result = None
    agent._step_env_callback = None
    agent._current_valid_actions = ["LEFT"]
    agent._python_timeout = 5
    agent._tool_output_chars = 16_000
    agent._tool_output_tokens = 4_000
    agent._model = AnalyzerModelConfig(
        provider="test", base_url="http://example.invalid/v1", model_id="test-model"
    )
    agent._system_prompt = "normal system prompt"
    agent._tool_steps = 1
    agent._yield_seconds = None
    agent._save_request_logs = False
    agent._max_output_tokens = None
    agent._reply_reserve_tokens = 512
    agent._context_budget_tokens = 100_000
    agent._request_safety_margin_tokens = 512
    return agent


def _pending(old_frame: Frame) -> _PendingLevelTransition:
    return _PendingLevelTransition(
        completed_frame=old_frame,
        completed_history=[HistoryEntry(action="", frame=old_frame)],
        winning_result={
            "executed": True,
            "level_completed": True,
            "run_complete": False,
            "action_display": "LEFT",
        },
        completed_level_models={
            "world_model": "Old world",
            "goal_model": "Old goal",
            "action_model": "Old actions",
            "cross_level_notes": "Old transferable note",
        },
    )


def test_transition_capture_uses_history_pre_win_frame_and_snapshots_models() -> None:
    agent = _bare_agent()
    agent._summarized_knowledge.update(
        {
            "world_model": "Completed world",
            "goal_model": "Completed goal",
            "action_model": "Completed actions",
            "cross_level_notes": "Transfer this",
        }
    )
    old_frame = _frame(1, step=4, level=1)
    new_frame = _frame(15, step=5, level=2)

    agent._capture_level_transition(
        fallback_frame=new_frame,
        history_entries=[
            HistoryEntry(action="", frame=old_frame),
            HistoryEntry(action="LEFT", frame=new_frame),
        ],
        winning_result={
            "level_completed": True,
            "run_complete": False,
            "action_display": "LEFT",
        },
    )

    pending = agent._pending_level_transition
    assert pending is not None
    assert pending.completed_frame == old_frame
    assert pending.completed_history[-1].frame == old_frame
    assert pending.completed_level_models["world_model"] == "Completed world"
    agent._summarized_knowledge["world_model"] = ""
    assert pending.completed_level_models["world_model"] == "Completed world"


def test_final_win_does_not_create_pending_review() -> None:
    agent = _bare_agent()
    frame = _frame(1, step=4, level=1)
    agent._capture_level_transition(
        fallback_frame=frame,
        history_entries=[HistoryEntry(action="", frame=frame)],
        winning_result={"level_completed": True, "run_complete": True},
    )
    assert agent._pending_level_transition is None


def test_level_review_isolated_to_completed_frame_and_replaces_only_cross_notes() -> None:
    agent = _bare_agent()
    old_frame = _frame(1, step=4, level=1)
    new_frame = _frame(15, step=5, level=2)
    pending = _pending(old_frame)
    agent._pending_level_transition = pending
    agent._history_messages = [
        {"role": "user", "content": "NEW_LEVEL_SENTINEL"},
    ]
    calls: list[dict[str, Any]] = []

    def build_user_message(prompt: str, frame: Frame | None) -> dict[str, Any]:
        assert frame == old_frame
        return {"role": "user", "content": f"{prompt}\nFRAME={frame.grid if frame else None}"}

    def chat_completion(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        request_timeout_seconds: float | None = None,
    ) -> _ChatCompletionResult:
        calls.append({"messages": messages, "tools": tools})
        return _ChatCompletionResult(
            message={
                "content": (
                    "World model: Revised completed world.\n"
                    "Goal model: Revised completed goal.\n"
                    "Action model: Revised completed actions.\n"
                    "Cross-level notes: Transfer invariant mechanics only."
                )
            }
        )

    agent._build_user_message = build_user_message
    agent._chat_completion = chat_completion
    sections: list[tuple[str, str]] = []
    succeeded, _ = agent._complete_level_review(
        pending,
        append_transcript=lambda label, content: sections.append((label, content)),
        request_timeout_seconds=None,
    )

    assert succeeded
    assert len(calls) == 1
    assert calls[0]["tools"] is None
    rendered_request = json.dumps(calls[0]["messages"])
    assert "NEW_LEVEL_SENTINEL" not in rendered_request
    assert str(old_frame.grid) in rendered_request
    assert str(new_frame.grid) not in rendered_request
    assert agent._summarized_knowledge["world_model"] == ""
    assert agent._summarized_knowledge["goal_model"] == ""
    assert agent._summarized_knowledge["action_model"] == ""
    assert (
        agent._summarized_knowledge["cross_level_notes"]
        == "Transfer invariant mechanics only."
    )
    assert pending.phase == "initialization"
    assert agent._history_messages == []


def test_analyze_reviews_old_frame_before_initializing_with_new_frame(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "tool_runtime_state.json"
    transcript_path = tmp_path / "transcript.txt"
    old_frame = _frame(1, step=4, level=1)
    new_frame = _frame(15, step=5, level=2)
    write_runtime_state(
        state_path,
        current_frame=new_frame,
        history=[
            HistoryEntry(action="", frame=old_frame),
            HistoryEntry(action="LEFT", frame=new_frame),
        ],
    )
    agent = _bare_agent(tmp_path)
    agent._pending_level_transition = _pending(old_frame)
    agent._history_messages = [{"role": "user", "content": "NEW_LEVEL_SENTINEL"}]
    attached_frames: list[Frame | None] = []
    requests: list[dict[str, Any]] = []

    def build_user_message(prompt: str, frame: Frame | None) -> dict[str, Any]:
        attached_frames.append(frame)
        return {"role": "user", "content": f"{prompt}\nFRAME_LEVEL={frame.level if frame else None}"}

    responses = iter(
        [
            _ChatCompletionResult(
                message={
                    "content": (
                        "World model: Reviewed old world.\n"
                        "Goal model: Reviewed old goal.\n"
                        "Action model: Reviewed old actions.\n"
                        "Cross-level notes: Reviewed transferable invariant."
                    )
                }
            ),
            _ChatCompletionResult(
                message={
                    "content": (
                        "World model: Fresh new world.\n"
                        "Goal model: Fresh new goal.\n"
                        "Action model: Fresh new actions."
                    )
                }
            ),
        ]
    )

    def chat_completion(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        request_timeout_seconds: float | None = None,
    ) -> _ChatCompletionResult:
        requests.append({"messages": messages, "tools": tools})
        return next(responses)

    agent._build_user_message = build_user_message
    agent._chat_completion = chat_completion
    result = agent.analyze(
        state_path,
        action_num=5,
        valid_actions=["LEFT"],
        step_env=lambda _request: {},
        transcript_path=transcript_path,
        analysis_step=2,
    )

    assert result is not None
    assert not result.retryable_failure
    assert not result.step_executed
    assert attached_frames == [old_frame, new_frame]
    assert requests[0]["tools"] is None
    assert requests[1]["tools"] is not None
    assert "NEW_LEVEL_SENTINEL" not in json.dumps(requests[0]["messages"])
    transcript = transcript_path.read_text(encoding="utf-8")
    assert "[LEVEL REVIEW]" in transcript
    assert "[NEXT LEVEL INITIALIZATION]" in transcript
    assert agent._pending_level_transition is None
    assert agent._summarized_knowledge["world_model"] == "Fresh new world."
    assert (
        agent._summarized_knowledge["cross_level_notes"]
        == "Reviewed transferable invariant."
    )


def test_missing_review_fields_get_one_correction_then_retryable_checkpoint() -> None:
    agent = _bare_agent()
    pending = _pending(_frame(1, step=4, level=1))
    agent._pending_level_transition = pending
    calls: list[list[dict[str, Any]]] = []

    agent._build_user_message = lambda prompt, frame: {
        "role": "user",
        "content": prompt,
    }

    def chat_completion(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        request_timeout_seconds: float | None = None,
    ) -> _ChatCompletionResult:
        calls.append(messages)
        return _ChatCompletionResult(message={"content": "World model: Still incomplete."})

    agent._chat_completion = chat_completion
    succeeded, _ = agent._complete_level_review(
        pending,
        append_transcript=lambda _label, _content: None,
        request_timeout_seconds=None,
    )

    assert not succeeded
    assert len(calls) == 2
    assert "Format correction required" in calls[1][-1]["content"]
    assert pending.phase == "review"


def test_analyze_returns_retryable_failure_when_review_correction_still_fails(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "tool_runtime_state.json"
    frame = _frame(15, step=5, level=2)
    write_runtime_state(
        state_path,
        current_frame=frame,
        history=[HistoryEntry(action="LEFT", frame=frame)],
    )
    agent = _bare_agent(tmp_path)
    pending = _pending(_frame(1, step=4, level=1))
    agent._pending_level_transition = pending
    agent._chat_completion = lambda messages, tools=None, request_timeout_seconds=None: (
        _ChatCompletionResult(message={"content": "World model: Incomplete."})
    )

    result = agent.analyze(
        state_path,
        action_num=5,
        valid_actions=["LEFT"],
        transcript_path=tmp_path / "transcript.txt",
    )

    assert result is not None
    assert result.retryable_failure
    assert not result.step_executed
    assert agent._pending_level_transition is pending
    assert pending.phase == "review"


def test_python_action_transition_hides_new_frame_and_captures_old_models(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "tool_runtime_state.json"
    old_frame = _frame(1, step=4, level=1)
    new_frame = _frame(15, step=5, level=2)
    write_runtime_state(
        state_path,
        current_frame=old_frame,
        history=[HistoryEntry(action="", frame=old_frame)],
    )
    agent = _bare_agent(tmp_path)
    agent._summarized_knowledge.update(
        {
            "world_model": "Completed world",
            "goal_model": "Completed goal",
            "action_model": "Completed actions",
            "cross_level_notes": "Old notes",
        }
    )

    def step_env(_request: dict[str, Any]) -> dict[str, Any]:
        write_runtime_state(
            state_path,
            current_frame=new_frame,
            history=[
                HistoryEntry(action="", frame=old_frame),
                HistoryEntry(action="LEFT", frame=new_frame),
            ],
        )
        return {
            "executed": True,
            "action_num": 5,
            "level": 2,
            "valid_actions": ["LEFT"],
            "level_completed": True,
            "run_complete": False,
            "action_display": "LEFT",
        }

    agent._step_env_callback = step_env
    dispatch = agent._run_python_tool(
        state_path,
        {"code": "action('LEFT')\nresult = current_frame.ascii"},
    )
    payload = json.loads(dispatch.content)

    assert dispatch.step_executed
    assert payload["result"] == old_frame.ascii
    assert payload["result"] != new_frame.ascii
    assert agent._pending_level_transition is not None
    assert agent._pending_level_transition.completed_frame == old_frame
    assert (
        agent._pending_level_transition.completed_level_models["world_model"]
        == "Completed world"
    )


def test_initialization_allows_inspection_but_blocks_action_until_models_exist(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "tool_runtime_state.json"
    new_frame = _frame(15, step=5, level=2)
    write_runtime_state(
        state_path,
        current_frame=new_frame,
        history=[HistoryEntry(action="LEFT", frame=new_frame)],
    )
    agent = _bare_agent(tmp_path)
    pending = _pending(_frame(1, step=4, level=1))
    pending.phase = "initialization"
    agent._pending_level_transition = pending
    agent._summarized_knowledge["cross_level_notes"] = "Reviewed invariant"
    callback_calls: list[dict[str, Any]] = []
    agent._step_env_callback = lambda request: callback_calls.append(request) or {}

    inspection = agent._run_python_tool(
        state_path,
        {"code": "result = current_frame.level"},
    )
    blocked = agent._run_python_tool(
        state_path,
        {"code": "result = action('LEFT')"},
    )

    assert json.loads(inspection.content)["result"] == 2
    blocked_payload = json.loads(blocked.content)["result"]
    assert not blocked.step_executed
    assert blocked_payload["error"].startswith("Environment actions are blocked")
    assert callback_calls == []

    assert not agent._record_next_level_models(
        "World model: New world.\nGoal model: New goal."
    )
    assert agent._pending_level_transition is pending
    assert agent._record_next_level_models(
        "Action model: New actions.\nCross-level notes: Layout-specific overwrite."
    )
    assert agent._pending_level_transition is None
    assert agent._summarized_knowledge["cross_level_notes"] == "Reviewed invariant"


def test_viewer_classifies_transition_sections_as_reasoning() -> None:
    sections = _split_labeled_sections(
        "[LEVEL REVIEW]\nreview\n\n"
        "[NEXT LEVEL INITIALIZATION]\nremapping\n"
    )
    assert sections == [
        {"label": "LEVEL REVIEW", "content": "review", "kind": "reasoning"},
        {
            "label": "NEXT LEVEL INITIALIZATION",
            "content": "remapping",
            "kind": "reasoning",
        },
    ]
