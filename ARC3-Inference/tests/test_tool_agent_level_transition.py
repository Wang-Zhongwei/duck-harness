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
from viewer.data import (
    _build_lightweight_viewer_steps,
    _build_viewer_steps,
    _hydrate_lightweight_step,
    _normalize_event,
    _split_labeled_sections,
)


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
    agent._last_step_summary_reported = False
    agent._level_review_context_pending = False
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
    # Even stale transition state must be discarded when the whole run ends.
    agent._pending_level_transition = _pending(frame)
    agent._level_review_context_pending = True
    agent._capture_level_transition(
        fallback_frame=frame,
        history_entries=[HistoryEntry(action="", frame=frame)],
        winning_result={"level_completed": True, "run_complete": True},
    )
    assert agent._pending_level_transition is None

    assert not agent._level_review_context_pending

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
    assert agent._pending_level_transition is None
    assert agent._level_review_context_pending
    assert agent._history_messages == []


def test_analyze_reviews_old_frame_then_allows_first_new_level_action(
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
    callback_calls: list[dict[str, Any]] = []

    def build_user_message(prompt: str, frame: Frame | None) -> dict[str, Any]:
        attached_frames.append(frame)
        return {"role": "user", "content": f"{prompt}\nFRAME_LEVEL={frame.level if frame else None}"}

    responses = iter(
        [
            _ChatCompletionResult(
                message={
                    "content": chr(10).join(
                        [
                            "World model: Reviewed old world.",
                            "Goal model: Reviewed old goal.",
                            "Action model: Reviewed old actions.",
                            "Cross-level notes: Reviewed transferable invariant.",
                        ]
                    )
                }
            ),
            _ChatCompletionResult(
                message={
                    "content": chr(10).join(
                        [
                            "World model: Fresh new world.",
                            "Open questions: Is the step size unchanged?",
                            "Plan: Probe one move, then compare frames.",
                            "Cross-level notes: Layout-specific overwrite.",
                        ]
                    ),
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "python",
                                "arguments": json.dumps(
                                    {"code": "action([valid_actions[0]])"}
                                ),
                            },
                        }
                    ],
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

    def step_env(request: dict[str, Any]) -> dict[str, Any]:
        callback_calls.append(request)
        return {
            "executed": True,
            "action_num": 6,
            "level": 2,
            "valid_actions": ["LEFT"],
            "level_completed": False,
            "run_complete": False,
            "action_display": "LEFT",
        }

    agent._build_user_message = build_user_message
    agent._chat_completion = chat_completion
    result = agent.analyze(
        state_path,
        action_num=5,
        valid_actions=["LEFT"],
        step_env=step_env,
        transcript_path=transcript_path,
        analysis_step=2,
    )

    assert result is not None
    assert not result.retryable_failure
    assert result.step_executed
    assert callback_calls == [{"actions": [{"action": "LEFT"}]}]
    assert attached_frames == [old_frame, new_frame]
    assert requests[0]["tools"] is None
    assert requests[1]["tools"] is not None
    assert "NEW_LEVEL_SENTINEL" not in json.dumps(requests[0]["messages"])
    first_exposure_request = json.dumps(requests[1]["messages"])
    assert "Use the completed-level review as context" in first_exposure_request
    assert "Map the reviewed mechanics" in first_exposure_request
    # the merged prompt keeps the ordinary tool contract rather than a one-shot
    # initialization prompt
    assert "Only tool: `python`." in first_exposure_request
    assert "Focus on what changed most recently" not in first_exposure_request
    transcript = transcript_path.read_text(encoding="utf-8")
    assert "[LEVEL REVIEW]" in transcript
    assert "[NEXT LEVEL INITIALIZATION]" not in transcript
    assert agent._pending_level_transition is None
    assert agent._level_review_context_pending
    assert agent._summarized_knowledge["world_model"] == "Fresh new world."
    assert agent._summarized_knowledge["goal_model"] == ""
    assert agent._summarized_knowledge["action_model"] == ""
    assert (
        agent._summarized_knowledge["open_questions"]
        == "Is the step size unchanged?"
    )
    assert (
        agent._summarized_knowledge["current_plan"]
        == "Probe one move, then compare frames."
    )
    assert (
        agent._summarized_knowledge["cross_level_notes"]
        == "Reviewed transferable invariant."
    )
    carried_models = chr(10).join(agent._summarized_knowledge_lines())
    assert "World model: Fresh new world." in carried_models
    assert "Goal model:" not in carried_models
    assert "Action model:" not in carried_models
    assert "Open questions: Is the step size unchanged?" in carried_models
    assert "Plan: Probe one move, then compare frames." in carried_models


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
    assert agent._pending_level_transition is pending
    assert not agent._level_review_context_pending


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
    assert agent._pending_level_transition is pending
    assert not agent._level_review_context_pending


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


def test_initialization_allows_action_and_captures_models_best_effort(
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
    agent._pending_level_transition = None
    agent._level_review_context_pending = True
    agent._summarized_knowledge["cross_level_notes"] = "Reviewed invariant"
    callback_calls: list[dict[str, Any]] = []

    def step_env(request: dict[str, Any]) -> dict[str, Any]:
        callback_calls.append(request)
        return {
            "executed": True,
            "action_num": 6,
            "level": 2,
            "valid_actions": ["LEFT"],
            "action_display": "LEFT",
        }

    agent._step_env_callback = step_env
    inspection = agent._run_python_tool(
        state_path,
        {"code": "result = current_frame.level"},
    )
    acted = agent._run_python_tool(
        state_path,
        {"code": "result = action([valid_actions[0]])"},
    )

    assert json.loads(inspection.content)["result"] == 2
    assert acted.step_executed
    assert callback_calls == [{"actions": [{"action": "LEFT"}]}]
    # The context remains until the resulting authoritative-state prompt is shown.
    assert agent._level_review_context_pending

    agent._level_review_context_pending = True
    agent._update_summarized_knowledge_from_assistant(
        chr(10).join(
            [
                "World model: New world.",
                "Goal model: New goal.",
                "Open questions: New uncertainty.",
                "Plan: Test the uncertainty.",
                "Cross-level notes: Layout-specific overwrite.",
            ]
        )
    )
    assert agent._summarized_knowledge["world_model"] == "New world."
    assert agent._summarized_knowledge["goal_model"] == "New goal."
    assert agent._summarized_knowledge["action_model"] == ""
    assert agent._summarized_knowledge["open_questions"] == "New uncertainty."
    assert agent._summarized_knowledge["current_plan"] == "Test the uncertainty."
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


def test_viewer_preserves_multiple_analysis_events_for_one_step() -> None:
    events = [
        {"type": "initial", "board": [[0]], "score": 0, "level": 1},
        {
            "type": "analysis",
            "analysis_step": 2,
            "transcript": "[LEVEL REVIEW]\nWorld model: Reviewed completed level.",
        },
        {
            "type": "analysis",
            "analysis_step": 2,
            "transcript": (
                "[NEXT LEVEL INITIALIZATION]\n"
                "World model: Initialized new level."
            ),
        },
    ]
    normalized_events = [_normalize_event(event) for event in events]

    full_step = _build_viewer_steps(normalized_events)[0]
    assert [section["label"] for section in full_step["localContext"]["sections"]] == [
        "LEVEL REVIEW",
        "NEXT LEVEL INITIALIZATION",
    ]

    lightweight_step = _build_lightweight_viewer_steps(events)[0]
    hydrated_step = _hydrate_lightweight_step(
        lightweight_step,
        events,
        request_snapshots=[],
        step_index=0,
    )
    assert [
        section["label"] for section in hydrated_step["localContext"]["sections"]
    ] == ["LEVEL REVIEW", "NEXT LEVEL INITIALIZATION"]


def _inspection_only_response() -> _ChatCompletionResult:
    return _ChatCompletionResult(
        message={
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": "result = current_frame.level"}),
                    },
                }
            ]
        }
    )


def test_new_level_prompt_never_reports_the_finished_levels_actions(
    tmp_path: Path,
) -> None:
    """Regression: ar25 pass0 re-injected a finished level's action report.

    The separate initialization turn consumed the pending transition, so every
    following turn fell back to the ordinary prompt built from a
    `_last_step_summary` that was never cleared -- opening with "You have
    progressed to a new level!" plus a run of DOWNs taken on the level the agent
    had already left. Those actions belong to the previous level and must not
    appear on this one at all.
    """
    state_path = tmp_path / "tool_runtime_state.json"
    new_frame = _frame(15, step=19, level=2)
    write_runtime_state(
        state_path,
        current_frame=new_frame,
        history=[HistoryEntry(action="DOWN", frame=new_frame)],
    )
    agent = _bare_agent(tmp_path)
    agent._last_step_summary = {
        "executed_count": 16,
        "executed_actions": ["DOWN"] * 16,
        "level": 2,
        "level_transition": True,
        "board_changed": True,
    }
    agent._last_step_summary_reported = False
    agent._level_review_context_pending = True
    agent._summarized_knowledge["cross_level_notes"] = "Mirror mechanic carries over."

    prompts: list[str] = []

    def build_user_message(prompt: str, frame: Frame | None) -> dict[str, Any]:
        prompts.append(prompt)
        return {"role": "user", "content": prompt}

    agent._build_user_message = build_user_message
    agent._chat_completion = lambda messages, *, tools, request_timeout_seconds=None: (
        _inspection_only_response()
    )

    def step_env(request: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("no action should be executed in this test")

    for _ in range(3):
        agent.analyze(
            state_path,
            action_num=19,
            valid_actions=["UP", "DOWN"],
            step_env=step_env,
            transcript_path=tmp_path / "transcript.txt",
            analysis_step=4,
        )

    assert len(prompts) == 3
    for prompt in prompts:
        assert "The code executed 16 actions" not in prompt
        assert "Executed actions" not in prompt
        assert "You have progressed to a new level!" not in prompt
        # the review is what carried the agent here, and it is what the prompt says
        assert "Use the completed-level review as context" in prompt
        assert "Current state: step 20, level 2." in prompt
        assert "Map the reviewed mechanics" in prompt


def test_acting_on_the_new_level_closes_the_mapping_prompt(tmp_path: Path) -> None:
    state_path = tmp_path / "tool_runtime_state.json"
    new_frame = _frame(15, step=19, level=2)
    write_runtime_state(
        state_path,
        current_frame=new_frame,
        history=[HistoryEntry(action="DOWN", frame=new_frame)],
    )
    agent = _bare_agent(tmp_path)
    agent._level_review_context_pending = True

    prompts: list[str] = []

    def build_user_message(prompt: str, frame: Frame | None) -> dict[str, Any]:
        prompts.append(prompt)
        return {"role": "user", "content": prompt}

    responses = iter(
        [
            _ChatCompletionResult(
                message={
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "python",
                                "arguments": json.dumps(
                                    {"code": "action([valid_actions[0]])"}
                                ),
                            },
                        }
                    ]
                }
            ),
            _inspection_only_response(),
        ]
    )

    def step_env(request: dict[str, Any]) -> dict[str, Any]:
        return {
            "executed": True,
            "action_num": 20,
            "level": 2,
            "valid_actions": ["UP", "DOWN"],
            "action_display": "UP",
        }

    agent._build_user_message = build_user_message
    agent._chat_completion = lambda messages, *, tools, request_timeout_seconds=None: next(
        responses
    )

    agent.analyze(
        state_path,
        action_num=19,
        valid_actions=["UP", "DOWN"],
        step_env=step_env,
        transcript_path=tmp_path / "transcript.txt",
        analysis_step=4,
    )
    assert agent._level_review_context_pending
    agent.analyze(
        state_path,
        action_num=20,
        valid_actions=["UP", "DOWN"],
        step_env=step_env,
        transcript_path=tmp_path / "transcript.txt",
        analysis_step=5,
    )

    agent.analyze(
        state_path,
        action_num=20,
        valid_actions=["UP", "DOWN"],
        step_env=step_env,
        transcript_path=tmp_path / "transcript.txt",
        analysis_step=5,
    )

    assert "Map the reviewed mechanics" in prompts[0]
    assert "Use the completed-level review as context" in prompts[0]
    # The first post-action prompt keeps the review grounding without replaying
    # the initial exploration sequence or repeating the level status.
    assert prompts[1].startswith("Current state: step 21, level 2.")
    assert prompts[1].splitlines()[1].startswith("Use the completed-level review as context")
    assert "Map the reviewed mechanics" in prompts[1]
    assert prompts[0].splitlines()[1:3] == prompts[1].splitlines()[1:3]
    assert "The code executed" not in prompts[1]
    assert "Executed actions" not in prompts[1]
    assert "You are still on the same level" not in prompts[1]
    assert not agent._level_review_context_pending
    assert "Map the reviewed mechanics" not in prompts[2]
    assert "Executed actions" not in prompts[2]
    assert "No new actions have been executed since the previous turn." in prompts[2]


def test_reset_notice_disappears_once_the_model_refills_the_fields() -> None:
    """The carry block is the authority; the notice must not contradict it."""
    agent = _bare_agent()
    agent._level_review_context_pending = True
    agent._summarized_knowledge["cross_level_notes"] = "Mirror mechanic carries over."

    def prompt() -> str:
        return agent._build_user_prompt(
            19,
            valid_actions=["UP", "DOWN"],
            current_frame=_frame(15, step=19, level=2),
            history_entries=[],
            previous_step_summary=None,
            previous_step_summary_reported=False,
            level_review_context=True,
        )

    first = prompt()
    assert "The per-level fields were reset by the completed-level review" in first
    assert "Map the reviewed mechanics" in first

    agent._summarized_knowledge["world_model"] = "Two mirrored sprites and a target."
    later = prompt()
    assert "The per-level fields were reset by the completed-level review" not in later
    assert "- World model: Two mirrored sprites and a target." in later
    # the rest of the mapping guidance still stands until an action lands
    assert "Map the reviewed mechanics" in later
