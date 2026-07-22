"""Step-window eviction and the per-step before/after board images.

Context is bounded by a rolling window of whole steps rather than by tokens.
Only the step being opened carries its two board images; older steps keep a
placeholder where each board was. The window shifts the prefix on every new
step by design, so prefix-cache reuse is scoped to one step's requests --
which is also why rewriting history to drop the images costs nothing.
"""

import pytest

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import _STEP_IMAGE_PLACEHOLDER as _PLACEHOLDER
from inference.agent.tool_agent import ToolAgent, _render_user_message_text


def _agent(*, context_steps: int = 5, images: bool = False) -> ToolAgent:
    agent = ToolAgent(
        model="test-model",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )
    agent._context_steps = context_steps
    agent._grid_images_enabled = images
    return agent


def _frame(fill: int, step: int = 0) -> Frame:
    return Frame(grid=tuple((fill,) * 4 for _ in range(4)), step=step, level=1)


def _turn(index: int, *, tool_calls: bool = True) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": f"user prompt {index}"}]
    if tool_calls:
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [{"id": f"call-{index}", "function": {"name": "python", "arguments": "{}"}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call-{index}", "content": f"tool result {index}"})
    messages.append({"role": "assistant", "content": f"assistant reply {index}"})
    return messages


def _image_step(index: int) -> list[dict]:
    """A step whose opening user message carries the before/after images."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"user prompt {index}\n- Board before this sequence:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
                {"type": "text", "text": "- Board after this sequence (the current board):"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": f"rest of prompt {index}"},
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [{"id": f"call-{index}", "function": {"name": "python", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": f"call-{index}", "content": f"tool result {index}"},
        {"role": "assistant", "content": f"assistant reply {index}"},
    ]


# --- step-boundary detection ------------------------------------------------


def test_image_bearing_user_message_starts_a_step() -> None:
    agent = _agent(images=True)
    assert agent._is_history_turn_start(_image_step(0)[0])
    # Mid-step follow-up prompts are plain strings and must not open a step.
    assert not agent._is_history_turn_start({"role": "user", "content": "You have not acted yet."})
    assert not agent._is_history_turn_start({"role": "assistant", "content": "reply"})


def test_plain_user_message_starts_a_step_when_images_are_disabled() -> None:
    agent = _agent(images=False)
    assert agent._is_history_turn_start({"role": "user", "content": "user prompt"})


# --- whole-step eviction ----------------------------------------------------


def test_drop_oldest_history_turn_removes_whole_turn() -> None:
    agent = _agent()
    history = [*_turn(0), *_turn(1)]
    assert agent._drop_oldest_history_turn(history)
    assert history == _turn(1)


def test_drop_oldest_history_turn_never_drops_final_turn() -> None:
    agent = _agent()
    history = _turn(0)
    assert not agent._drop_oldest_history_turn(history)
    assert history == _turn(0)


def test_drop_oldest_history_turn_removes_leading_debris_with_first_turn() -> None:
    agent = _agent()
    debris = [{"role": "tool", "tool_call_id": "stale", "content": "orphan"}]
    history = [*debris, *_turn(0), *_turn(1)]
    assert agent._drop_oldest_history_turn(history)
    assert history == _turn(1)


def test_drop_oldest_step_keeps_follow_up_prompt_with_its_step() -> None:
    agent = _agent(images=True)
    follow_up = {"role": "user", "content": "You have not acted yet."}
    step0 = [*_image_step(0), follow_up, {"role": "assistant", "content": "retry 0"}]
    history = [*step0, *_image_step(1)]
    assert agent._drop_oldest_history_turn(history)
    assert history == _image_step(1)


# --- rolling step window ----------------------------------------------------


def test_keep_recent_steps_keeps_last_n_whole_steps() -> None:
    agent = _agent(images=True)
    steps = [_image_step(i) for i in range(8)]
    history = [m for step in steps for m in step]
    kept = agent._keep_recent_steps(history, max_steps=3)
    assert kept == [m for step in steps[-3:] for m in step]


def test_keep_recent_steps_is_a_no_op_below_the_window() -> None:
    agent = _agent(images=True)
    history = [m for i in range(2) for m in _image_step(i)]
    assert agent._keep_recent_steps(history, max_steps=5) == history
    assert agent._keep_recent_steps(history, max_steps=0) == history


def test_persistent_history_strips_images_and_keeps_only_the_last_n_steps() -> None:
    """Only the step being opened carries images; history keeps placeholders.

    Two boards cost ~2k tokens, so carrying them across the window would
    spend a third of a 32k budget re-sending boards already described in text.
    """
    agent = _agent(context_steps=2, images=True)
    system = {"role": "system", "content": "system prompt"}
    steps = [_image_step(i) for i in range(5)]
    persisted = agent._persistent_history_messages([system, *(m for s in steps for m in s)])

    # The last two steps survive, identical except that boards became placeholders.
    expected = [agent._strip_images_from_message(m) for s in steps[-2:] for m in s]
    assert persisted == expected
    parts = [p for m in persisted if isinstance(m.get("content"), list) for p in m["content"]]
    assert not any(p.get("type") == "image_url" for p in parts)
    assert sum(1 for p in parts if p.get("text") == _PLACEHOLDER) == 4  # 2 steps x 2 boards


def test_placeholder_still_marks_a_step_boundary() -> None:
    """Eviction must land on whole steps after the images are gone."""
    agent = _agent(context_steps=2, images=True)
    stripped = agent._strip_images_from_message(_image_step(0)[0])
    assert agent._is_history_turn_start(stripped)
    assert not agent._is_history_turn_start({"role": "user", "content": "You have not acted yet."})


def test_request_carries_two_images_regardless_of_window_size() -> None:
    agent = _agent(context_steps=4, images=True)
    system = {"role": "system", "content": "system prompt"}
    history: list[dict] = []
    for index in range(8):
        request = agent._trim_messages_for_context([system, *history, *_image_step(index)[:1]])
        images = [
            part
            for message in request
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        assert len(images) == 2, f"step {index} sent {len(images)} images"
        history = agent._persistent_history_messages([*request, *_image_step(index)[1:]])


def test_persistent_history_strips_images_in_legacy_token_only_mode() -> None:
    agent = _agent(context_steps=0, images=True)
    system = {"role": "system", "content": "system prompt"}
    persisted = agent._persistent_history_messages([system, *_image_step(0)])
    parts = [p for m in persisted if isinstance(m.get("content"), list) for p in m["content"]]
    assert not any(p.get("type") == "image_url" for p in parts)


# --- image cost in the token estimate ---------------------------------------


def test_images_are_costed_as_vision_tokens_not_base64_length() -> None:
    """A base64 board is ~7.3k chars but only 1024 vision tokens.

    Counting the data URL verbatim overshot by ~1.4k tokens per image, which
    at two images per step evicted steps that comfortably fit.
    """
    agent = _agent(images=True)
    blob = "data:image/png;base64," + "A" * 7300
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "prompt"},
                {"type": "image_url", "image_url": {"url": blob}},
                {"type": "image_url", "image_url": {"url": blob}},
            ],
        }
    ]
    estimate = agent._estimate_request_input_tokens(messages)
    assert 2 * 1024 <= estimate < 2 * 1024 + 200  # two images plus a little text
    assert estimate < len(blob) // 3  # nowhere near the base64 character count


# --- token-budget safety net ------------------------------------------------


def test_trim_messages_evicts_oldest_turns_first() -> None:
    agent = _agent()
    agent._context_budget_tokens = 1024
    system = {"role": "system", "content": "system prompt"}
    turns = [_turn(i) for i in range(40)]
    messages = [system, *(m for t in turns for m in t)]
    trimmed = agent._trim_messages_for_context(messages)
    assert trimmed[0] == system
    assert len(trimmed) < len(messages)
    survivors = trimmed[1:]
    assert survivors[0]["role"] == "user"
    flat = [m for t in turns for m in t]
    assert flat[len(flat) - len(survivors) :] == survivors
    assert agent._estimate_request_input_tokens(trimmed) <= 1024


def test_trim_messages_leaves_an_under_budget_request_alone() -> None:
    agent = _agent()
    agent._context_budget_tokens = 1024
    system = {"role": "system", "content": "system prompt"}
    small = [system, *_turn(0), *_turn(1)]
    assert agent._trim_messages_for_context(small) == small


def test_persistent_history_preserves_messages_verbatim() -> None:
    agent = _agent()
    system = {"role": "system", "content": "system prompt"}
    turns = [m for i in range(3) for m in _turn(i)]
    assert agent._persistent_history_messages([system, *turns]) == turns


# --- before/after frame selection and message layout ------------------------


@pytest.mark.parametrize(
    ("executed_count", "expected_fill"),
    [(1, 2), (2, 1), (3, 0)],
)
def test_step_boundary_frames_walks_back_by_executed_count(executed_count: int, expected_fill: int) -> None:
    agent = _agent()
    history = [HistoryEntry(action="DOWN", frame=_frame(fill, step=fill)) for fill in range(4)]
    current = history[-1].frame
    before, after = agent._step_boundary_frames(
        current,
        history,
        {"executed_count": executed_count, "executed_actions": ["DOWN"] * executed_count},
    )
    assert after is current
    assert before is not None and before.grid[0][0] == expected_fill


def test_step_boundary_frames_has_no_before_frame_on_the_first_step() -> None:
    agent = _agent()
    history = [HistoryEntry(action="", frame=_frame(0))]
    before, after = agent._step_boundary_frames(history[-1].frame, history, None)
    assert before is None
    assert after is history[-1].frame


def test_build_user_message_places_boards_inside_the_narrative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    agent = _agent(images=True)
    message = agent._build_user_message(
        "intro text",
        "rest text",
        before_frame=_frame(1),
        after_frame=_frame(2),
    )
    content = message["content"]
    assert message["role"] == "user"
    assert [part["type"] for part in content] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert content[0]["text"] == "intro text\n- Board before this sequence:"
    assert content[2]["text"] == "- Board after this sequence (the current board):"
    assert content[4]["text"] == "rest text"
    # Distinct boards render to distinct images.
    assert content[1]["image_url"]["url"] != content[3]["image_url"]["url"]


def test_transcript_rendering_marks_image_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    agent = _agent(images=True)
    message = agent._build_user_message(
        "intro text",
        "rest text",
        before_frame=_frame(1),
        after_frame=_frame(2),
    )
    assert _render_user_message_text(message) == (
        "intro text\n- Board before this sequence:\n"
        "[board image here]\n"
        "- Board after this sequence (the current board):\n"
        "[board image here]\n"
        "rest text"
    )
    # Text-only messages render verbatim.
    assert _render_user_message_text({"role": "user", "content": "plain"}) == "plain"


def test_build_user_message_falls_back_to_text_without_images() -> None:
    agent = _agent(images=False)
    message = agent._build_user_message("intro text", "rest text", before_frame=_frame(1), after_frame=_frame(2))
    assert message == {"role": "user", "content": "intro text\nrest text"}


def test_build_user_message_omits_before_board_on_the_first_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    agent = _agent(images=True)
    content = agent._build_user_message("intro text", "rest text", before_frame=None, after_frame=_frame(2))["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    assert content[0]["text"] == "intro text\n- Current board (no prior action to compare against):"
    assert content[2]["text"] == "rest text"
