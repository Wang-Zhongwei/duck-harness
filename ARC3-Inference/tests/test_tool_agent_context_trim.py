"""Turn-boundary eviction and ephemeral image-message behavior.

Evictions must land on whole-turn boundaries (never orphaning tool results)
and the grid image must live in a separate trailing message so persisted
history is byte-identical to what was sent -- both are what keep the vLLM
prefix (KV) cache valid across turns.
"""

from inference.agent.tool_agent import ToolAgent


def _agent() -> ToolAgent:
    return ToolAgent(
        model="test-model",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )


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


_IMAGE_MESSAGE = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Current grid image (same board as `current_frame`):"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ],
}


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


def test_image_message_is_not_a_turn_start() -> None:
    agent = _agent()
    assert not agent._is_history_turn_start(_IMAGE_MESSAGE)
    assert agent._is_history_turn_start({"role": "user", "content": "plain text"})
    assert agent._is_history_turn_start(
        {"role": "user", "content": [{"type": "text", "text": "text-only parts"}]}
    )


def test_drop_oldest_history_turn_keeps_image_with_its_turn() -> None:
    agent = _agent()
    current_turn = [{"role": "user", "content": "current prompt"}, dict(_IMAGE_MESSAGE)]
    history = [*_turn(0), *current_turn]
    assert agent._drop_oldest_history_turn(history)
    assert history == current_turn
    # The [user, image] pair is the final turn; it must never be split or dropped.
    assert not agent._drop_oldest_history_turn(history)
    assert history == current_turn


def test_trim_messages_evicts_oldest_turns_first() -> None:
    agent = _agent()
    agent._context_budget_tokens = 1024
    system = {"role": "system", "content": "system prompt"}
    turns = [_turn(i) for i in range(40)]
    messages = [system, *(m for t in turns for m in t)]
    trimmed = agent._trim_messages_for_context(messages)
    assert trimmed[0] == system
    assert len(trimmed) < len(messages)
    # Whatever survives is a suffix of the original turn sequence.
    survivors = trimmed[1:]
    assert survivors[0]["role"] == "user"
    flat = [m for t in turns for m in t]
    assert flat[len(flat) - len(survivors):] == survivors
    assert agent._estimate_request_input_tokens(trimmed) <= 1024


def test_trim_messages_low_water_trims_past_budget() -> None:
    agent = _agent()
    agent._context_budget_tokens = 1024
    agent._trim_low_water_tokens = 512
    system = {"role": "system", "content": "system prompt"}
    messages = [system, *(m for i in range(40) for m in _turn(i))]
    trimmed = agent._trim_messages_for_context(messages)
    assert agent._estimate_request_input_tokens(trimmed) <= 1024 - 512 + 64  # one turn of slack

    # Under budget: nothing is evicted even with a low-water mark set.
    small = [system, *_turn(0), *_turn(1)]
    assert agent._trim_messages_for_context(small) == small


def test_persistent_history_preserves_messages_verbatim() -> None:
    agent = _agent()
    system = {"role": "system", "content": "system prompt"}
    turns = [m for i in range(3) for m in _turn(i)]
    persisted = agent._persistent_history_messages([system, *turns])
    assert persisted == turns
