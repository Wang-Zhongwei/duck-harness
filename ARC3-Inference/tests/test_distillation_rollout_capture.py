import json
from pathlib import Path

from inference.agent import tool_agent
from inference.agent.tool_agent import _ChatCompletionResult, _append_distill_rollout


def test_capture_persists_exact_harness_request_and_token_ids(
    tmp_path: Path, monkeypatch
) -> None:
    rollout_dir = tmp_path / "rollouts"
    monkeypatch.setattr(tool_agent, "_DISTILL_ROLLOUT_DIR", str(rollout_dir))
    state_path = tmp_path / "game-pass-0_tool_runtime_state.json"
    result = _ChatCompletionResult(
        message={
            "role": "assistant",
            "reasoning": "inspect",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": '{"code":"action([\'LEFT\'])"}',
                    },
                }
            ],
        },
        finish_reason="tool_calls",
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        prompt_token_ids=[10, 11, 12],
        output_token_ids=[20, 21],
        logprobs={"content": [{"token": "token_id:20", "logprob": -0.5}]},
    )

    _append_distill_rollout(
        state_path,
        policy_id="adapter-step-0",
        model_id="vrfai/Qwen3.6-27B-FP8",
        messages=[
            {"role": "system", "content": "play"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "current board"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
        ],
        tools=[{"type": "function", "function": {"name": "python"}}],
        tool_choice="auto",
        result=result,
        analysis_step=1,
        action=1,
        request_index_within_turn=1,
    )

    rows = [
        json.loads(line)
        for line in (rollout_dir / "game-pass-0_tool_runtime_state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["episode_id"] == "game-pass-0_tool_runtime_state"
    assert row["policy_id"] == "adapter-step-0"
    assert row["prompt_token_ids"] == [10, 11, 12]
    assert row["output_token_ids"] == [20, 21]
    assert row["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_capture_rejects_responses_without_exact_token_ids(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(tool_agent, "_DISTILL_ROLLOUT_DIR", str(tmp_path))
    result = _ChatCompletionResult(message={"role": "assistant", "content": "x"})

    try:
        _append_distill_rollout(
            tmp_path / "state.json",
            policy_id="step-0",
            model_id="student",
            messages=[{"role": "user", "content": "state"}],
            tools=[],
            tool_choice=None,
            result=result,
            analysis_step=1,
            action=1,
            request_index_within_turn=1,
        )
    except RuntimeError as exc:
        assert "token_ids" in str(exc)
    else:
        raise AssertionError("capture accepted a response without token IDs")
