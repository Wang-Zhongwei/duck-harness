import pytest

from inference.distillation.teacher import (
    _extract_prompt_logprobs,
    _last_subsequence_start,
    _teacher_messages,
)


def test_extracts_current_vllm_prompt_logprobs():
    payload = {
        "prompt_logprobs": [
            None,
            {"11": {"logprob": -1.25, "rank": 2}},
            {"12": {"logprob": -0.5, "rank": 1}},
        ]
    }
    assert _extract_prompt_logprobs(payload, [10, 11, 12]) == [None, -1.25, -0.5]


def test_extracts_legacy_echo_logprobs():
    payload = {"choices": [{"logprobs": {"token_logprobs": [None, -1.0, -2.0, -3.0]}}]}
    assert _extract_prompt_logprobs(payload, [10, 11, 12]) == [None, -1.0, -2.0]


def test_rejects_missing_sampled_token():
    payload = {"prompt_logprobs": [None, {"99": {"logprob": -1.0}}]}
    with pytest.raises(ValueError, match="omitted"):
        _extract_prompt_logprobs(payload, [10, 11])


def test_finds_last_exact_student_output_in_rendered_teacher_chat():
    assert _last_subsequence_start([1, 2, 3, 2, 3, 4], [2, 3]) == 3
    assert _last_subsequence_start([1, 2], [3]) is None


def test_teacher_inlines_openai_tool_call_without_changing_serialization():
    row = {
        "messages": [{"role": "user", "content": "state"}],
        "assistant_message": {
            "role": "assistant",
            "content": None,
            "reasoning": "inspect",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": '{"code":"print(1)"}',
                    },
                }
            ],
        },
    }
    message = _teacher_messages(row)[-1]
    assert "tool_calls" not in message
    assert message["reasoning_content"] == "inspect"
    assert message["content"] == (
        "<tool_call>\n"
        "<function=python>\n"
        "<parameter=code>\n"
        "print(1)\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
