from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import time
from typing import Any

import requests


def _logprob_value(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    raise ValueError(f"unrecognized logprob value: {value!r}")


def _extract_prompt_logprobs(
    payload: dict[str, Any], token_ids: list[int]
) -> list[float | None]:
    """Handle current vLLM and legacy OpenAI-completions response shapes."""
    prompt_logprobs = payload.get("prompt_logprobs")
    if prompt_logprobs is not None:
        if len(prompt_logprobs) < len(token_ids):
            raise ValueError(
                "teacher returned fewer prompt logprobs than prompt tokens"
            )
        result: list[float | None] = []
        for token_id, candidates in zip(
            token_ids, prompt_logprobs[: len(token_ids)], strict=True
        ):
            if candidates is None:
                result.append(None)
                continue
            candidate = candidates.get(str(token_id), candidates.get(token_id))
            if candidate is None:
                raise ValueError(
                    f"teacher omitted the sampled prompt token id {token_id}"
                )
            result.append(_logprob_value(candidate))
        return result

    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("teacher response has no choices")
    logprobs = (choices[0].get("logprobs") or {}).get("token_logprobs")
    if logprobs is None or len(logprobs) < len(token_ids):
        raise ValueError("teacher response has no usable prompt token logprobs")
    return [
        None if value is None else float(value) for value in logprobs[: len(token_ids)]
    ]


def _teacher_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for message in [*row["messages"], row["assistant_message"]]:
        normalized = dict(message)
        if (
            isinstance(normalized.get("reasoning"), str)
            and "reasoning_content" not in normalized
        ):
            normalized["reasoning_content"] = normalized["reasoning"]
        if isinstance(normalized.get("tool_calls"), list):
            rendered_calls = []
            for tool_call in normalized["tool_calls"]:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                rendered = [f"<tool_call>\n<function={function['name']}>\n"]
                for name, value in arguments.items():
                    if not isinstance(value, str):
                        value = json.dumps(value, ensure_ascii=False)
                    rendered.extend(
                        [
                            f"<parameter={name}>\n",
                            value,
                            "\n</parameter>\n",
                        ]
                    )
                rendered.append("</function>\n</tool_call>")
                rendered_calls.append("".join(rendered))
            content = normalized.get("content") or ""
            separator = "\n\n" if content.strip() else ""
            normalized["content"] = content + separator + "\n".join(rendered_calls)
            normalized.pop("tool_calls")
        messages.append(normalized)
    return messages


@dataclass(frozen=True)
class VllmTeacherScorer:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_seconds: float = 600.0
    concurrency: int = 8
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    @property
    def completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/completions"

    def score(self, token_ids: list[int], completion_start: int) -> list[float]:
        if not 0 < completion_start < len(token_ids):
            raise ValueError(
                "completion_start must leave a non-empty prefix and completion"
            )
        request = {
            "model": self.model,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "echo": True,
            "logprobs": 1,
            "prompt_logprobs": 1,
            "return_tokens_as_token_ids": True,
        }
        response = None
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.completions_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=request,
                    timeout=self.timeout_seconds,
                )
                if response.ok or (
                    response.status_code < 500 and response.status_code != 429
                ):
                    break
                last_error = RuntimeError(
                    f"teacher request failed ({response.status_code}): {response.text[:1000]}"
                )
            except requests.RequestException as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        if response is None or not response.ok:
            if last_error is not None:
                raise RuntimeError(
                    "teacher request failed after retries"
                ) from last_error
            raise RuntimeError(
                f"teacher request failed ({response.status_code}): {response.text[:1000]}"
            )
        all_logprobs = _extract_prompt_logprobs(response.json(), token_ids)
        result = all_logprobs[completion_start:]
        if any(value is None for value in result):
            raise ValueError("teacher returned a null logprob inside the completion")
        return [float(value) for value in result]

    def score_batch(
        self,
        token_id_batches: list[list[int]],
        completion_starts: list[int],
    ) -> list[list[float]]:
        if len(token_id_batches) != len(completion_starts):
            raise ValueError(
                "token batches and completion starts must have equal lengths"
            )
        if not token_id_batches:
            return []
        with ThreadPoolExecutor(
            max_workers=min(self.concurrency, len(token_id_batches))
        ) as pool:
            futures = [
                pool.submit(self.score, token_ids, completion_start)
                for token_ids, completion_start in zip(
                    token_id_batches, completion_starts, strict=True
                )
            ]
            return [future.result() for future in futures]

    def score_game_turn(self, row: dict[str, Any]) -> list[float]:
        """Score one captured assistant response with its multimodal chat context."""
        output_ids = [int(value) for value in row["output_token_ids"]]
        request = {
            "model": self.model,
            "messages": _teacher_messages(row),
            "tools": row.get("tools") or None,
            "tool_choice": "none",
            "max_tokens": 1,
            "temperature": 0.0,
            "add_generation_prompt": False,
            "prompt_logprobs": 1,
            "return_token_ids": True,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        }
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=request,
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(
                f"teacher request failed ({response.status_code}): "
                f"{response.text[:2000]}"
            )
        payload = response.json()
        prompt_ids = [int(value) for value in payload.get("prompt_token_ids") or []]
        start = _last_subsequence_start(prompt_ids, output_ids)
        if start is None:
            raise ValueError(
                "teacher chat rendering did not preserve the student's exact output token IDs"
            )
        prompt_logprobs = _extract_prompt_logprobs(payload, prompt_ids)
        selected = prompt_logprobs[start : start + len(output_ids)]
        if any(value is None for value in selected):
            raise ValueError("teacher returned null logprobs inside student output")
        return [float(value) for value in selected]


def _last_subsequence_start(values: list[int], subsequence: list[int]) -> int | None:
    if not subsequence or len(subsequence) > len(values):
        return None
    for start in range(len(values) - len(subsequence), -1, -1):
        if values[start : start + len(subsequence)] == subsequence:
            return start
    return None
