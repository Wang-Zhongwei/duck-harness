from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GameTurn:
    episode_id: str
    policy_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    student_logprobs: list[float]
    teacher_logprobs: list[float]
    advantages: list[float]


def _sampled_logprobs(payload: dict[str, Any] | None) -> list[float]:
    content = (payload or {}).get("content")
    if not isinstance(content, list):
        raise ValueError("rollout has no vLLM output_logprobs.content")
    return [float(item["logprob"]) for item in content]


def load_game_episodes(
    path: str | Path,
    *,
    gamma: float = 1.0,
    max_sequence_tokens: int = 0,
) -> list[list[GameTurn]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    policy_ids: set[str] = set()
    skipped_long = 0
    for file in sorted(Path(path).glob("*.jsonl")):
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                # A turn cut off at the token cap emitted no tool call and took
                # no game action; imitating it teaches the student to stop
                # mid-thought.
                if row.get("finish_reason") == "length":
                    continue
                # Peak training memory is set by the longest single turn: the
                # backward graph scales with sequence length, and one 20k-token
                # turn can exhaust an 80 GiB card on top of the frozen base.
                if max_sequence_tokens > 0 and (
                    len(row["prompt_token_ids"]) + len(row["output_token_ids"])
                    > max_sequence_tokens
                ):
                    skipped_long += 1
                    continue
                grouped.setdefault(str(row["episode_id"]), []).append(row)
                policy_ids.add(str(row["policy_id"]))
    if skipped_long:
        print(
            json.dumps(
                {
                    "event": "turns_skipped_too_long",
                    "turns": skipped_long,
                    "max_sequence_tokens": max_sequence_tokens,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not grouped:
        raise ValueError(f"{path}: no rollout JSONL records found")
    if len(policy_ids) != 1:
        raise ValueError(
            "one optimizer update requires exactly one behavior policy; found "
            + ", ".join(sorted(policy_ids))
        )

    episodes: list[list[GameTurn]] = []
    for episode_id, rows in grouped.items():
        flat_rewards: list[float] = []
        lengths: list[int] = []
        for row in rows:
            output_ids = [int(value) for value in row["output_token_ids"]]
            student = _sampled_logprobs(row.get("output_logprobs"))
            teacher = [float(value) for value in row["teacher_logprobs"]]
            if not len(output_ids) == len(student) == len(teacher):
                raise ValueError(f"{episode_id}: token/logprob lengths differ")
            lengths.append(len(output_ids))
            flat_rewards.extend(q - p for p, q in zip(student, teacher, strict=True))

        running = 0.0
        flat_advantages = [0.0] * len(flat_rewards)
        for index in range(len(flat_rewards) - 1, -1, -1):
            running = flat_rewards[index] + gamma * running
            flat_advantages[index] = running

        turns: list[GameTurn] = []
        offset = 0
        for row, length in zip(rows, lengths, strict=True):
            turns.append(
                GameTurn(
                    episode_id=episode_id,
                    policy_id=str(row["policy_id"]),
                    messages=row["messages"],
                    tools=row.get("tools") or [],
                    prompt_token_ids=[int(value) for value in row["prompt_token_ids"]],
                    output_token_ids=[int(value) for value in row["output_token_ids"]],
                    student_logprobs=_sampled_logprobs(row.get("output_logprobs")),
                    teacher_logprobs=[
                        float(value) for value in row["teacher_logprobs"]
                    ],
                    advantages=flat_advantages[offset : offset + length],
                )
            )
            offset += length
        episodes.append(turns)
    return episodes
