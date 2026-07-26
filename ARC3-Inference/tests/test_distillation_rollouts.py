import json
from pathlib import Path

import pytest

from inference.distillation.rollouts import load_game_episodes


def _row(episode: str, output: list[int], student: list[float], teacher: list[float]):
    return {
        "episode_id": episode,
        "policy_id": "iteration-0000",
        "messages": [{"role": "user", "content": "game state"}],
        "tools": [],
        "prompt_token_ids": [1, 2],
        "output_token_ids": output,
        "output_logprobs": {
            "content": [
                {"token": f"token_id:{token}", "logprob": value}
                for token, value in zip(output, student, strict=True)
            ]
        },
        "teacher_logprobs": teacher,
    }


def test_reward_to_go_crosses_game_turn_boundaries(tmp_path: Path) -> None:
    rows = [
        _row("game-1", [10, 11], [-2.0, -2.0], [-1.0, -1.5]),
        _row("game-1", [12], [-3.0], [-1.0]),
    ]
    path = tmp_path / "game-1.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    episode = load_game_episodes(tmp_path)[0]

    assert episode[0].advantages == pytest.approx([3.5, 2.5])
    assert episode[1].advantages == pytest.approx([2.0])


def test_update_batch_rejects_mixed_behavior_policies(tmp_path: Path) -> None:
    first = _row("game-1", [10], [-2.0], [-1.0])
    second = _row("game-2", [11], [-2.0], [-1.0])
    second["policy_id"] = "iteration-0001"
    (tmp_path / "games.jsonl").write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one behavior policy"):
        load_game_episodes(tmp_path)
