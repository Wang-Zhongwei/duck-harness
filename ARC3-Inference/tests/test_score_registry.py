from __future__ import annotations

import json
from pathlib import Path

from inference.tools.score_registry import (
    DEFAULT_REGISTRY_PATH,
    append_semi_private_score,
    load_score_registry,
    update_score_registry,
)


def _score(run_id: str, value: float) -> dict:
    return {
        "version": 3,
        "score": value,
        "games": {"game-1": {"score": value}},
        "usage": {"mean": {"actions": 1, "turns": 2, "total_tokens": 3}},
        "server": {},
        "metadata": {
            "experiment_dirs": [f"runs/{run_id}"],
            "created_at": "2026-08-15T00:00:00Z",
        },
    }


def _write_score(root: Path, run_id: str, value: float) -> Path:
    path = root / "runs" / run_id / "score.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_score(run_id, value)), encoding="utf-8")
    return path


def test_default_registry_lives_under_runs() -> None:
    assert DEFAULT_REGISTRY_PATH.parent.name == "runs"


def test_update_appends_full_public_score_payload(tmp_path: Path) -> None:
    run_id = "20260815_run-a"
    registry_path = tmp_path / "runs" / "inference_score_comparison.json"
    score_path = _write_score(tmp_path, run_id, 1.5)

    saved_path, saved_run_id = update_score_registry(
        score_payload=_score(run_id, 1.5),
        score_path=score_path,
        registry_path=registry_path,
    )

    registry = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_run_id == run_id
    assert registry["version"] == 3
    assert registry["minimum_run_date"] == "20260810"
    assert registry["run_order"] == [run_id]
    public = registry["runs"][run_id]["scores"]["public"]
    assert public["score"]["score"] == 1.5
    assert public["score_path"] == f"{run_id}/score.json"


def test_update_replaces_same_public_score_without_duplicating_order(tmp_path: Path) -> None:
    run_id = "20260815_run-a"
    registry_path = tmp_path / "runs" / "inference_score_comparison.json"
    score_path = _write_score(tmp_path, run_id, 1.5)
    update_score_registry(
        score_payload=_score(run_id, 1.5),
        score_path=score_path,
        registry_path=registry_path,
    )
    before = load_score_registry(registry_path)["runs"][run_id]["scores"]["public"]["added_at"]

    update_score_registry(
        score_payload=_score(run_id, 2.5),
        score_path=score_path,
        registry_path=registry_path,
    )

    registry = load_score_registry(registry_path)
    public = registry["runs"][run_id]["scores"]["public"]
    assert registry["run_order"] == [run_id]
    assert public["added_at"] == before
    assert public["score"]["score"] == 2.5


def test_old_runs_are_excluded_from_registry(tmp_path: Path) -> None:
    run_id = "20260809_old-run"
    registry_path = tmp_path / "runs" / "inference_score_comparison.json"
    score_path = _write_score(tmp_path, run_id, 1.5)

    update_score_registry(
        score_payload=_score(run_id, 1.5),
        score_path=score_path,
        registry_path=registry_path,
    )

    registry = load_score_registry(registry_path)
    assert registry["run_order"] == []
    assert registry["runs"] == {}


def test_semi_private_imports_append_with_blank_manual_link(tmp_path: Path) -> None:
    registry_path = tmp_path / "runs" / "inference_score_comparison.json"
    score_path = tmp_path / "runs" / "semi_private_scores" / "first.json"
    score_path.parent.mkdir(parents=True)
    payload = _score("remote-kaggle-run", 4.25)
    score_path.write_text(json.dumps(payload), encoding="utf-8")

    _, first_id = append_semi_private_score(
        score_payload=payload,
        score_path=score_path,
        source_label="score.json",
        registry_path=registry_path,
    )
    _, second_id = append_semi_private_score(
        score_payload=payload,
        score_path=score_path,
        source_label="score.json",
        registry_path=registry_path,
    )

    registry = load_score_registry(registry_path)
    imports = registry["semi_private_scores"]
    assert first_id != second_id
    assert len(imports) == 2
    assert all(item["local_run_path"] == "" for item in imports)
    assert all(item["score"]["score"] == 4.25 for item in imports)


def test_v2_registry_migrates_embedded_scores_to_public_set(tmp_path: Path) -> None:
    run_a = "20260815_run-a"
    run_b = "20260816_run-b"
    registry_path = tmp_path / "runs" / "inference_score_comparison.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "version": 2,
                "run_order": [run_a, run_b],
                "runs": {
                    run_a: {
                        "run_id": run_a,
                        "score_path": f"runs/{run_a}/score.json",
                        "score": _score(run_a, 1.5),
                    },
                    run_b: {
                        "run_id": run_b,
                        "score_path": f"runs/{run_b}/score.json",
                        "score": _score(run_b, 2.5),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    registry = load_score_registry(registry_path)

    assert registry["version"] == 3
    assert registry["run_order"] == [run_a, run_b]
    assert registry["runs"][run_a]["scores"]["public"]["score"]["score"] == 1.5
    assert registry["runs"][run_b]["scores"]["public"]["score"]["score"] == 2.5
