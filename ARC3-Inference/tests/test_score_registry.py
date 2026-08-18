from __future__ import annotations

import json
from pathlib import Path

from inference.tools.score_registry import (
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
            "created_at": "2026-01-01T00:00:00Z",
        },
    }


def _write_score(root: Path, run_id: str, value: float) -> Path:
    path = root / "runs" / run_id / "score.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_score(run_id, value)), encoding="utf-8")
    return path


def test_update_appends_full_score_payload(tmp_path: Path) -> None:
    registry_path = tmp_path / "inference_score_comparison.json"
    score_path = _write_score(tmp_path, "run-a", 1.5)

    saved_path, run_id = update_score_registry(
        score_payload=_score("run-a", 1.5),
        score_path=score_path,
        registry_path=registry_path,
    )

    registry = json.loads(saved_path.read_text(encoding="utf-8"))
    assert run_id == "run-a"
    assert registry["version"] == 2
    assert registry["run_order"] == ["run-a"]
    assert registry["runs"]["run-a"]["score"]["score"] == 1.5
    assert registry["runs"]["run-a"]["score_path"] == "runs/run-a/score.json"


def test_update_replaces_existing_run_without_duplicating_order(tmp_path: Path) -> None:
    registry_path = tmp_path / "inference_score_comparison.json"
    score_path = _write_score(tmp_path, "run-a", 1.5)
    update_score_registry(
        score_payload=_score("run-a", 1.5),
        score_path=score_path,
        registry_path=registry_path,
    )
    before = load_score_registry(registry_path)["runs"]["run-a"]["added_at"]

    update_score_registry(
        score_payload=_score("run-a", 2.5),
        score_path=score_path,
        registry_path=registry_path,
    )

    registry = load_score_registry(registry_path)
    assert registry["run_order"] == ["run-a"]
    assert registry["runs"]["run-a"]["added_at"] == before
    assert registry["runs"]["run-a"]["score"]["score"] == 2.5


def test_v1_snapshot_migrates_source_score_files(tmp_path: Path) -> None:
    baseline = _write_score(tmp_path, "run-a", 1.5)
    candidate = _write_score(tmp_path, "run-b", 2.5)
    registry_path = tmp_path / "inference_score_comparison.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "baseline_run": "run-a",
                "candidate_run": "run-b",
                "runs": {
                    "baseline": {"source_score_json": str(baseline.relative_to(tmp_path))},
                    "candidate": {"source_score_json": str(candidate.relative_to(tmp_path))},
                },
            }
        ),
        encoding="utf-8",
    )

    registry = load_score_registry(registry_path)

    assert registry["version"] == 2
    assert registry["run_order"] == ["run-a", "run-b"]
    assert registry["runs"]["run-a"]["score"]["score"] == 1.5
    assert registry["runs"]["run-b"]["score"]["score"] == 2.5
