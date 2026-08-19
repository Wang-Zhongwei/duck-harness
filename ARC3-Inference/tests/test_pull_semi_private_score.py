from __future__ import annotations

import json
from pathlib import Path

from inference.tools.pull_semi_private_score import (
    append_kaggle_submissions,
    parse_kaggle_submissions_csv,
    pull_score_payload,
    save_semi_private_score,
)
from inference.tools.score_registry import load_score_registry


def test_pull_local_score_and_append_unique_unlinked_items(tmp_path: Path) -> None:
    source = tmp_path / "remote-score.json"
    payload = {"score": 7.5, "games": {"ar25": {"score": 7.5}}}
    source.write_text(json.dumps(payload), encoding="utf-8")

    loaded = pull_score_payload(str(source))
    first_path, registry_path, first_id = save_semi_private_score(
        payload=loaded,
        source_label=source.name,
        runs_dir=tmp_path / "runs",
    )
    second_path, _, second_id = save_semi_private_score(
        payload=loaded,
        source_label=source.name,
        runs_dir=tmp_path / "runs",
    )

    registry = load_score_registry(registry_path)
    assert first_path != second_path
    assert first_id != second_id
    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")
    assert len(registry["semi_private_scores"]) == 2
    assert all(item["local_run_path"] == "" for item in registry["semi_private_scores"])



def test_kaggle_pull_adds_only_unseen_post_cutoff_submissions(tmp_path: Path) -> None:
    rows = [
        {
            "ref": "old",
            "fileName": "submission.parquet",
            "date": "2026-08-03 00:00:00",
            "description": "",
            "status": "SubmissionStatus.COMPLETE",
            "publicScore": "0.95",
            "privateScore": "",
        },
        {
            "ref": "one",
            "fileName": "submission.parquet",
            "date": "2026-08-17 14:50:50",
            "description": "65k",
            "status": "SubmissionStatus.COMPLETE",
            "publicScore": "1.37",
            "privateScore": "",
        },
        {
            "ref": "two",
            "fileName": "submission.parquet",
            "date": "2026-08-18 14:32:33",
            "description": "32k",
            "status": "SubmissionStatus.COMPLETE",
            "publicScore": "2.59",
            "privateScore": "",
        },
    ]
    runs_dir = tmp_path / "runs"

    added, skipped = append_kaggle_submissions(
        rows=rows,
        competition="arc-test",
        minimum_date="2026-08-10",
        runs_dir=runs_dir,
    )
    added_again, skipped_again = append_kaggle_submissions(
        rows=rows,
        competition="arc-test",
        minimum_date="2026-08-10",
        runs_dir=runs_dir,
    )

    registry = load_score_registry(runs_dir / "inference_score_comparison.json")
    imports = registry["semi_private_scores"]
    assert len(added) == 2
    assert skipped == []
    assert added_again == []
    assert skipped_again == ["one", "two"]
    assert [item["score"]["score"] for item in imports] == [1.37, 2.59]
    assert all(item["local_run_path"] == "" for item in imports)
    assert not (runs_dir / "semi_private_scores" / "kaggle-old.json").exists()



def test_parse_kaggle_csv_ignores_cli_warning_preamble() -> None:
    rows = parse_kaggle_submissions_csv(
        "Warning: outdated client\n"
        "ref,fileName,date,description,status,publicScore,privateScore\n"
        "123,submission.parquet,2026-08-18 00:00:00,test,SubmissionStatus.COMPLETE,2.59,\n"
    )

    assert rows == [
        {
            "ref": "123",
            "fileName": "submission.parquet",
            "date": "2026-08-18 00:00:00",
            "description": "test",
            "status": "SubmissionStatus.COMPLETE",
            "publicScore": "2.59",
            "privateScore": "",
        }
    ]
