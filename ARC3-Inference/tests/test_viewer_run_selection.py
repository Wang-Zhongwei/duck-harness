import json
from pathlib import Path

from viewer.data import find_latest_run_dir, list_run_dirs, load_run_summary
from viewer.server import _load_comparison_payload


def _write_viewer_artifact(run_dir: Path) -> None:
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "ar25-test_p0_viewer_data.json").write_text("{}", encoding="utf-8")


def test_auto_selection_uses_latest_displayable_qwen38_era_run(tmp_path: Path) -> None:
    old_run = tmp_path / "20260809_235959_old"
    first_qwen38_run = tmp_path / "20260810_000000_q38"
    latest_displayable_run = tmp_path / "20260818_195342_q38"
    empty_newer_run = tmp_path / "20260819_062601_q38"
    evaluated_run = tmp_path / "20260820_081324_q38"

    for run_dir in (old_run, first_qwen38_run, latest_displayable_run):
        _write_viewer_artifact(run_dir)
    empty_newer_run.mkdir()
    evaluated_run.mkdir()
    (evaluated_run / "evaluation.json").write_text("{}", encoding="utf-8")

    assert list_run_dirs(tmp_path) == [evaluated_run, latest_displayable_run, first_qwen38_run]
    assert find_latest_run_dir(tmp_path) == evaluated_run


def test_explicit_run_directory_remains_available_as_override(tmp_path: Path) -> None:
    old_run = tmp_path / "20260809_235959_old"
    _write_viewer_artifact(old_run)

    assert list_run_dirs(old_run) == [old_run]
    assert find_latest_run_dir(old_run) == old_run


def test_run_summary_exposes_evaluation_details(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260820_120000_scored"
    _write_viewer_artifact(run_dir)
    evaluation = {"score": 2.5, "games": {"game-1": {"total_tokens": {"pass-0": 120}}}}
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    payload = load_run_summary(runs_dir=run_dir)

    assert payload["evaluation"]["source_file"] == "evaluation.json"
    assert payload["evaluation"]["games"]["game-1"]["total_tokens"]["pass-0"] == 120


def test_run_viewer_renders_results_and_switches_to_trace() -> None:
    html = (Path(__file__).parents[1] / "viewer" / "index.html").read_text(encoding="utf-8")

    for element_id in ("results-view", "score-kpis", "score-matrix", "game-inspector", "pass-breakdown"):
        assert f'id="{element_id}"' in html
    assert 'openTrace(selectedResultGame, target.dataset.resultTrial)' in html
    assert "gameSelectEl.hidden = showingResults" in html
    assert "switchView(currentView);" in html
    assert html.index('const showingResults = currentView === "results";') < html.index('classList.toggle("results-mode", showingResults)')
    assert 'reload-button' not in html


def test_comparison_payload_uses_evaluated_homepage_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    score_entry = {"added_at": "2026-08-20T00:00:00Z", "scores": {"public": {"score": {"score": 1.0}}}}
    first_qwen38_run = "20260810_000000_q38"
    unevaluated_run = "20260819_120000_q38"
    distill_run = "20260819_130000_distill"
    latest_qwen38_run = "20260820_000000_q38"
    renamed_registry_run = "20260821_000000_old-name"
    run_order = [first_qwen38_run, unevaluated_run, distill_run, latest_qwen38_run, renamed_registry_run]
    for run_id in (first_qwen38_run, unevaluated_run, latest_qwen38_run):
        _write_viewer_artifact(runs_dir / run_id)
    for run_id in (first_qwen38_run, distill_run, latest_qwen38_run):
        run_dir = runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        (run_dir / "evaluation.json").write_text(json.dumps({"score": 1.0}), encoding="utf-8")
        bundled_root = run_dir if run_id == distill_run else run_dir / "passes" / "0"
        inference_path = bundled_root / "src" / "ARC3-Inference" / "configs" / "inference.json"
        inference_path.parent.mkdir(parents=True)
        inference_path.write_text(
            json.dumps({"chat": {"temperature": 0.5 if run_id == latest_qwen38_run else 1.0}}),
            encoding="utf-8",
        )
    (runs_dir / latest_qwen38_run / "evaluation.json").write_text(
        json.dumps({
            "score": 2.5,
            "metadata": {"created_at": "2026-08-22T10:00:00Z"},
            "server": {"per_server": [{"sample_count": 3}]},
        }),
        encoding="utf-8",
    )
    registry = {"version": 3, "minimum_run_date": "20260810", "run_order": run_order, "runs": {run_id: score_entry for run_id in run_order}}
    (runs_dir / "inference_score_comparison.json").write_text(json.dumps(registry), encoding="utf-8")

    payload = _load_comparison_payload(runs_dir)

    assert payload["run_order"] == [latest_qwen38_run, distill_run, first_qwen38_run]
    assert set(payload["runs"]) == {first_qwen38_run, distill_run, latest_qwen38_run}
    assert payload["runs"][latest_qwen38_run]["inference_config"]["chat"]["temperature"] == 0.5
    latest_public = payload["runs"][latest_qwen38_run]["scores"]["public"]
    assert latest_public["score"]["score"] == 2.5
    assert latest_public["score_path"] == f"{latest_qwen38_run}/evaluation.json"
    assert latest_public["updated_at"] == "2026-08-22T10:00:00Z"
    assert payload["runs"][latest_qwen38_run]["added_at"] == "2026-08-20T00:00:00Z"
    assert payload["runs"][first_qwen38_run]["scores"]["public"]["score"] == {"score": 1.0}


def test_comparison_payload_lists_evaluated_runs_missing_from_registry(tmp_path: Path) -> None:
    """A renamed or never-registered run is comparable as soon as evaluation.json exists."""
    runs_dir = tmp_path / "runs"
    registered_run = "20260815_224235_q38-registered"
    unregistered_run = "20260821_191552_my-user-prompt-mtp3-vllm"
    for run_id, score in ((registered_run, 3.2), (unregistered_run, 5.16)):
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "evaluation.json").write_text(json.dumps({"score": score, "games": {}}), encoding="utf-8")
    registry = {
        "version": 3,
        "minimum_run_date": "20260810",
        "run_order": [registered_run, "20260821_191552_level-review-fixed-mtp3-vllm"],
        "runs": {
            registered_run: {"scores": {"public": {"score": {"score": 0.0}}}},
            "20260821_191552_level-review-fixed-mtp3-vllm": {"scores": {"public": {"score": {"score": 5.16}}}},
        },
        "semi_private_scores": [{"id": "import-1", "local_run_path": f"runs/{registered_run}", "score": {"score": 1.1}}],
        "public_frontier": {"configs": {}},
    }
    (runs_dir / "inference_score_comparison.json").write_text(json.dumps(registry), encoding="utf-8")

    payload = _load_comparison_payload(runs_dir)

    assert payload["run_order"] == [unregistered_run, registered_run]
    assert payload["runs"][unregistered_run]["scores"]["public"]["score"]["score"] == 5.16
    assert payload["runs"][registered_run]["scores"]["public"]["score"]["score"] == 3.2
    assert payload["semi_private_scores"][0]["id"] == "import-1"
    assert payload["public_frontier"] == {"configs": {}}


def test_comparison_payload_works_without_registry_and_skips_broken_evaluations(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    good_run = "20260820_000000_good"
    broken_run = "20260821_000000_broken"
    for run_id, body in ((good_run, json.dumps({"score": 4.0})), (broken_run, "{not json")):
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "evaluation.json").write_text(body, encoding="utf-8")

    payload = _load_comparison_payload(runs_dir)

    assert payload["version"] == 3
    assert payload["run_order"] == [good_run]
    assert payload["runs"][good_run]["scores"]["public"]["score"]["score"] == 4.0
    assert "score_sets" in payload


def test_comparison_defaults_to_previous_run_and_newest_run() -> None:
    html = (Path(__file__).parents[1] / "viewer" / "comparison.html").read_text(encoding="utf-8")

    assert "baselineId = order[1];" in html
    assert "candidateId = order[0];" in html
