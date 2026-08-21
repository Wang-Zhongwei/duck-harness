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

    for run_dir in (old_run, first_qwen38_run, latest_displayable_run):
        _write_viewer_artifact(run_dir)
    empty_newer_run.mkdir()

    assert list_run_dirs(tmp_path) == [latest_displayable_run, first_qwen38_run]
    assert find_latest_run_dir(tmp_path) == latest_displayable_run


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
    score_entry = {"scores": {"public": {"score": {"score": 1.0}}}}
    first_qwen38_run = "20260810_000000_q38"
    unevaluated_run = "20260819_120000_q38"
    distill_run = "20260819_130000_distill"
    latest_qwen38_run = "20260820_000000_q38"
    run_order = [first_qwen38_run, unevaluated_run, distill_run, latest_qwen38_run]
    for run_id in (first_qwen38_run, unevaluated_run, latest_qwen38_run):
        _write_viewer_artifact(runs_dir / run_id)
    for run_id in (first_qwen38_run, distill_run, latest_qwen38_run):
        run_dir = runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        (run_dir / "evaluation.json").write_text("{}", encoding="utf-8")
    registry = {"version": 3, "minimum_run_date": "20260810", "run_order": run_order, "runs": {run_id: score_entry for run_id in run_order}}
    (runs_dir / "inference_score_comparison.json").write_text(json.dumps(registry), encoding="utf-8")

    payload = _load_comparison_payload(runs_dir)

    assert payload["run_order"] == [first_qwen38_run, latest_qwen38_run]
    assert set(payload["runs"]) == {first_qwen38_run, latest_qwen38_run}
