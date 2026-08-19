from pathlib import Path

from viewer.data import find_latest_run_dir, list_run_dirs


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
