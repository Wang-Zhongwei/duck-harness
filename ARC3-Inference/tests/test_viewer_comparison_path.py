from pathlib import Path

from viewer.server import _comparison_json_path


def test_comparison_registry_follows_configured_runs_dir(tmp_path: Path) -> None:
    assert _comparison_json_path(tmp_path) == tmp_path / "inference_score_comparison.json"


def test_game_comparison_exposes_sortable_linked_game_ids() -> None:
    html = (Path(__file__).parents[1] / "viewer" / "comparison.html").read_text(encoding="utf-8")

    for sort_key in ("gameId", "baselineValue", "candidateValue", "delta"):
        assert f'data-game-sort="{sort_key}"' in html
    assert "text-decoration-line:underline" in html
    assert "gameSortDirection" in html


def test_frontier_comparison_can_select_oracle_or_specific_model() -> None:
    html = (Path(__file__).parents[1] / "viewer" / "comparison.html").read_text(encoding="utf-8")

    assert 'id="frontier-model-select"' in html
    assert 'option value="best"' in html
    assert 'game.models[frontierReference]' in html
    assert 'frontierReference = frontierModelSelect.value' in html
    assert "rows.slice(0, 10)" not in html
    assert "All frontier gaps" not in html
