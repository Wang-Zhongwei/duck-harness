from pathlib import Path

from viewer.server import _apply_unified_file_diff, _comparison_json_path


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


def test_inference_hyperparameter_comparison_only_renders_differences() -> None:
    html = (Path(__file__).parents[1] / "viewer" / "comparison.html").read_text(encoding="utf-8")

    assert "inferenceConfigDiffSection()" in html
    assert "Only differing inference.json settings are shown." in html
    assert "JSON.stringify(before[key]) !== JSON.stringify(after[key])" in html
    assert "grid-template-columns:minmax(220px,.8fr) repeat(2,minmax(260px,1fr))" in html
    scroll_css = html[html.index(".config-diff-scroll {"):html.index(".config-diff {")]
    assert "max-height:min(280px, 30vh)" in scroll_css
    assert "overflow:auto" in scroll_css
    head_css = html[html.index(".config-head {"):html.index(".config-key {")]
    assert "position:sticky; top:0" in head_css
    assert 'class="config-cell config-value config-baseline"' in html
    assert 'class="config-cell config-value config-candidate"' in html
    assert '<table class="config-diff"' not in html
    assert html.index("inferenceConfigDiffSection() +") < html.index("unlinkedImportsSection() +")
    assert "function normalizedServerMetric" in html
    assert "values.reduce((total, value) => total + value, 0) / values.length" in html
    assert "Equal-weight per-server means" in html
    assert 'servingMetricLabel("Prompt throughput / tok/s",serverMetrics.prompt)' in html
    assert "baselineInference.server.prompt_throughput_tokens_per_s.mean" not in html


def test_run_git_diff_reconstructs_dirty_inference_config() -> None:
    base = '{\n  "chat": {\n    "temperature": 1.0,\n    "top_p": 0.95\n  }\n}\n'
    diff = """diff --git a/ARC3-Inference/configs/inference.json b/ARC3-Inference/configs/inference.json
--- a/ARC3-Inference/configs/inference.json
+++ b/ARC3-Inference/configs/inference.json
@@ -1,6 +1,6 @@
 {
   "chat": {
-    "temperature": 1.0,
+    "temperature": 0.7,
     "top_p": 0.95
   }
 }
"""

    reconstructed = _apply_unified_file_diff(
        base,
        diff,
        "ARC3-Inference/configs/inference.json",
    )

    assert '"temperature": 0.7' in reconstructed
    assert '"temperature": 1.0' not in reconstructed
