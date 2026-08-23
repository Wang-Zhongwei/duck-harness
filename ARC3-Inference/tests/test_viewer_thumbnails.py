from pathlib import Path

import pytest

from viewer import thumbnails

VIEWER_DIR = Path(__file__).parents[1] / "viewer"


def test_base_game_id_and_play_url():
    assert thumbnails.base_game_id("ls20-9607627b") == "ls20"
    assert thumbnails.base_game_id("LS20") == "ls20"
    assert thumbnails.base_game_id("../etc") is None
    assert thumbnails.base_game_id("") is None
    assert thumbnails.play_url("ls20-9607627b") == "https://arcprize.org/tasks/ls20"


def test_render_grid_png_scales_to_thumbnail_size():
    Image = pytest.importorskip("PIL.Image")
    import io

    png = thumbnails.render_grid_png([[0, 9], [12, 5]], size=64)
    assert png.startswith(b"\x89PNG")
    image = Image.open(io.BytesIO(png)).convert("RGB")
    assert image.size == (64, 64)
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((63, 0)) == (30, 147, 255)
    assert image.getpixel((0, 63)) == (255, 133, 27)
    assert image.getpixel((63, 63)) == (0, 0, 0)


def test_game_thumbnail_png_rejects_invalid_ids(tmp_path):
    assert thumbnails.game_thumbnail_png("../etc", environments_dir=tmp_path) is None
    assert thumbnails.game_thumbnail_png("zzzz", environments_dir=tmp_path) is None
    assert thumbnails.game_thumbnail_png("ls20", environments_dir=tmp_path / "missing") is None


def test_pages_include_game_peek_hover():
    for name in ("index.html", "comparison.html"):
        html = (VIEWER_DIR / name).read_text(encoding="utf-8")
        assert '<script src="/game-peek.js"></script>' in html, name
        assert "data-game-peek=" in html, name
    js = (VIEWER_DIR / "game_peek.js").read_text(encoding="utf-8")
    assert "/api/thumbnail?game=" in js
    assert "https://arcprize.org/tasks/" in js
