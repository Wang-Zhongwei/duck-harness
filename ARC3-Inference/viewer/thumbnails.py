"""Render per-game thumbnails (initial frame) from the offline ARC-AGI-3 env files."""
from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from viewer.data import _ARC_COLOR_MAP, _hex_to_rgb

log = logging.getLogger(__name__)

ARC_PLAY_BASE_URL = "https://arcprize.org/tasks"
THUMBNAIL_SIZE = 256
_BASE_ID_RE = re.compile(r"^[a-z0-9]{4}$")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / ".cache" / "viewer-thumbnails"
_MEMORY_CACHE: dict[str, bytes] = {}
_ARCADE_LOCK = threading.Lock()
_ARCADE: Any = None
_ARCADE_ENV_DIR: str | None = None


def base_game_id(game_id: str) -> str | None:
    """``ls20-9607627b`` -> ``ls20``; None when the id is not a valid ARC game id."""
    base = str(game_id or "").strip().lower().split("-")[0]
    return base if _BASE_ID_RE.match(base) else None


def play_url(game_id: str) -> str:
    return f"{ARC_PLAY_BASE_URL}/{base_game_id(game_id) or game_id}"


def default_environments_dir() -> Path | None:
    """Resolve the offline env dir from ``$ARC3_ENVIRONMENTS_DIR`` or ``configs/inference.json``."""
    override = os.environ.get("ARC3_ENVIRONMENTS_DIR", "").strip()
    if override:
        return Path(override)
    config_path = _PROJECT_ROOT / "configs" / "inference.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = str((config.get("environment") or {}).get("environments_dir") or "").strip()
    if not value or value == "__auto__":
        return None
    return Path(value)


def _env_metadata_mtime(environments_dir: Path, base_id: str) -> float | None:
    game_dir = environments_dir / base_id
    if not game_dir.is_dir():
        return None
    mtimes = [p.stat().st_mtime for p in game_dir.glob("*/metadata.json")]
    return max(mtimes) if mtimes else None


def _arcade(environments_dir: Path) -> Any:
    global _ARCADE, _ARCADE_ENV_DIR
    import arc_agi  # noqa: PLC0415 - optional dependency; only needed for thumbnails

    env_dir = str(environments_dir)
    if _ARCADE is None or _ARCADE_ENV_DIR != env_dir:
        arcade_logger = logging.getLogger("viewer.thumbnails.arcade")
        arcade_logger.setLevel(logging.WARNING)
        logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
        _ARCADE = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE,
            environments_dir=env_dir,
            recordings_dir=str(_CACHE_DIR / "recordings"),
            logger=arcade_logger,
        )
        _ARCADE_ENV_DIR = env_dir
    return _ARCADE


def _initial_grid(environments_dir: Path, base_id: str) -> list[list[int]] | None:
    with _ARCADE_LOCK:
        env = _arcade(environments_dir).make(base_id, save_recording=False)
        if env is None:
            return None
        try:
            observation = env.observation_space
            frame = getattr(observation, "frame", None) if observation is not None else None
            if frame is None:
                return None
            layers = list(frame)
            if not layers:
                return None
            grid = layers[-1]
            return [[int(value) for value in row] for row in grid]
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best effort cleanup
                    log.debug("env.close() failed for %s", base_id, exc_info=True)


def render_grid_png(grid: list[list[int]], *, size: int = THUMBNAIL_SIZE) -> bytes:
    """Encode an ARC integer grid as a nearest-neighbour upscaled PNG."""
    from PIL import Image  # noqa: PLC0415

    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    if not height or not width:
        raise ValueError("empty grid")
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y, row in enumerate(grid):
        for x in range(width):
            value = row[x] if x < len(row) else 0
            pixels[x, y] = _hex_to_rgb(_ARC_COLOR_MAP.get(int(value), "#000000FF"))
    scale = max(1, size // max(width, height))
    image = image.resize((width * scale, height * scale), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def game_thumbnail_png(game_id: str, *, environments_dir: Path | None = None) -> bytes | None:
    """Return the PNG thumbnail for ``game_id`` (cached in memory and on disk), or None."""
    base_id = base_game_id(game_id)
    if base_id is None:
        return None
    environments_dir = environments_dir or default_environments_dir()
    if environments_dir is None or not environments_dir.is_dir():
        return None
    cached = _MEMORY_CACHE.get(base_id)
    if cached is not None:
        return cached
    source_mtime = _env_metadata_mtime(environments_dir, base_id)
    if source_mtime is None:
        return None
    cache_path = _CACHE_DIR / f"{base_id}.png"
    try:
        if cache_path.is_file() and cache_path.stat().st_mtime >= source_mtime:
            data = cache_path.read_bytes()
            _MEMORY_CACHE[base_id] = data
            return data
    except OSError:
        pass
    try:
        grid = _initial_grid(environments_dir, base_id)
    except Exception:
        log.warning("Could not render thumbnail for %s", base_id, exc_info=True)
        return None
    if grid is None:
        return None
    data = render_grid_png(grid)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".png.tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, cache_path)
    except OSError:
        log.debug("Could not persist thumbnail cache for %s", base_id, exc_info=True)
    _MEMORY_CACHE[base_id] = data
    return data
