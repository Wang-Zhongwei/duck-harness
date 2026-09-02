"""Optional multimodal context helpers for ARC analyzer prompts."""
from __future__ import annotations

import base64
import functools
import io
import math
import os
from typing import Any

from PIL import Image

from inference.agent.runtime_state import Frame


ARC_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (204, 204, 204),
    2: (153, 153, 153),
    3: (102, 102, 102),
    4: (51, 51, 51),
    5: (0, 0, 0),
    6: (229, 58, 163),
    7: (255, 123, 204),
    8: (249, 60, 49),
    9: (30, 147, 255),
    10: (136, 216, 241),
    11: (255, 220, 0),
    12: (255, 133, 27),
    13: (146, 18, 49),
    14: (79, 204, 48),
    15: (163, 86, 214),
}


def multimodal_context() -> str:
    return os.environ.get("MULTIMODAL_CONTEXT", "").strip().lower()


def current_grid_image_enabled() -> bool:
    return multimodal_context() == "current_grid"


def current_grid_image_upscale() -> int:
    raw = os.environ.get("MULTIMODAL_UPSCALE", "").strip()
    if not raw:
        return 16
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def frame_to_png_data_url(frame: Frame, *, upscale: int | None = None) -> str:
    rows = len(frame.grid)
    cols = max((len(row) for row in frame.grid), default=0)
    if rows <= 0 or cols <= 0:
        raise ValueError("Cannot render an empty grid as an image.")

    scale = current_grid_image_upscale() if upscale is None else max(1, int(upscale))
    image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
    pixels = image.load()
    for row_idx, row in enumerate(frame.grid):
        for col_idx in range(cols):
            value = row[col_idx] if col_idx < len(row) else 0
            pixels[col_idx, row_idx] = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
    if scale > 1:
        image = image.resize((cols * scale, rows * scale), Image.Resampling.NEAREST)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def current_grid_image_part(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None or not current_grid_image_enabled():
        return None
    return {
        "type": "image_url",
        "image_url": {
            "url": frame_to_png_data_url(frame),
        },
    }


# --- Exact prompt-token cost of an image part -------------------------------
#
# Qwen3-VL vision geometry, read from the deployed checkpoint's
# ``preprocessor_config.json`` (RadixArk/unsloth Qwen3.8-27B):
#
#     patch_size 16, merge_size 2  ->  one visual token covers 32x32 px
#     size.shortest_edge / longest_edge are TOTAL-pixel bounds, not side
#     lengths: 65536 px (= 64 tokens) and 16777216 px (= 16384 tokens).
#
# The processor resizes each side to a multiple of ``patch * merge`` and
# clamps the total pixel count into that band, then emits
# ``(h / 16) * (w / 16) / 2**2`` image tokens between a ``<|vision_start|>``
# and ``<|vision_end|>`` marker. Reproducing that arithmetic here lets the
# context trimmer charge an image what the server will actually bill,
# instead of the size of its base64 payload.

_VISION_PATCH_SIZE = 16
_VISION_MERGE_SIZE = 2
_VISION_MIN_PIXELS = 65536
_VISION_MAX_PIXELS = 16777216
_VISION_MARKER_TOKENS = 2  # <|vision_start|> + <|vision_end|>


def _vision_factor() -> int:
    return _VISION_PATCH_SIZE * _VISION_MERGE_SIZE


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Port of the Qwen2/3-VL processor's ``smart_resize``.

    Kept byte-for-byte faithful to the transformers implementation --
    ``round`` to the nearest multiple of ``factor`` first, then ``floor``
    when shrinking to fit ``max_pixels`` and ``ceil`` when growing to reach
    ``min_pixels``. The asymmetry is deliberate upstream: both directions
    stay inside the band they were clamped to.
    """
    factor = _vision_factor()
    h_bar = max(factor, int(round(height / factor)) * factor)
    w_bar = max(factor, int(round(width / factor)) * factor)
    if h_bar * w_bar > _VISION_MAX_PIXELS:
        beta = math.sqrt((height * width) / _VISION_MAX_PIXELS)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < _VISION_MIN_PIXELS:
        beta = math.sqrt(_VISION_MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@functools.lru_cache(maxsize=64)
def _data_url_size(url: str) -> tuple[int, int] | None:
    """``(width, height)`` of a base64 data URL, or None if unreadable.

    ``Image.open`` parses only the header, so this never decodes pixels.
    Cached because the trimmer re-estimates the whole payload once per
    evicted block.
    """
    prefix, _, payload = url.partition(",")
    if not payload or "base64" not in prefix:
        return None
    try:
        with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
            return image.size
    except Exception:  # noqa: BLE001 -- estimation must never raise
        return None


def image_part_prompt_tokens(part: Any) -> int | None:
    """Exact token cost of one ``image_url`` message part.

    Returns None when ``part`` is not an image part, or when its size
    cannot be read (a remote URL, a malformed payload) -- callers then fall
    back to their generic estimate rather than under-counting.
    """
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else None
    if not isinstance(url, str):
        return None
    size = _data_url_size(url)
    if size is None:
        return None
    width, height = size
    if width <= 0 or height <= 0:
        return None
    h_bar, w_bar = smart_resize(height, width)
    factor = _vision_factor()
    return (h_bar // factor) * (w_bar // factor) + _VISION_MARKER_TOKENS
