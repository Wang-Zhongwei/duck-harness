"""The context trimmer must charge an image what the server bills for it.

An inlined board is a multi-kilobyte base64 payload but only a few dozen
prompt tokens. Estimating it by rendered length over-counted it ~8x, so the
trimmer evicted real conversation history to make room for pixels that were
never going to cost that much.
"""
from __future__ import annotations

import base64
import io
import json

from PIL import Image

from inference.agent.tool_agent import _estimate_tokens
from inference.agent.vision_context import (
    _VISION_MAX_PIXELS,
    _VISION_MIN_PIXELS,
    image_part_prompt_tokens,
    smart_resize,
)


def _image_part(width: int, height: int) -> dict:
    image = Image.new("RGB", (width, height), (30, 147, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _raw_length_estimate(value) -> int:
    rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return max(1, (len(rendered) + 2) // 3)


def test_smart_resize_rounds_to_the_token_grid() -> None:
    # 32 px per visual token (patch 16 x merge 2); sides land on multiples of it.
    for height, width in [(256, 256), (512, 512), (1024, 1024), (768, 768)]:
        assert smart_resize(height, width) == (height, width)
    h_bar, w_bar = smart_resize(300, 300)
    assert h_bar % 32 == 0 and w_bar % 32 == 0


def test_smart_resize_clamps_into_the_pixel_band() -> None:
    # Below the floor the processor scales back up, above the ceiling it shrinks.
    h_bar, w_bar = smart_resize(64, 64)
    assert h_bar * w_bar >= _VISION_MIN_PIXELS
    h_bar, w_bar = smart_resize(9000, 9000)
    assert h_bar * w_bar <= _VISION_MAX_PIXELS


def test_board_image_costs_its_vision_tokens_not_its_payload_size() -> None:
    # A 64x64 grid at upscale 4 is 256x256 px = 8x8 tokens, plus the two
    # <|vision_start|>/<|vision_end|> markers.
    part = _image_part(256, 256)
    assert image_part_prompt_tokens(part) == 8 * 8 + 2

    estimated = _estimate_tokens(part)
    assert estimated <= 80
    # Guard the regression this fixes: the old estimate was several hundred.
    assert _raw_length_estimate(part) > 4 * estimated


def test_larger_render_costs_proportionally_more() -> None:
    # Quadratic in upscale: 512x512 is 4x the tokens of 256x256.
    small = image_part_prompt_tokens(_image_part(256, 256)) - 2
    large = image_part_prompt_tokens(_image_part(512, 512)) - 2
    assert large == 4 * small


def test_non_image_values_are_unchanged() -> None:
    for value in [
        {"role": "user", "content": "hello"},
        {"messages": [{"role": "system", "content": "x" * 500}]},
        ["a", 1, None],
    ]:
        assert _estimate_tokens(value) == _raw_length_estimate(value)


def test_unreadable_image_falls_back_instead_of_undercounting() -> None:
    # A remote URL has no payload to measure; keep the generic estimate
    # rather than silently charging zero.
    remote = {"type": "image_url", "image_url": {"url": "https://example.com/board.png"}}
    assert image_part_prompt_tokens(remote) is None
    assert _estimate_tokens(remote) == _raw_length_estimate(remote)

    malformed = {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}}
    assert image_part_prompt_tokens(malformed) is None


def test_images_nested_in_a_message_payload_are_found() -> None:
    # The estimate must decompose exactly: text measured by length, image
    # measured by its vision-token count, wherever it sits in the payload.
    image = _image_part(256, 256)

    def payload(part):
        return {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "board:"}, part]},
            ]
        }

    expected = _raw_length_estimate(payload("")) + image_part_prompt_tokens(image)
    assert _estimate_tokens(payload(image)) == expected
    assert _estimate_tokens(payload(image)) < _raw_length_estimate(payload(image))
