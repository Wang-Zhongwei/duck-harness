"""Tests for node boundary and hole (internal boundary) tracing."""
from inference.utils.grid_utils import ARC_COLOR_CHARS as COLOR_CHARS
from inference.utils.segmentation import segment_layer




def test_solid_block_has_no_holes() -> None:
    layer = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    seg = segment_layer(layer, COLOR_CHARS)
    node = seg.find(color=COLOR_CHARS[1]).one()
    assert node["holes"] == []
    assert node["boundary"] == [[1, 1], [1, 2], [2, 2], [2, 1]]


def test_donut_has_one_hole_ring() -> None:
    layer = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    seg = segment_layer(layer, COLOR_CHARS)
    ring = seg.find(color=COLOR_CHARS[1]).one()
    assert ring["boundary"] == [[0, 0], [0, 4], [3, 4], [3, 0]]
    assert len(ring["holes"]) == 1
    assert ring["holes"][0] == [[1, 1], [1, 3], [2, 3], [2, 1]]


def test_two_holes_traced_separately_in_top_left_order() -> None:
    layer = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    seg = segment_layer(layer, COLOR_CHARS)
    node = seg.find(color=COLOR_CHARS[1]).one()
    assert node["holes"] == [[[1, 1]], [[1, 5]]]


def test_notch_is_not_a_hole() -> None:
    # complement region connected to the bbox border is a notch, not a hole
    layer = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    seg = segment_layer(layer, COLOR_CHARS)
    node = seg.find(color=COLOR_CHARS[1]).one()
    assert node["holes"] == []
