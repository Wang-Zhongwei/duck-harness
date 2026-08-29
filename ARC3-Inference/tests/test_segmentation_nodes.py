import pytest

from inference.utils.segmentation import segment_layer

# 0 = white background, 5 = gray, 2 = red.
COLOR_CHARS = "WwgGcBMPRbSYOrNp"


def _seg(grid):
    return segment_layer(grid, COLOR_CHARS)


def _one(seg, color):
    """The single node of ``color``; nodes are filtered by plain comprehension now
    that the segmentation result is an ordinary dict."""
    nodes = [n for n in seg["nodes"] if n["color"] == color]
    assert len(nodes) == 1, f"expected exactly one {color!r} node, got {len(nodes)}"
    return nodes[0]


def test_segmentation_is_a_plain_dict():
    seg = _seg([[5]])
    assert type(seg) is dict
    assert set(seg) == {"nodes", "adjacency_list"}


def test_node_exposes_exactly_the_documented_fields():
    node = _one(_seg([[5, 5], [5, 0]]), "B")
    assert set(node) == {"id", "color", "area", "bbox", "boundary", "children"}


def test_node_area_is_an_int_cell_count():
    gray = _one(_seg([[5, 5], [5, 0]]), "B")

    assert gray["area"] == 3
    with pytest.raises(TypeError):
        len(gray["area"])


def test_node_rejects_the_old_pixels_name_with_a_pointer_to_area():
    gray = _one(_seg([[5, 5], [5, 0]]), "B")

    with pytest.raises(KeyError) as excinfo:
        gray["pixels"]
    assert "area" in str(excinfo.value)
    assert "not a list of coordinates" in str(excinfo.value)


def test_node_rejects_the_old_hash_name_with_a_pointer_to_id():
    gray = _one(_seg([[5]]), "B")

    with pytest.raises(KeyError) as excinfo:
        gray["hash"]
    assert "use id" in str(excinfo.value)


@pytest.mark.parametrize("removed", ["centroid", "h", "w"])
def test_removed_geometry_fields_point_at_bbox(removed):
    gray = _one(_seg([[5, 5], [5, 0]]), "B")

    with pytest.raises(KeyError) as excinfo:
        gray[removed]
    assert "bbox" in str(excinfo.value)


def test_id_is_shared_by_identical_objects_and_stable_across_frames():
    # Two identical vertical gray dominoes; one differently shaped gray blob.
    frame_a = _seg([
        [5, 0, 5],
        [5, 0, 5],
        [0, 0, 0],
        [5, 5, 0],
        [5, 0, 0],
    ])
    dominoes = [n for n in frame_a["nodes"] if n["color"] == "B" and n["area"] == 2]
    assert len(dominoes) == 2
    assert dominoes[0]["id"] == dominoes[1]["id"]  # same look => same id

    other = [n for n in frame_a["nodes"] if n["color"] == "B" and n["area"] == 3]
    assert len(other) == 1
    assert other[0]["id"] != dominoes[0]["id"]  # different shape => different id

    # The domino moved: same object id in a new frame, at a new position.
    frame_b = _seg([
        [0, 0, 0],
        [0, 5, 0],
        [0, 5, 0],
    ])
    moved = _one(frame_b, "B")
    assert moved["id"] == dominoes[0]["id"]
    assert isinstance(moved["id"], str)

    # Look-alikes share an id, so disambiguate spatially on bbox.
    left = [
        n for n in frame_a["nodes"]
        if n["id"] == dominoes[0]["id"] and n["bbox"] == [0, 0, 1, 0]
    ]
    assert len(left) == 1


def test_children_and_adjacency_reference_object_ids():
    # A gray ring enclosing a single red cell.
    seg = _seg([
        [5, 5, 5],
        [5, 2, 5],
        [5, 5, 5],
    ])
    ring = _one(seg, "B")
    red = _one(seg, "g")

    assert ring["children"] == [red["id"]]
    assert sorted([ring["id"], red["id"]]) in seg["adjacency_list"]
