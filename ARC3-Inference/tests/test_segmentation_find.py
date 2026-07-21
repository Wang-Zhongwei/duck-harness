import pytest

from inference.utils.segmentation import segment_layer

# 0 = white background, 5 = gray, 2 = red.
COLOR_CHARS = "WwgGcBMPRbSYOrNp"


def _seg(grid):
    return segment_layer(grid, COLOR_CHARS)


def test_node_area_is_an_int_cell_count():
    seg = _seg([[5, 5], [5, 0]])
    gray = seg.find(color="B").one()

    assert gray["area"] == 3
    with pytest.raises(TypeError):
        len(gray["area"])


def test_node_rejects_the_old_pixels_name_with_a_pointer_to_area():
    seg = _seg([[5, 5], [5, 0]])
    gray = seg.find(color="B").one()

    with pytest.raises(KeyError) as excinfo:
        gray["pixels"]
    assert "area" in str(excinfo.value)
    assert "not a list of coordinates" in str(excinfo.value)


def test_find_supports_every_documented_keyword():
    # Two separate gray blobs of different sizes, plus a red one.
    grid = [
        [5, 0, 2],
        [5, 0, 0],
        [0, 0, 5],
    ]
    seg = _seg(grid)

    tall = seg.find(color="B", h=2).one()
    assert tall["area"] == 2
    assert seg.find(color="B", area=1).one()["bbox"] == [2, 2, 2, 2]
    assert seg.find(color="B", min_area=2).one() is tall
    assert seg.find(id=tall["id"]).one() is tall
    assert seg.find(color="B", w=1, min_h=2, max_h=2).one() is tall
    assert [n["id"] for n in seg.find(in_bbox=(0, 0, 1, 1))] == [tall["id"]]
    assert seg.find(not_color={"W", "B"}).one()["color"] == "g"


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
    other = frame_a.find(color="B", area=3).one()
    assert other["id"] != dominoes[0]["id"]  # different shape => different id

    # The domino moved: same object id in a new frame, at a new position.
    frame_b = _seg([
        [0, 0, 0],
        [0, 5, 0],
        [0, 5, 0],
    ])
    moved = frame_b.find(color="B").one()
    assert moved["id"] == dominoes[0]["id"]
    assert isinstance(moved["id"], str)

    # find(id=...) returns every look-alike; disambiguate spatially.
    both = frame_a.find(id=dominoes[0]["id"])
    assert len(both) == 2
    left = frame_a.find(id=dominoes[0]["id"], in_bbox=(0, 0, 1, 1)).one()
    assert left["bbox"] == [0, 0, 1, 0]


def test_children_and_adjacency_reference_object_ids():
    # A gray ring enclosing a single red cell.
    seg = _seg([
        [5, 5, 5],
        [5, 2, 5],
        [5, 5, 5],
    ])
    ring = seg.find(color="B").one()
    red = seg.find(color="g").one()
    assert ring["children"] == [red["id"]]
    assert sorted([ring["id"], red["id"]]) in seg["adjacency_list"]


def test_find_rejects_unknown_keywords_with_the_full_valid_list():
    seg = _seg([[5]])

    with pytest.raises(TypeError) as excinfo:
        seg.find(px=1)
    message = str(excinfo.value)
    assert "'px'" in message
    assert "min_area" in message  # the exhaustive keyword list is included
    assert "Did you mean 'px' -> 'area'?" in message

    with pytest.raises(TypeError) as excinfo:
        seg.find(hash="ab12cd34")
    assert "Did you mean 'hash' -> 'id'?" in str(excinfo.value)

    node = seg.find(color="B").one()
    with pytest.raises(KeyError) as excinfo:
        node["hash"]
    assert "use id" in str(excinfo.value)


def test_one_error_names_the_nodes_that_matched():
    seg = _seg([[5, 0, 5]])

    with pytest.raises(ValueError) as excinfo:
        seg.find(color="B").one()
    message = str(excinfo.value)
    assert "found 2" in message
    assert "bbox=[0, 0, 0, 0]" in message
    assert "bbox=[0, 2, 0, 2]" in message


def test_one_error_on_no_match_says_so():
    seg = _seg([[0]])

    with pytest.raises(ValueError) as excinfo:
        seg.find(color="R").one()
    assert "found 0" in str(excinfo.value)
    assert "no nodes matched the filter" in str(excinfo.value)
