"""Connected-component segmentation of a single frame layer.

This module is intentionally self-contained -- standard library only, no project
imports, no ``from __future__`` import -- so its source can be spliced verbatim into
the Python-tool sandbox bootstrap, where project packages are not importable.
"""

import hashlib

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
# clockwise Moore-neighbour offsets, starting at NW
_CW = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
_CW_INDEX = {off: i for i, off in enumerate(_CW)}


def _trace_outer_contour(cells, start):
    """Moore-neighbour trace of a 4-connected component's outer perimeter, clockwise."""
    if len(cells) == 1:
        return [start]

    contour = [start]
    b = start
    prev = (start[0], start[1] - 1)  # W neighbour: outside the component since start is reading-order-min
    second = None
    for _ in range(8 * len(cells) + 16):
        idx = _CW_INDEX[(prev[0] - b[0], prev[1] - b[1])]
        nxt = None
        for k in range(1, 9):
            off = _CW[(idx + k) % 8]
            cand = (b[0] + off[0], b[1] + off[1])
            if cand in cells:
                nxt = cand
                back = _CW[(idx + k - 1) % 8]
                new_prev = (b[0] + back[0], b[1] + back[1])
                break
        if nxt is None:
            break
        if second is None:
            second = nxt
        elif b == start and nxt == second:  # Jacob's stopping criterion
            break
        contour.append(nxt)
        prev, b = new_prev, nxt

    if len(contour) > 1 and contour[-1] == contour[0]:
        contour.pop()
    return contour


def _corner_points(contour):
    """Reduce a traced contour loop to only the points where its direction changes."""
    if len(contour) <= 2:
        return list(contour)
    m = len(contour)
    corners = []
    for i in range(m):
        prev, cur, nxt = contour[i - 1], contour[i], contour[(i + 1) % m]
        d_in = (cur[0] - prev[0], cur[1] - prev[1])
        d_out = (nxt[0] - cur[0], nxt[1] - cur[1])
        if d_in != d_out:
            corners.append(cur)
    return corners


def _object_id(cells, color):
    """Translation-invariant identity of an object: its color plus its cell shape,
    normalized so the top-left of its bounding box is the origin. Same shape + color
    => same id regardless of position, so objects can be matched across frames.
    Identical-looking objects share an id."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    payload = repr((color, norm)).encode()
    return hashlib.sha1(payload).hexdigest()[:8]


NODE_FIELDS = (
    "id", "color", "area", "bbox", "boundary", "children",
)

# Names that read like node fields but are not, mapped to what to use instead. Keeps a
# wrong guess from failing as a confusing TypeError several lines later.
_NODE_FIELD_HINTS = {
    "hash": "id",
    "pixels": "area (an int cell count, not a list of coordinates)",
    "px": "area (an int cell count)",
    "n_pixels": "area",
    "size": "area",
    "cells": "area for the count, or boundary/bbox for the shape",
    "coords": "boundary for the outline, or bbox for position",
    "shape": "bbox, as [min_row, min_col, max_row, max_col]",
    "centroid": "bbox -- midpoint is ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)",
    "h": "bbox -- height is bbox[2] - bbox[0] + 1",
    "w": "bbox -- width is bbox[3] - bbox[1] + 1",
    "x": "bbox[1] / bbox[3] (columns)",
    "y": "bbox[0] / bbox[2] (rows)",
}


class Node(dict):
    """One segmentation node. Behaves as a plain dict; a missing key raises with a
    pointer to the right field name rather than a bare ``KeyError``."""

    def __missing__(self, key):
        hint = _NODE_FIELD_HINTS.get(key)
        if hint is not None:
            raise KeyError(
                f"node has no field {key!r}; use {hint}. "
                f"Node fields: {', '.join(NODE_FIELDS)}."
            )
        raise KeyError(
            f"node has no field {key!r}. Node fields: {', '.join(NODE_FIELDS)}."
        )


def segment_layer(layer, color_chars):
    """Segment one frame layer into connected-component nodes.

    Pass a single layer (if the frame has multiple) and ``color_chars``, the ARC
    color-symbol mapping (indexed by integer color value -> single-char label). The
    layer is partitioned into 4-connected components of equal integer value via flood
    fill, and each component becomes a node. Nodes are listed in reading order of
    their top-most-left-most cell.

    Each node is a dict with:
      - ``id``: the object's identity -- a short string derived from its color plus its
        cell shape normalized to a top-left origin, so the same-looking object gets the
        same id regardless of position or frame (lets objects be matched across frames;
        identical-looking objects share an id).
      - ``color``: the component's ARC color character (looked up in ``color_chars``).
      - ``area``: number of cells in the component (an int, not a coordinate list).
        Counts only the component's own cells -- enclosed children are separate
        components and are not included, though ``bbox`` does span them.
      - ``bbox``: ``[r0, c0, r1, c1]`` -- the component's inclusive bounding box.
      - ``boundary``: the component's outer perimeter as an ordered, clockwise list of
        ``[row, col]`` corner points -- a Moore-neighbour trace reduced to only the
        vertices where the contour changes direction (enclosed holes are not traced).
      - ``children``: ids of components directly enclosed by this node. A is a child of
        B only if B is the innermost component that fully surrounds A (every path from A
        to the grid edge crosses B), which yields a clean nesting tree. When several
        enclosed objects look identical their ids coincide; disambiguate spatially by
        filtering on ``bbox``.

    Returns a dict with:
      - ``nodes``: list of the node dicts above, in reading order.
      - ``adjacency_list``: sorted, de-duplicated list of ``[id_a, id_b]`` pairs for
        components that share a 4-connected edge (includes parent/child pairs, since
        they physically touch).
    """
    height = len(layer)
    width = len(layer[0]) if height else 0

    # connected components, 4-connectivity. Reading-order scan => component ids are
    # already ordered by top-most-left-most cell (each layer is 64x64, so it is unique).
    comp_id = [[-1] * width for _ in range(height)]
    components = []  # each: {"value": int, "cells": set[(r, c)], "start": (r, c)}
    for sr in range(height):
        for sc in range(width):
            if comp_id[sr][sc] != -1:
                continue
            value = layer[sr][sc]
            cid = len(components)
            cells = set()
            stack = [(sr, sc)]
            comp_id[sr][sc] = cid
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for dr, dc in _ORTH:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and comp_id[nr][nc] == -1 and layer[nr][nc] == value:
                        comp_id[nr][nc] = cid
                        stack.append((nr, nc))
            components.append({"value": int(value), "cells": cells, "start": (sr, sc)})

    n = len(components)

    # adjacency between components: any two components with 4-adjacent cells
    adj_pairs = set()
    for r in range(height):
        for c in range(width):
            cid = comp_id[r][c]
            if r + 1 < height and comp_id[r + 1][c] != cid:
                other = comp_id[r + 1][c]
                adj_pairs.add((min(cid, other), max(cid, other)))
            if c + 1 < width and comp_id[r][c + 1] != cid:
                other = comp_id[r][c + 1]
                adj_pairs.add((min(cid, other), max(cid, other)))

    # containment: for each component b, flood-fill its complement inward from the grid
    # border; any component whose cells are never reached is enclosed by b.
    enclosers = [set() for _ in range(n)]
    for b in range(n):
        reached = [[False] * width for _ in range(height)]
        stack = []
        for r in range(height):
            for c in (0, width - 1):
                if comp_id[r][c] != b and not reached[r][c]:
                    reached[r][c] = True
                    stack.append((r, c))
        for c in range(width):
            for r in (0, height - 1):
                if comp_id[r][c] != b and not reached[r][c]:
                    reached[r][c] = True
                    stack.append((r, c))
        while stack:
            r, c = stack.pop()
            for dr, dc in _ORTH:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not reached[nr][nc] and comp_id[nr][nc] != b:
                    reached[nr][nc] = True
                    stack.append((nr, nc))
        for a in range(n):
            if a == b:
                continue
            ar, ac = components[a]["start"]
            if not reached[ar][ac]:
                enclosers[a].add(b)

    # parent = innermost encloser. enclosers are transitive, so along a nesting chain the
    # innermost component is the one that is itself most deeply enclosed.
    children = [[] for _ in range(n)]
    for a in range(n):
        if enclosers[a]:
            parent = max(enclosers[a], key=lambda e: (len(enclosers[e]), -e))
            children[parent].append(a)
    for child_list in children:
        child_list.sort()

    object_ids = [
        _object_id(components[cid]["cells"], color_chars[max(0, min(15, components[cid]["value"]))])
        for cid in range(n)
    ]

    nodes = []
    for cid in range(n):
        comp = components[cid]
        color = color_chars[max(0, min(15, comp["value"]))]
        boundary = _corner_points(_trace_outer_contour(comp["cells"], comp["start"]))
        cells = comp["cells"]
        rows_ = [r for r, _ in cells]
        cols_ = [c for _, c in cells]
        r0, c0, r1, c1 = min(rows_), min(cols_), max(rows_), max(cols_)
        nodes.append(
            Node(
                {
                    "id": object_ids[cid],
                    "color": color,
                    "area": len(cells),
                    "bbox": [r0, c0, r1, c1],
                    "boundary": [[r, c] for r, c in boundary],
                    "children": [object_ids[child] for child in children[cid]],
                }
            )
        )

    adjacency_list = sorted(
        {tuple(sorted((object_ids[a], object_ids[b]))) for a, b in adj_pairs}
    )
    adjacency_list = [list(pair) for pair in adjacency_list]

    return {"nodes": nodes, "adjacency_list": adjacency_list}
