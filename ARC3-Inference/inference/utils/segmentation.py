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


def _hole_regions(cells, bbox):
    """4-connected components of the complement, inside ``bbox``, not reachable from
    the bbox border -- i.e. the component's fully enclosed holes."""
    r0, c0, r1, c1 = bbox
    complement = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in cells
    }
    outside = {p for p in complement if p[0] in (r0, r1) or p[1] in (c0, c1)}
    stack = list(outside)
    while stack:
        r, c = stack.pop()
        for dr, dc in _ORTH:
            nxt = (r + dr, c + dc)
            if nxt in complement and nxt not in outside:
                outside.add(nxt)
                stack.append(nxt)
    enclosed = complement - outside
    holes = []
    seen = set()
    for p in sorted(enclosed):
        if p in seen:
            continue
        region = set()
        stack = [p]
        seen.add(p)
        while stack:
            r, c = stack.pop()
            region.add((r, c))
            for dr, dc in _ORTH:
                nxt = (r + dr, c + dc)
                if nxt in enclosed and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        holes.append(region)
    return holes


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


def _object_hash(cells, color):
    """Translation-invariant signature of an object: its color plus its cell shape,
    normalized so the top-left of its bounding box is the origin. Same shape + color
    => same hash regardless of position, so objects can be matched across frames."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    payload = repr((color, norm)).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


class NodeList(list):
    """List of segmentation nodes with accessors for the common match patterns."""

    def one(self):
        """Return the single matching node; raise if the match is not unique."""
        if len(self) != 1:
            raise ValueError(
                f"expected exactly 1 matching node, found {len(self)}"
            )
        return self[0]

    def first(self):
        """Return the top-most-left-most matching node; raise if there are none."""
        if not self:
            raise ValueError("no matching nodes")
        return self[0]


class Segmentation(dict):
    """Segmentation result dict (``nodes`` + ``adjacency_list``) with a query method."""

    def find(self, color=None, not_color=None, px=None, min_px=None, max_px=None,
             hash=None, in_bbox=None):
        """Filter nodes by keyword; returns a :class:`NodeList` in id (top-left) order.

        - ``color`` / ``not_color``: a color char or a set of them.
        - ``px``: exact pixel count; ``min_px`` / ``max_px``: inclusive bounds.
        - ``hash``: exact object hash (position-invariant; use to re-find an
          object after an action instead of holding a stale node reference).
        - ``in_bbox``: ``(r0, c0, r1, c1)`` -- keep nodes whose bbox lies fully inside.
        """
        if isinstance(color, str):
            color = {color}
        if isinstance(not_color, str):
            not_color = {not_color}
        out = NodeList()
        for node in self["nodes"]:
            if color is not None and node["color"] not in color:
                continue
            if not_color is not None and node["color"] in not_color:
                continue
            if px is not None and node["pixels"] != px:
                continue
            if min_px is not None and node["pixels"] < min_px:
                continue
            if max_px is not None and node["pixels"] > max_px:
                continue
            if hash is not None and node["hash"] != hash:
                continue
            if in_bbox is not None:
                r0, c0, r1, c1 = in_bbox
                nr0, nc0, nr1, nc1 = node["bbox"]
                if not (r0 <= nr0 and c0 <= nc0 and nr1 <= r1 and nc1 <= c1):
                    continue
            out.append(node)
        return out


def segment_layer(layer, color_chars):
    """Segment one frame layer into connected-component nodes.

    Pass a single layer (if the frame has multiple) and ``color_chars``, the ARC
    color-symbol mapping (indexed by integer color value -> single-char label). The
    layer is partitioned into 4-connected components of equal integer value via flood
    fill, and each component becomes a node. Nodes are ordered by their
    top-most-left-most cell -- unique within a 64x64 layer -- and that order is the
    node ``id``.

    Each node is a dict with:
      - ``id``: index in the top-left ordering.
      - ``color``: the component's ARC color character (looked up in ``color_chars``).
      - ``hash``: a translation-invariant signature of the object -- its color plus its
        cell shape normalized to a top-left origin -- so the same shape gets the same
        hash regardless of position (lets objects be matched across frames, or when
        several similar objects appear in one frame).
      - ``pixels``: number of cells in the component.
      - ``bbox``: ``[r0, c0, r1, c1]`` -- the component's inclusive bounding box.
      - ``centroid``: ``[r, c]`` -- the component's center of mass, rounded to ints.
      - ``h`` / ``w``: bounding-box height and width.
      - ``boundary``: the component's outer perimeter as an ordered, clockwise list of
        ``[row, col]`` corner points -- a Moore-neighbour trace reduced to only the
        vertices where the contour changes direction.
      - ``holes``: internal boundaries -- one corner ring (same format as ``boundary``)
        per fully enclosed hole, in top-left order; ``[]`` for solid components.
        ``boundary`` + ``holes`` together describe the exact shape of the region, so
        they can stand in for a full cell listing.
      - ``children``: ids of components directly enclosed by this node. A is a child of
        B only if B is the innermost component that fully surrounds A (every path from A
        to the grid edge crosses B), which yields a clean nesting tree.

    Returns a :class:`Segmentation` dict with:
      - ``nodes``: list of the node dicts above, in id order.
      - ``adjacency_list``: sorted list of ``[i, j]`` id pairs for components that share
        a 4-connected edge (includes parent/child pairs, since they physically touch).
    Query it with ``.find(color=..., px=..., ...)``.
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
    adjacency_list = sorted([a, b] for a, b in adj_pairs)

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

    nodes = []
    for cid in range(n):
        comp = components[cid]
        color = color_chars[max(0, min(15, comp["value"]))]
        boundary = _corner_points(_trace_outer_contour(comp["cells"], comp["start"]))
        cells = comp["cells"]
        rows_ = [r for r, _ in cells]
        cols_ = [c for _, c in cells]
        r0, c0, r1, c1 = min(rows_), min(cols_), max(rows_), max(cols_)
        holes = [
            [[r, c] for r, c in _corner_points(_trace_outer_contour(region, min(sorted(region))))]
            for region in _hole_regions(cells, (r0, c0, r1, c1))
        ]
        nodes.append(
            {
                "id": cid,
                "color": color,
                "hash": _object_hash(cells, color),
                "pixels": len(cells),
                "bbox": [r0, c0, r1, c1],
                "centroid": [round(sum(rows_) / len(cells)), round(sum(cols_) / len(cells))],
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
                "boundary": [[r, c] for r, c in boundary],
                "holes": holes,
                "children": children[cid],
            }
        )

    return Segmentation({"nodes": nodes, "adjacency_list": adjacency_list})
