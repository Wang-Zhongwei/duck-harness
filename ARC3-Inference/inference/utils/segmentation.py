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


def _object_hash(cells, color):
    """Translation-invariant signature of an object: its color plus its cell shape,
    normalized so the top-left of its bounding box is the origin. Same shape + color
    => same hash regardless of position, so objects can be matched across frames."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    payload = repr((color, norm)).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


NODE_FIELDS = (
    "id", "color", "hash", "area", "bbox", "centroid", "h", "w", "boundary", "children",
)

# Names that read like node fields but are not, mapped to what to use instead. Keeps a
# wrong guess from failing as a confusing TypeError several lines later.
_NODE_FIELD_HINTS = {
    "pixels": "area (an int cell count, not a list of coordinates)",
    "px": "area (an int cell count)",
    "n_pixels": "area",
    "size": "area",
    "cells": "area for the count, or boundary/bbox for the shape",
    "coords": "boundary for the outline, or bbox/centroid for position",
    "shape": "h/w, or bbox",
    "x": "centroid[1] or bbox[1] (columns)",
    "y": "centroid[0] or bbox[0] (rows)",
}


def _summarize_nodes(nodes, limit=6):
    """Compact one-line-per-node listing, used in error messages so a failed lookup
    shows what actually matched instead of only a count."""
    shown = [
        f"id={n['id']} color={n['color']} area={n['area']} bbox={n['bbox']}"
        for n in nodes[:limit]
    ]
    if len(nodes) > limit:
        shown.append(f"... and {len(nodes) - limit} more")
    return "; ".join(shown)


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


class NodeList(list):
    """List of segmentation nodes with accessors for the common match patterns."""

    def one(self):
        """Return the single matching node; raise if the match is not unique."""
        if len(self) != 1:
            detail = _summarize_nodes(self) if self else "no nodes matched the filter"
            raise ValueError(
                f"expected exactly 1 matching node, found {len(self)} -- {detail}. "
                "Tighten the filter, or use .first() / index the list if several "
                "matches are expected."
            )
        return self[0]

    def first(self):
        """Return the top-most-left-most matching node; raise if there are none."""
        if not self:
            raise ValueError("no matching nodes")
        return self[0]


class Segmentation(dict):
    """Segmentation result dict (``nodes`` + ``adjacency_list``) with a query method."""

    FIND_KEYWORDS = (
        "color", "not_color", "area", "min_area", "max_area", "id", "h", "w",
        "min_h", "max_h", "min_w", "max_w", "hash", "in_bbox",
    )

    # Rejected keywords that have an obvious intended target.
    _FIND_KEYWORD_HINTS = {
        "count": "area",
        "min_count": "min_area",
        "max_count": "max_area",
        "px": "area",
        "min_px": "min_area",
        "max_px": "max_area",
        "pixels": "area",
        "size": "area",
        "bbox": "in_bbox",
        "colour": "color",
    }

    def find(self, color=None, not_color=None, area=None, min_area=None, max_area=None,
             id=None, h=None, w=None, min_h=None, max_h=None, min_w=None, max_w=None,
             hash=None, in_bbox=None, **unknown):
        """Filter nodes by keyword; returns a :class:`NodeList` in id (top-left) order.

        Every keyword matches the node field of the same name, so anything printed
        from a node can be filtered on.

        - ``color`` / ``not_color``: a color char or a set of them.
        - ``area``: exact cell count; ``min_area`` / ``max_area``: inclusive bounds.
        - ``id``: exact node id (only stable within a single frame).
        - ``h`` / ``w``: exact bbox height/width; ``min_h`` / ``max_h`` / ``min_w`` /
          ``max_w``: inclusive bounds.
        - ``hash``: exact object hash (position-invariant; use to re-find an
          object after an action instead of holding a stale node reference).
        - ``in_bbox``: ``(r0, c0, r1, c1)`` -- keep nodes whose bbox lies fully inside.
        """
        if unknown:
            bad = sorted(unknown)
            hints = [
                f"{k!r} -> {self._FIND_KEYWORD_HINTS[k]!r}"
                for k in bad
                if k in self._FIND_KEYWORD_HINTS
            ]
            message = (
                f"find() got unexpected keyword(s): {', '.join(repr(k) for k in bad)}. "
                f"Valid keywords are exactly: {', '.join(self.FIND_KEYWORDS)}."
            )
            if hints:
                message += f" Did you mean {', '.join(hints)}?"
            raise TypeError(message)
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
            if id is not None and node["id"] != id:
                continue
            if area is not None and node["area"] != area:
                continue
            if min_area is not None and node["area"] < min_area:
                continue
            if max_area is not None and node["area"] > max_area:
                continue
            if h is not None and node["h"] != h:
                continue
            if min_h is not None and node["h"] < min_h:
                continue
            if max_h is not None and node["h"] > max_h:
                continue
            if w is not None and node["w"] != w:
                continue
            if min_w is not None and node["w"] < min_w:
                continue
            if max_w is not None and node["w"] > max_w:
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
      - ``area``: number of cells in the component (an int, not a coordinate list).
        Counts only the component's own cells -- enclosed children are separate
        components and are not included, though ``bbox``/``h``/``w`` do span them.
      - ``bbox``: ``[r0, c0, r1, c1]`` -- the component's inclusive bounding box.
      - ``centroid``: ``[r, c]`` -- the component's center of mass, rounded to ints.
      - ``h`` / ``w``: bounding-box height and width.
      - ``boundary``: the component's outer perimeter as an ordered, clockwise list of
        ``[row, col]`` corner points -- a Moore-neighbour trace reduced to only the
        vertices where the contour changes direction (enclosed holes are not traced).
      - ``children``: ids of components directly enclosed by this node. A is a child of
        B only if B is the innermost component that fully surrounds A (every path from A
        to the grid edge crosses B), which yields a clean nesting tree.

    Returns a :class:`Segmentation` dict with:
      - ``nodes``: list of the node dicts above, in id order.
      - ``adjacency_list``: sorted list of ``[i, j]`` id pairs for components that share
        a 4-connected edge (includes parent/child pairs, since they physically touch).
    Query it with ``.find(color=..., area=..., ...)``.
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
        nodes.append(
            Node(
                {
                    "id": cid,
                    "color": color,
                    "hash": _object_hash(cells, color),
                    "area": len(cells),
                    "bbox": [r0, c0, r1, c1],
                    "centroid": [
                        round(sum(rows_) / len(cells)),
                        round(sum(cols_) / len(cells)),
                    ],
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "boundary": [[r, c] for r, c in boundary],
                    "children": children[cid],
                }
            )
        )

    return Segmentation({"nodes": nodes, "adjacency_list": adjacency_list})
