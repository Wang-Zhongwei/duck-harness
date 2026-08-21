# Python tool-call analysis — run `20260717_142339`

Corpus: 250 episode transcripts (passes 0 and 5), **21,406 `python` tool calls**, mean 85.6 calls/episode.
All numbers produced by regex census over extracted `code` payloads; scripts in `$CLAUDE_JOB_DIR/tmp/{extract,analyze,analyze2,analyze3,savings}.py`.

## Table 1 — Call size

| metric | mean | median | p90 | p99 | max |
|---|---|---|---|---|---|
| lines | 14.4 | 12 | 29 | 55 | 140 |
| chars | 564 | 480 | 1117 | 2021 | 4924 |

## Table 2 — Token budget of the python channel (whole run)

| stream | chars | ~tokens |
|---|---|---|
| agent-written code (out) | 12,080,074 | 3,020,018 |
| tool results (in) | 20,946,278 | 5,236,569 |
| **total** | **33,026,352** | **8,256,588** |

Tool results: mean 979 chars, median 804, p99 4,123. Only 0.9% hit the 1024-token cap — the agent is *not* fighting the output limit; it is spending its budget on re-deriving the same views.

## Table 3 — Idiom frequency (share of 21,406 calls)

| idiom | calls | % |
|---|---|---|
| `print(...)` | 15,663 | 73.2% |
| fetch `current_frame.segmentation` | 12,746 | 59.5% |
| read node `color` | 12,623 | 59.0% |
| read node `boundary` | 12,244 | 57.2% |
| iterate `seg['nodes']` | 12,092 | 56.5% |
| executes `action(...)` | 12,453 | 58.2% |
| read node `pixels` | 10,530 | 49.2% |
| filter nodes by color | 6,925 | 32.4% |
| read node `id` | 6,135 | 28.7% |
| MOUSE action constructed | 5,398 | 25.2% |
| `.ascii` used | 5,175 | 24.2% |
| `.ascii` → splitlines grid | 4,734 | 22.1% |
| `result =` assigned | 4,408 | 20.6% |
| `transitions` / `last_transition` | 1,649 | 7.7% |
| `children` / containment | 1,542 | 7.2% |
| `last_action_result` | 1,442 | 6.7% |
| explicit frame diff | 1,320 | 6.2% |
| node `hash` (object tracking) | 1,195 | 5.6% |
| re-defines a helper `def` | 572 | 2.7% |
| `adjacency_list` | 500 | 2.3% |
| `history[...]` indexing | 323 | 1.5% |
| `previous_frame` | 228 | 1.1% |
| BFS/deque search | 103 | 0.5% |
| `valid_actions` | 101 | 0.5% |

## Table 4 — Redundancy buckets, weighted by code volume

| bucket | calls | % calls | code chars | % of all code |
|---|---|---|---|---|
| loop over nodes → print summary | 10,477 | 48.9% | 7,487,408 | 62.0% |
| fetch segmentation | 11,751 | 54.9% | 8,306,883 | 68.8% |
| rebuild 2D grid from `.ascii` | 5,150 | 24.1% | 3,690,438 | 30.5% |
| **pure `action()` wrapper, zero inspection** | 5,571 | 26.0% | 941,140 | 7.8% |
| hand bbox from `boundary` | 1,882 | 8.8% | 1,604,398 | 13.3% |

## Table 5 — Code duplication (normalized: comments/numbers/strings stripped)

| metric | value |
|---|---|
| unique normalized snippets | 9,163 |
| snippets appearing >1× | 1,467 |
| **calls that are a repeat of another call** | **13,710 (64.0%)** |

Top repeats: `action([{MOUSE,row,col}])` ×2,430 · `action([DIR])` ×1,893 · `action([D,D])` ×196 · `action([D×5])` ×143 · `find_player(seg)` boilerplate ×126.

## Table 6 — The `boundary` mismatch

| measure | value |
|---|---|
| calls reading `boundary` | 12,244 (57.2%) |
| …that immediately call `min()`/`max()` on it | 4,997 (40.8% of those) |
| …that use it as a real polygon/perimeter | **65 (0.5% of those)** |
| hand-written bbox lines total | 8,046 across 245 distinct spellings |

Top spellings: `min_r = min(p[0] for p in b)` ×1,553 · `min_c = min(p[1] for p in b)` ×1,535 · `max_r = max(p[0] for p in b)` ×1,106 · `max_c = max(p[1] for p in b)` ×1,098.

## Table 7 — `action()` payload shapes (13,336 calls)

| shape | count | % |
|---|---|---|
| single directional/SPACE | 6,058 | 45.4% |
| MOUSE dict form | 5,814 | 43.6% |
| multi-action batch | 1,464 | 11.0% |

## Table 8 — Most re-defined helper functions (session is stateless)

| function | times rewritten |
|---|---|
| `find_player` | 172 |
| `get_non_black` | 25 |
| `get_pos` | 25 |
| `extract_symbol` | 21 |
| `get_b_pattern` | 21 |
| `bfs` | 17 |
| `get_blocks` | 14 |
| `get_grid_state` | 13 |

653 `def` statements total across 572 calls.

---

# Proposed abstractions

Ranked by measured code volume removed.

| # | abstraction | calls hit | chars removed | % of agent-written code | status |
|---|---|---|---|---|---|
| 3 | auto node table / `seg.summary()` | 8,322 | 1,213,439 | 10.0% | deferred (set aside for now) |
| 2 | direct `action` tool (bypass python) | 5,571 | 941,140 | 7.8% | **remaining** |
| 1 | precomputed `node.bbox` / `node.centroid` | 4,587 | 580,787 | 4.8% | ✅ done (+ `h`/`w`) |
| 4 | `seg.find(color=…, pixels=…)` query | 8,075 | 555,268 | 4.6% | ✅ done (`.one()`/`.first()`) |
| 7 | `frame.at()` / `frame.crop()` | 5,323 | 277,494 | 2.3% | **remaining** |
| 5 | builtin `diff(a, b)` | 1,296 | 74,064 | 0.6% | ✅ done (`last_transition.diff` + `frame_diff` builtin, commit 3074c93) |
| 6 | persistent scratch namespace | 572 | 19,102 | 0.2% | **remaining** |
| | **union (deduplicated)** | | **3,652,713** | **30.2%** | |

≈913K output tokens saved on this run, before counting the reasoning tokens spent *composing* that boilerplate.

### 1. Give nodes their derived geometry up front
`boundary` is read by 57.2% of calls and used as a polygon by 0.5% of them. It is a polygon API for a population that wants a rectangle. Add to every node: `bbox=(r0,c0,r1,c1)`, `centroid=(r,c)`, `h`, `w`. Keep `boundary` for the 65 calls that need it. This deletes 8,046 lines the agent writes in 245 slightly-different spellings — each spelling an independent chance to write `p[1]` where it meant `p[0]`.

### 2. Promote `action` to a first-class tool
26% of all python calls contain no inspection at all — they are a JSON-escaped Python string whose entire content is `action(['LEFT'])`. That is a code-execution round-trip used as a keypress. Expose `action` as its own tool taking `{"actions": ["LEFT"]}` or `{"actions": [{"action":"MOUSE","row":4,"col":7}]}`. Keep `action()` callable inside python for search loops (11% batches, plus in-loop use).

### 3. Return the node table automatically
48.9% of calls loop over nodes and print a summary — 62% of all code volume passes through this shape. The agent writes a formatter every turn to see the same thing. Render a compact node table in the tool result by default (id, color, pixels, bbox, children, hash), so `seg` needs no printing loop at all.

### 4. Query instead of filter
59% read `color`, 49% read `pixels`, and 32% write an explicit color filter. `seg.find(color='R', min_pixels=4)` and `seg.by_id(7)` replace the loop-and-if in 8,075 calls.

### 5–7. Smaller wins
- `diff(a, b)` returning appeared/disappeared/moved nodes: the prompt spends five bullets warning about `history[-1].frame` vs `previous_frame`, yet only 6.2% diff at all and only 1.1% touch `previous_frame`. A builtin makes the correct comparison the easy one and lets those five bullets be deleted.
- `frame.at(r,c)` / `frame.crop(...)`: 24.1% rebuild a 2D grid from `.ascii` purely to index it — despite the prompt forbidding whole-board scans. Cheap indexing removes the incentive.
- A persistent scratch namespace (or a small stdlib of `find_player`-style helpers): `find_player` was rewritten 172 times.

### Second-order effect on the prompt
The system prompt currently spends ~40 bullets teaching the runtime — history semantics, which frame is which, don't-print-boards, use-segmentation-not-ascii. Most of those bullets exist to police an API that makes the wrong thing easy. Items 1–4 let a large fraction of that text be deleted, which is a fixed saving on *every* turn of every episode, not just the turns that would have written boilerplate.

---

# Designing `seg.summary()`

Measured over the 10,477 calls (48.9%) that loop over nodes and print a summary.

## Table 11 — What the agent puts in its own summaries

| field | calls | % of summary loops | verdict |
|---|---|---|---|
| pixels | 6,767 | 64.6% | **always** |
| color | 6,274 | 59.9% | **always** |
| boundary | 5,360 | 51.2% | **always, but as bbox** |
| id | 4,985 | 47.6% | **always** |
| children | 977 | 9.3% | count only |
| hash | 652 | 6.2% | implicit (grouping key) |
| adjacency | 404 | 3.9% | on demand |

Geometry actually emitted: full boundary polygon dumped **50.4%**, bbox 12.7%, centroid 11.8%, h×w 0.8%.

## Table 12 — Sizing

| measure | median | p90 | p99 | max |
|---|---|---|---|---|
| nodes per frame | 11 | 28 | 60 | 94 |
| distinct colors | 5 | 7 | — | 12 |
| node size (px) | 12 | 126 | 3,794 | 3,952 |

Nodes ≤2 px: 20.1% of all nodes. Nodes ≥500 px: 5.6%.

## Table 13 — Two findings that drive the design

| finding | value |
|---|---|
| summary loops that **filter before printing** | 8,241 (78.7%) |
| summary loops that print every node | 2,236 (21.3%) |
| nodes that are a **shape-duplicate** of another node in the same frame | **41.1%** |
| frames where ≥30% of nodes are duplicate shapes | 203 / 315 |
| mean rows per frame: raw → collapsed by hash | **24.4 → 10.9** |
| result-stream chars that are raw `[r,c]` pairs | 1,762,712 (8.4%, ~441K tokens) |

**A naive full dump would be a regression.** The agent already filters 78.7% of the time; replacing a filtered print with an unfiltered table adds tokens. The win comes from collapsing duplicates, not from dumping everything.

## Proposed format

```
L3 s=87 64x64 | 24 obj -> 11 shapes, 6 colors
 id  c    px  bbox         ctr    ch
  0  w  3920   0,0-63,63   31,31   6
  4  b    28  12,8-15,11   13,9    1
 x5  W     1  (5 same) @ 3,7 3,19 9,7 9,19 14,7
```

- One row per **distinct shape-hash**, not per node. Repeated shapes collapse to an `xN` row listing centroids — this is the 41.1% finding and takes 24.4 rows to 10.9.
- `bbox` as `r0,c0-r1,c1` replaces the boundary polygon. Polygons are used as polygons in 0.5% of reads; keep `node.boundary` available for those.
- `ch` is a child **count**; `seg.tree(id)` expands it. Only 9.3% need children, and almost none need the full id list.
- Hash is the grouping key but is never printed — 16 hex chars × 24 rows is pure cost at 6.2% explicit use. Expose as `node.hash`.
- Adjacency omitted (3.9%); `seg.adj(id)` on demand.
- Header carries level, step, shape, and counts — the frame context the agent otherwise re-prints.

Measured against 172 real frames: median result **2,060 → 1,013 chars (−49%)**.

## Signature

```python
seg.summary()                  # collapsed table (default)
seg.summary(full=True)         # one row per node, no collapsing
seg.summary(color='R')         # filtered — replaces the 78.7% hand-filter
seg.summary(min_px=3)          # drop the 20.1% of <=2px noise nodes
seg.summary(vs=previous_frame) # +added / -removed / ~moved only
```

`vs=` matters most: only 6.2% of calls diff at all and 1.1% touch `previous_frame`, despite the prompt spending five bullets on which frame is which. Making the diff a one-word argument is what actually retires those bullets.

## Do not include by default

Row cap at ~40 with an explicit `... N more (use full=True)` line — the p99 frame is 60 nodes and max is 94, so the tail is real but rare, and silent truncation would read as "that's the whole board" when it isn't.

---

# Designing `seg.find()`

Measured over the 8,796 calls (41.1%) containing node predicates.

## Table 14 — Predicate kinds the agent writes

| predicate | calls | % of predicate calls |
|---|---|---|
| `color ==` | 4,633 | 52.7% |
| `color in {set}` | 2,874 | 32.7% |
| `pixels ==` | 2,163 | 24.6% |
| `pixels > / >=` | 1,937 | 22.0% |
| `pixels < / <=` | 1,266 | 14.4% |
| position/bbox test | 839 | 9.5% |
| `color not in {set}` | 476 | 5.4% |
| `color !=` | 359 | 4.1% |
| `id in {set}` | 312 | 3.5% |
| children test | 256 | 2.9% |
| `id ==` | 152 | 1.7% |
| `hash ==` | 84 | 1.0% |

Top combos: `color==` alone ×1,570 · `color in set` ×1,243 · `color== AND pixels==` ×797 · `color== AND pixels>=` ×363.

## Table 15 — What happens to matches (10,180 filter calls)

| use | % |
|---|---|
| extract position (bbox/centroid) | **97.5%** |
| count matches | 23.2% |
| take first match | 17.4% |
| collect list | 6.4% |

The 172 `find_player` bodies: 100% test `color==`, 99% `pixels==`, 77% a children test.

## Signature

```python
seg.find(color='b')                 # color str, or set: color={'b','S'}
seg.find(color='b', px=24)          # exact size — the find_player idiom
seg.find(color='R', min_px=4)       # size range: min_px= / max_px=
seg.find(not_color={'W','B'})       # negative filter (9.5% combined)
seg.find(hash='e67a64de5c2f1162')   # track object across frames
seg.find(in_bbox=(10,0,40,63))      # region restriction (9.5%)
```

Returns a list of node objects that already carry `.bbox`, `.centroid`, `.color`, `.px`, `.id`, `.hash` — because 97.5% of filter calls exist to extract a position, the return type must make position free.

Convenience accessors on the result list:
```python
seg.find(color='b', px=24).one()    # asserts exactly 1 match, returns node (find-the-player, 1,317 calls)
seg.find(color='R').first()         # first in top-left order (17.4%)
len(seg.find(color='S'))            # count (23.2%)
```

`.one()` matters: silent first-match on 2 candidates is a classic wrong-player bug; asserting uniqueness converts it into a visible error.

## Before / after (real call from `g50t-5849a774`)

Before — 377 chars, 12 lines:
```python
# Check player position
seg = current_frame.segmentation
nodes = seg['nodes']
for n in nodes:
    if n['color'] == 'b' and n['pixels'] == 24:
        b = n['boundary']
        min_r = min(p[0] for p in b)
        min_c = min(p[1] for p in b)
        max_r = max(p[0] for p in b)
        max_c = max(p[1] for p in b)
        print(f"Player: ({min_r},{min_c})-({max_r},{max_c})")
```

After — 62 chars, 1 line:
```python
print(seg.find(color='b', px=24).one().bbox)
```

## Out of scope, deliberately

- children/containment predicates (2.9%) — expose `node.children` and let the rare call filter in Python.
- adjacency queries (2.3%) — `seg.adj(id)` if anything.
- arbitrary lambdas — `seg.find(where=lambda n: ...)` adds API surface for cases plain Python already handles; the point is covering the 90% with keywords.
