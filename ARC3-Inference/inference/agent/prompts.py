"""Prompt templates for the analyzer agent."""

from inference.utils.grid_utils import ARC_COLOR_LEGEND

TOOL_CALL_FORMAT_GUIDANCE = (
    "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
    "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
    "Do not quote or place tool-call markup inside explanatory text; when you decide to call the tool, emit the tool call itself."
)

GAME_OVERVIEW_ADDENDUM = (
    "\n\nGame overview:\n"
    "- You are solving a multi-level grid puzzle game.\n"
    "- You are called repeatedly over the course of a run. Treat each turn as one observe-plan-act cycle: re-understand the current state from the newest frame, update your working world model, choose the next best action or short sequence against the goal as currently understood, execute it, and expect to re-evaluate on the next turn from the updated state.\n"
    "- Your job is to solve the entire game by clearing every level, not just the current screen.\n"
    "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n"
    "- Optimize for as few in-game actions as possible while still being reliable.\n"
    "- In this environment, boards are presented as 64 x 64 color grids rendered with ARC color symbols.\n"
    f"- Color legend: {ARC_COLOR_LEGEND}.\n"
)

VISUAL_GAME_ADDENDUM = (
    "\n\nVisual-game guidance:\n"
    "- Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.\n"
    "- Game entities are usually rendered as connected multi-tile shapes such as 2×2, 2×3, 3×3, or longer patterned structures. Sometimes they might also be 1x1 tokens.\n"
    "- Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board configuration.\n"
    "- Background colors are often white or gray/black-ish large regions, but not always. Verify background hypotheses by area, stability, and object boundaries rather than assuming them.\n"
    "- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar. It often shrinks or changes each step. If you identify such a bar, do not get distracted by it or treat it as core gameplay state unless there is concrete evidence that it interacts with the puzzle mechanics.\n"
    "- A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces. If a repeated strip of small blocks sits flush against the top, bottom, left, or right border and actions only change that strip while the interior board stays the same, classify it as HUD/timer state, not as an object to click through segment by segment. DON'T DO THIS!\n"
    "- Use coordinates only to target actions or describe local evidence. Do not frame the objective as reaching a specific absolute row or column.\n"
    "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n"
    "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n"
    "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n"
    "- When one action changes several objects at once, or the same action seems to behave differently across levels, look for one deeper mechanic that explains all the changes together\n" # RL idea
    "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n"
)

STRUCTURED_RUNTIME_STATE_ADDENDUM = (
    "\n\nRuntime variables inside every `python` tool call:\n"
    "- `current_frame` is a lightweight frame view for the latest environment state. It exposes only `.ascii` (the board as one newline-delimited string of ARC color chars), `.step`, `.level`, `.shape` (a `(rows, cols)` tuple), and `.segmentation`.\n"
    "- `current_frame.segmentation` parses the board into objects. It returns `{'nodes': [...], 'adjacency_list': [...]}`.\n"
    "- Each node is one 4-connected same-color object with fields: `id` (top-most-left-most order), `color` (ARC color char), `hash`, `pixels`, `bbox` (`[r0, c0, r1, c1]`, inclusive), `centroid` (`[r, c]`), `h`/`w`, `children` (ids of objects fully enclosed by this one). When the exact outline or shape matters, use `boundary` (outer perimeter as clockwise `[r, c]` corner points) plus `holes` (one corner ring per enclosed hole; `[]` if solid) -- together they describe the region exactly; use `bbox`/`centroid` when position is enough.\n"
    "- `segmentation['adjacency_list']` is a list of `[i, j]` node-id pairs whose objects share an edge.\n"
    "- Find objects with e.g. `segmentation.find(color='B', px=24).one()` -- `.one()` errors unless exactly one match, so use it whenever you expect a unique object. `segmentation.find(color='R')` returns a plain list in id order (`.first()` for the top-left-most). Other keywords: `not_color=`, `min_px=`/`max_px=`, `in_bbox=(r0, c0, r1, c1)`, `hash=`; color keywords accept a char or a set.\n"
    "- Nodes are per-frame snapshots; identity across frames is `hash` (position-invariant: same color+shape, anywhere). After `action(...)`, re-find in the fresh segmentation, e.g. `seg.find(hash=h).one()`.\n"
    "- The raw numeric grid is intentionally not exposed. Use `current_frame.segmentation` as your primary view of the board -- objects, colors, shapes, containment, adjacency, and cross-frame object hashes. Use `current_frame.ascii` only to read a small, specific region; do not scan the whole board with it.\n"
    "- `history` is a chronological list of action/frame snapshots.\n"
    "- `history` is a Python list of objects, not a dict.\n"
    "- Each history entry exposes only `.action` and `.frame`; entries are not subscriptable like `entry['action']`.\n"
    "- Each `history[i].frame` is the frame after `history[i].action`, and is the same frame-view type as `current_frame`.\n"
    "- Important history semantics: when `history` is non-empty, `history[-1].frame` is the same latest/post-action board as `current_frame`. It is not the previous board. To inspect the state before the latest action, use `previous_frame` or `history[-2].frame` when available.\n"
    "- `previous_frame` is the frame before the most recent real environment action, or `None` if no previous frame is available.\n"
    "- `last_action` is the most recent real environment action name/display, or `None` before any real action.\n"
    "- `transitions` is a chronological list of actual action transitions, excluding the initial seeded frame. Each transition exposes `.action`, `.before_frame`, `.after_frame`, `.frame` (alias of `.after_frame`), and `.result`.\n"
    "- `last_transition` is `transitions[-1]` or `None`. Its `.result` mirrors `last_action_result`; older transitions may have an empty `.result`.\n"
    "- Use `last_transition.diff` for the latest cell-level change; any transition in `transitions` exposes the same `.diff`. It returns `cells_changed` plus `groups` (largest first), each with `from`, `to`, `count`, inclusive `bbox` `[[r0, c0], [r1, c1]]`, and `cells`. Printing folds `cells` for groups larger than 12, but the full list stays accessible as `group['cells']`.\n"
    "- `last_action_result` is the persisted result dict from the most recent `action(...)` call (`{}` before any action; it survives inspection-only calls). Keys include `board_changed`, `done`, `level_completed`, `game_over`, `run_complete`, `reward`, and `valid_actions`.\n"
    "- `valid_actions` is the current list of valid action names.\n"
    "- Call `action(actions)` to execute one or more real environment actions from Python.\n"
    "- Pass `action(actions)` a list like `['LEFT']` or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]`.\n"
    "- One action usually returns one frame, but a single action can result in a short multi-frame animation.\n"
    "- After `action(actions)` returns, `current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, and `last_action_result` are refreshed.\n"
)

MULTIMODAL_CONTEXT_ADDENDUM = (
    "\n\nMultimodal context:\n"
    "- User turns include an attached image of the current ARC grid.\n"
    "- The image and `current_frame.ascii` are two representations of the same current frame.\n"
    "- You can use images and other tools to understand the game state and guide your strategy, each may be useful depending on the current uncertainty.\n"
)

PYTHON_ADDENDUM = (
    "\n\nPython tool guidance:\n"
    "- A typical inspect-act-verify call looks like:\n"
    "```python\n"
    "d = last_transition.diff\n"
    "print(d['cells_changed'], d['groups'][:3])          # what did the last action change?\n"
    "seg = current_frame.segmentation\n"
    "player = seg.find(color='B', px=24).one()\n"
    "print(player['centroid'], player['bbox'])\n"
    "action(['LEFT'])\n"
    "print(last_transition.diff['groups'][:3])           # did the move do what the model predicted?\n"
    "```\n"
    "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n"
    "- The only importable standard-library modules are: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string.\n"
    "- Call `python` with one ephemeral `code` string.\n"
    "- Always inspect `current_frame`, `history`, and `valid_actions` from Python instead of reasoning from the raw board by eye.\n"
    "- IMPORTANT: when the objective is understood but the best action order is unclear, write an explicit search over the inferred state space instead of guessing moves -- BFS by default; fall back to a custom scorer or heuristic only when BFS does not fit the problem.\n"
    "- Optimize for the shortest reliable sequence that advances the current goal as described by your world model. If confidence is low, program a discriminating probe and revise the world model from the result; once the important state variables and action effects are understood, stop probing and search in the inferred state space.\n"
    "- Never print or echo full board frames. It is fine to write a lot of Python, but return only compact decision-oriented summaries -- object lists, diffs, coordinates, counts, or tiny local crops.\n"
    "- A strong default loop is: read `last_transition.diff`, summarize the relevant objects, infer the desired environment change, write a small scorer or search over candidate sequences, execute the best probe or plan with `action(...)`, then inspect again until you understand exactly what changed.\n"
    "- For object tracking across frames, match by `hash` first; when a shape morphs (its hash changes), fall back to color, overlap, bounding-box proximity, area change, and edge contact rather than exact coordinates alone.\n"
    "- After every action, verify whether gameplay objects changed or whether only a timer, progress bar, or remaining-step bar moved. Do not treat HUD-only changes as evidence that the move worked.\n"
    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\n"
    "- Call `action(...)` inside Python rather than returning action text in the chat.\n"
    "- `action(...)` accepts an ordered list of actions and may be called repeatedly in one snippet, including inside loops; each call refreshes the preloaded variables before execution continues. Once your code has found a reliable sequence, batch it.\n"
    "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n"
)

COMPACT_TOOL_SESSION_ADDENDUM = (
    "\n\nTool session rules:\n"
    "- {tool_inventory}\n"
    f"- {TOOL_CALL_FORMAT_GUIDANCE}\n"
    "- You can call the `python` tool as many times as you want per step. Investigate until your code has a clear probe or plan.\n"
    "- Do not ration tool calls when the state is unclear; spend extra calls to confirm what actually changed.\n"
    "- After `action(...)` returns, the structured runtime state is refreshed before the next Python statement and before the next tool call. Inspection-only Python calls do not clear `last_action_result`.\n"
    "- Each `python` tool call has a hard time limit of 30 seconds.\n"
    "- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool result will tell you that.\n"
    "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n"
)


MODEL_UPDATE_TOOL_ADDENDUM = (
    "\n\nPersistent memory updates:\n"
    "- `update_memory` saves your models and notes across turns: the world, goal, and action models plus cross-level notes, open questions, recent findings, and plan.\n"
    "- Call it when the models you are carrying are no longer consistent with what you recently observed, or when your understanding of the game changes.\n"
)
