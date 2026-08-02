"""Smoke test for the dual-model (world_model + planner) training design.

Tests the design's mechanical data claims against one logged run:

 1. EXTRACT  — (obs, action, obs') transitions are recoverable from the
    ``*_events.jsonl`` sidecars (including ACTION6 click coordinates,
    which are only present inside ``action_display``).
 2. REPLAY   — the offline arcengine env deterministically reproduces the
    logged boards when the extracted action sequence is re-stepped
    (exact ``frame[-1]`` match per action). This is the load-bearing
    claim behind branched data generation and free prediction-check
    filtering of hindsight labels.
 3. DIFF     — frame-diffs between consecutive boards are small relative
    to the full 64x64 board, so diff-prediction targets are cheap to
    decode compared to full-frame prediction.
 4. SIGNAL   — the logs carry belief-relevant material: no-op actions
    (board_changed=false), level transitions, game_overs, and full
    analysis transcripts usable as hindsight-relabeling context.

Usage (venv python of ARC3-Inference so arc_agi/arcengine import):

    .venv/bin/python dual_model/smoke/replay_smoke.py \
        --run-dir ARC3-Inference/runs/20260720_134304_ar25-10-pass \
        --environments-dir /path/to/environment_files \
        --game ar25 [--episode p2] [--json-out results.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_MOUSE_RE = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")

# Quiet the Arcade's self-installed stdout INFO handler (same trick as
# taaf.game_api: pass a private logger so it takes the "user provided" branch).
_SMOKE_LOGGER = logging.getLogger("dual_model.smoke.arcade")
_SMOKE_LOGGER.setLevel(logging.WARNING)


# --- extraction --------------------------------------------------------------


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_action(evt: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """Map a logged action event to (engine action name, step data).

    ACTION6 coordinates are not structured fields in the event — they only
    survive inside ``action_display`` as ``MOUSE(row=R, col=C)``. The engine
    expects ``{"x": col, "y": row}`` (see solver.py ``data = {"x": column,
    "y": row}``).
    """
    name = evt["action_name"]
    if name == "ACTION6":
        m = _MOUSE_RE.search(evt.get("action_display") or "")
        if not m:
            raise ValueError(f"ACTION6 event without parsable coords: {evt.get('action_display')!r}")
        return name, {"x": int(m.group(2)), "y": int(m.group(1))}
    return name, {}


def extract_transitions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(board_before, action, board_after) triples from the event stream.

    The stream is: one ``initial`` event (board after make-time RESET), then
    interleaved ``action`` events (board AFTER that action) and ``analysis``
    events (LLM transcript, board unchanged). board_before of action k is the
    board of the closest preceding initial/action event.
    """
    transitions: list[dict[str, Any]] = []
    prev_board: list[list[int]] | None = None
    prev_meta: dict[str, Any] | None = None
    for evt in events:
        etype = evt.get("type")
        if etype == "initial":
            prev_board = evt["board"]
            prev_meta = evt
        elif etype == "action":
            if prev_board is None:
                raise ValueError("action event before initial event")
            name, data = parse_action(evt)
            transitions.append(
                {
                    "action_name": name,
                    "action_data": data,
                    "action_display": evt.get("action_display"),
                    "board_before": prev_board,
                    "board_after": evt["board"],
                    "board_changed": evt.get("board_changed"),
                    "level_before": prev_meta.get("level") if prev_meta else None,
                    "level_after": evt.get("level"),
                    "score_after": evt.get("score"),
                    "level_completed": evt.get("level_completed"),
                    "game_over": evt.get("game_over"),
                    "action_num": evt.get("action_num"),
                }
            )
            prev_board = evt["board"]
            prev_meta = evt
    return transitions


# --- replay ------------------------------------------------------------------


def to_grid(frame_like: Any) -> list[list[int]]:
    rows = frame_like.tolist() if hasattr(frame_like, "tolist") else frame_like
    return [[int(c) for c in row] for row in rows]


def replay_episode(
    game: str, environments_dir: str, transitions: list[dict[str, Any]], initial_board: list[list[int]]
) -> dict[str, Any]:
    """Re-step the extracted actions through a fresh offline env; compare boards."""
    import arc_agi
    import arcengine

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=environments_dir,
        logger=_SMOKE_LOGGER,
    )
    scorecard_id = arcade.create_scorecard()
    env = arcade.make(game, scorecard_id=scorecard_id)
    if env is None:
        raise RuntimeError(f"arcade.make({game!r}) returned None")
    # Mirror taaf.game_api: set AFTER make so the make-time RESET still
    # full-resets; in-episode RESETs then reset the current level only.
    os.environ["ONLY_RESET_LEVELS"] = "true"

    result: dict[str, Any] = {"n_actions": len(transitions)}
    initial = env.observation_space
    initial_match = to_grid(initial.frame[-1]) == initial_board
    result["initial_board_match"] = initial_match

    matches = 0
    first_divergence: dict[str, Any] | None = None
    for i, tr in enumerate(transitions):
        action_id = arcengine.GameAction.from_name(tr["action_name"])
        resp = env.step(action_id, data=dict(tr["action_data"]))
        if resp is None or not resp.frame:
            first_divergence = first_divergence or {
                "step": i,
                "action": tr["action_display"],
                "reason": "engine returned empty frame",
            }
            break
        got = to_grid(resp.frame[-1])
        want = tr["board_after"]
        if got == want:
            matches += 1
        elif first_divergence is None:
            ncell = sum(
                1 for r in range(len(want)) for c in range(len(want[0])) if got[r][c] != want[r][c]
            )
            first_divergence = {
                "step": i,
                "action_num": tr["action_num"],
                "action": tr["action_display"],
                "cells_differing": ncell,
            }
    result["exact_matches"] = matches
    result["match_rate"] = matches / len(transitions) if transitions else None
    result["first_divergence"] = first_divergence
    return result


# --- diff / signal stats -----------------------------------------------------


def diff_cells(b1: list[list[int]], b2: list[list[int]]) -> list[tuple[int, int, int, int]]:
    return [
        (r, c, b1[r][c], b2[r][c])
        for r in range(len(b1))
        for c in range(len(b1[0]))
        if b1[r][c] != b2[r][c]
    ]


def episode_stats(events: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> dict[str, Any]:
    changed_counts = [len(diff_cells(t["board_before"], t["board_after"])) for t in transitions]
    nonzero = [n for n in changed_counts if n > 0]
    board_cells = (
        len(transitions[0]["board_before"]) * len(transitions[0]["board_before"][0])
        if transitions
        else 0
    )
    # Char-based (tokenizer-agnostic) size comparison: full board serialized as
    # ascii rows (the harness's own text rendering) vs a per-cell diff line
    # "(r,c) old->new".
    full_board_chars = board_cells + len(transitions[0]["board_before"]) if transitions else 0
    diff_chars = [sum(len(f"({r},{c}) {a}->{b} ") for r, c, a, b in
                     diff_cells(t["board_before"], t["board_after"])) for t in transitions]

    analysis_events = [e for e in events if e.get("type") == "analysis"]
    transcript_chars = sum(len(e.get("transcript") or "") for e in analysis_events)

    levels = [t["level_after"] for t in transitions if t.get("level_after") is not None]
    return {
        "n_events": len(events),
        "n_actions": len(transitions),
        "n_analysis_events": len(analysis_events),
        "transcript_chars_total": transcript_chars,
        "action_histogram": dict(Counter(t["action_name"] for t in transitions)),
        "noop_actions": sum(1 for t in transitions if t.get("board_changed") is False),
        "level_completions": sum(1 for t in transitions if t.get("level_completed")),
        "game_overs": sum(1 for t in transitions if t.get("game_over")),
        "max_level": max(levels) if levels else None,
        "board_cells": board_cells,
        "cells_changed_mean": round(statistics.mean(changed_counts), 1) if changed_counts else None,
        "cells_changed_median": statistics.median(changed_counts) if changed_counts else None,
        "cells_changed_p90": (
            sorted(changed_counts)[int(0.9 * (len(changed_counts) - 1))] if changed_counts else None
        ),
        "cells_changed_max": max(changed_counts) if changed_counts else None,
        "cells_changed_nonzero_mean": round(statistics.mean(nonzero), 1) if nonzero else None,
        "full_board_chars": full_board_chars,
        "diff_chars_mean": round(statistics.mean(diff_chars), 1) if diff_chars else None,
        "diff_chars_p90": (
            sorted(diff_chars)[int(0.9 * (len(diff_chars) - 1))] if diff_chars else None
        ),
    }


def transcript_section_headers(events: list[dict[str, Any]], limit: int = 30) -> list[str]:
    """Candidate section markers inside the first analysis transcript — the
    hindsight-relabeling context we claim exists (world-model text is mingled
    in here)."""
    for evt in events:
        if evt.get("type") == "analysis" and evt.get("transcript"):
            headers = []
            for line in evt["transcript"].splitlines():
                s = line.strip()
                if s.startswith("[") and s.endswith("]") and len(s) < 80:
                    headers.append(s)
                if len(headers) >= limit:
                    break
            return headers
    return []


# --- main --------------------------------------------------------------------


def find_episode_files(run_dir: Path, game: str) -> list[Path]:
    return sorted(run_dir.glob(f"passes/*/artifacts/{game}*_events.jsonl"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--environments-dir", required=True)
    ap.add_argument("--game", required=True, help="engine env name, e.g. 'ar25'")
    ap.add_argument("--episode", default=None, help="only this episode label, e.g. 'p2'")
    ap.add_argument("--json-out", default=None, type=Path)
    args = ap.parse_args()

    files = find_episode_files(args.run_dir, args.game)
    if args.episode:
        files = [f for f in files if f.name.endswith(f"_{args.episode}_events.jsonl")]
    if not files:
        print(f"no *_events.jsonl found under {args.run_dir}", file=sys.stderr)
        return 2

    all_results: list[dict[str, Any]] = []
    for path in files:
        # passes/<J>/artifacts/<game>_pN_events.jsonl — pN restarts at 0 in
        # each job dir, so the label needs the job index to be unique.
        label = f"job{path.parts[-3]}/{path.name.replace('_events.jsonl', '')}"
        events = load_events(path)
        transitions = extract_transitions(events)
        initial_board = next(e["board"] for e in events if e.get("type") == "initial")
        stats = episode_stats(events, transitions)
        replay = replay_episode(args.game, args.environments_dir, transitions, initial_board)
        res = {"episode": label, "stats": stats, "replay": replay}
        all_results.append(res)

        print(f"\n=== {label} ===")
        print(
            f"  extract: {stats['n_actions']} transitions from {stats['n_events']} events "
            f"({stats['n_analysis_events']} analysis events, "
            f"{stats['transcript_chars_total'] / 1e6:.1f}M transcript chars)"
        )
        print(f"  actions: {stats['action_histogram']}")
        print(
            f"  signal:  noop={stats['noop_actions']} level_completions={stats['level_completions']} "
            f"game_overs={stats['game_overs']} max_level={stats['max_level']}"
        )
        print(
            f"  diffs:   cells changed mean={stats['cells_changed_mean']} "
            f"median={stats['cells_changed_median']} p90={stats['cells_changed_p90']} "
            f"max={stats['cells_changed_max']} (board={stats['board_cells']} cells); "
            f"diff chars mean={stats['diff_chars_mean']} p90={stats['diff_chars_p90']} "
            f"vs full board ~{stats['full_board_chars']} chars"
        )
        print(
            f"  replay:  initial_match={replay['initial_board_match']} "
            f"exact {replay['exact_matches']}/{replay['n_actions']} "
            f"(rate={replay['match_rate']:.3f})"
            + (f" first_divergence={replay['first_divergence']}" if replay["first_divergence"] else "")
        )

    headers = transcript_section_headers(load_events(files[0]))
    print(f"\ntranscript section markers (first analysis event of {files[0].name}):")
    for h in headers:
        print(f"  {h}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.json_out}")

    n_ok = sum(1 for r in all_results if r["replay"]["match_rate"] == 1.0)
    print(f"\nSUMMARY: {n_ok}/{len(all_results)} episodes replay with 100% exact board match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
