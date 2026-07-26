from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import time

from inference.distillation.teacher import VllmTeacherScorer

POLL_SECONDS = 10.0


def _game_id(path: Path) -> str:
    return path.name.split("_", 1)[0]


def _score_lines(
    lines: list[str], scorer: VllmTeacherScorer, workers: int
) -> list[str]:
    """Score complete JSONL records, preserving input order.

    Order matters: load_game_episodes() accumulates rows in file order and
    computes discounted returns backwards over that sequence, so a reordered
    file silently corrupts the advantages.
    """
    rows = [json.loads(line) for line in lines]
    if workers > 1 and len(rows) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scored = list(pool.map(scorer.score_game_turn, rows))
    else:
        scored = [scorer.score_game_turn(row) for row in rows]
    out: list[str] = []
    for row, logprobs in zip(rows, scored, strict=True):
        row["teacher_logprobs"] = logprobs
        out.append(json.dumps(row, ensure_ascii=True) + "\n")
    return out


def _consume(source: Path, offset: int) -> tuple[list[str], int]:
    """Read whole lines appended since `offset`; leave any partial tail."""
    size = source.stat().st_size
    if size <= offset:
        return [], offset
    with source.open("rb") as handle:
        handle.seek(offset)
        blob = handle.read(size - offset)
    end = blob.rfind(b"\n")
    if end < 0:
        return [], offset
    text = blob[: end + 1].decode("utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    return lines, offset + end + 1


def score_directory(
    input_dir: Path,
    output_dir: Path,
    scorer: VllmTeacherScorer,
    *,
    workers: int = 1,
    games: set[str] | None = None,
    done_marker: Path | None = None,
) -> None:
    """Add teacher logprobs to captured rollouts.

    Without `done_marker` this is a single pass over whatever is already on
    disk. With one, it tails the directory and scores records as the collect
    job appends them, finishing once the marker exists and every file has been
    drained -- so teacher scoring overlaps rollout collection instead of
    waiting a full collect wallclock for it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets: dict[Path, int] = {}
    totals: dict[str, int] = {}

    while True:
        collect_finished = done_marker is None or done_marker.exists()
        progressed = False
        for source in sorted(input_dir.glob("*.jsonl")):
            if games is not None and _game_id(source) not in games:
                continue
            lines, new_offset = _consume(source, offsets.get(source, 0))
            if not lines:
                continue
            offsets[source] = new_offset
            progressed = True
            with (output_dir / source.name).open("a", encoding="utf-8") as writer:
                writer.writelines(_score_lines(lines, scorer, workers))
            totals[source.name] = totals.get(source.name, 0) + len(lines)
            print(
                f"scored {len(lines)} records from {source.name} "
                f"(total {totals[source.name]})",
                flush=True,
            )
        if done_marker is None:
            break
        if collect_finished and not progressed:
            break
        if not progressed:
            time.sleep(POLL_SECONDS)

    manifest = input_dir / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, output_dir / manifest.name)
    if not totals:
        print("no rollout records scored", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"scored {sum(totals.values())} records across {len(totals)} episodes",
        flush=True,
    )


def _load_games(path: Path | None, key: str) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload[key])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add teacher logprobs to captured Duck game rollouts"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-GPTQ-Int4")
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent scoring requests; keep <= the teacher's --max-num-seqs",
    )
    parser.add_argument(
        "--games-file",
        type=Path,
        default=None,
        help="JSON with a 'train' list; only those games are scored",
    )
    parser.add_argument(
        "--done-marker",
        type=Path,
        default=None,
        help="tail the input directory until this file appears and all records drain",
    )
    args = parser.parse_args()
    score_directory(
        args.input,
        args.output,
        VllmTeacherScorer(
            base_url=args.base_url,
            model=args.model,
            api_key=os.environ.get(args.api_key_env, "EMPTY"),
        ),
        workers=args.workers,
        games=_load_games(args.games_file, "train"),
        done_marker=args.done_marker,
    )


if __name__ == "__main__":
    main()
