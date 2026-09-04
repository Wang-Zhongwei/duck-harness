"""How many board images does a request actually carry, and what did they cost?

The context trimmer's budget is denominated in estimated tokens, so the number
of images retained in a request sets how much of that budget the old
length-based estimator wasted. ``ba48881`` replaced the estimate with the
processor's own arithmetic; this script measures the size of the correction
per game, which is what a compensating context-window change has to match.

Two measurements, joined per game:

  images/request  Counted from the run's prompt logs. ``_build_user_message``
                  appends the literal "Current grid image:" to a user prompt
                  exactly when it attaches an image, so occurrences of that
                  string in a rendered log equal the image count. The logs
                  drop image parts (``_normalize_message_content`` keeps only
                  ``type == "text"``), so this marker is the only handle.

  tokens/image    Computed from the run's real boards in events.jsonl,
                  rendered at the run's upscale, priced both ways: the naive
                  ``(len(json)+2)//3`` over the base64 payload, and the exact
                  vision-token count from ``image_part_prompt_tokens``.

CAVEAT: a prompt log is a "LATEST MODEL CALL SNAPSHOT" -- it is opened with
mode "w" once per model call, so only the final request of each game-run
survives. Every row is therefore one late-game observation, which is when
context is fullest. Read the numbers as a near-steady-state ceiling, not as a
per-turn average over the whole run.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

IMAGE_MARKER = "Current grid image:"


def _load_agent_module(repo_root: Path):
    """Import the harness's own vision helpers so pricing matches production."""
    inference_root = repo_root / "ARC3-Inference"
    sys.path.insert(0, str(inference_root))
    from inference.agent.vision_context import (  # noqa: E402
        frame_to_png_data_url,
        image_part_prompt_tokens,
    )
    from inference.agent.runtime_state import Frame  # noqa: E402

    return frame_to_png_data_url, image_part_prompt_tokens, Frame


def naive_estimate(part: dict) -> int:
    """The pre-ba48881 estimator, applied to one image part."""
    rendered = json.dumps(part, ensure_ascii=True, sort_keys=True, default=str)
    return max(1, (len(rendered) + 2) // 3)


def count_images(log_path: Path) -> tuple[int, int, int]:
    """(images, user_messages, message_count) for one prompt log."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    images = text.count(IMAGE_MARKER)
    users = sum(1 for line in text.splitlines() if line.rstrip() == "[USER]")
    message_count = 0
    for line in text.splitlines():
        if line.startswith("message_count:"):
            message_count = int(line.split(":", 1)[1].strip())
            break
    return images, users, message_count


def sample_boards(events_path: Path, limit: int) -> list[list[list[int]]]:
    """Up to ``limit`` real boards, spread across the game rather than bunched
    at the start -- board complexity drifts as a level fills in, and the naive
    estimate tracks PNG compressibility, so an early-only sample understates."""
    boards: list[list[list[int]]] = []
    with open(events_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            board = event.get("board")
            grid = board.get("grid") if isinstance(board, dict) else board
            if isinstance(grid, list) and grid and isinstance(grid[0], list):
                boards.append(grid)
    if len(boards) <= limit:
        return boards
    stride = len(boards) / limit
    return [boards[int(i * stride)] for i in range(limit)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="runs/local/<run> directory")
    parser.add_argument("--pass-index", type=int, default=0)
    parser.add_argument("--upscale", type=int, default=4,
                        help="MULTIMODAL_UPSCALE the run deployed with")
    parser.add_argument("--boards-per-game", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True, help="output CSV")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    to_png, exact_tokens, Frame = _load_agent_module(repo_root)

    pass_dir = args.run_dir / "passes" / str(args.pass_index)
    prompts_dir = pass_dir / "prompts"
    artifacts_dir = pass_dir / "artifacts"
    if not prompts_dir.is_dir():
        parser.error(f"no prompts/ under {pass_dir}")

    rows = []
    for log_path in sorted(prompts_dir.glob("*.log")):
        stem = log_path.stem                     # e.g. ar25-0c556536_p0
        game = stem.rsplit("_p", 1)[0]
        images, users, message_count = count_images(log_path)

        events_path = artifacts_dir / f"{stem}_events.jsonl"
        naive_mean = exact = None
        if events_path.is_file():
            boards = sample_boards(events_path, args.boards_per_game)
            naives, exacts = [], []
            for grid in boards:
                url = to_png(Frame(grid=grid, step=0, level=0), upscale=args.upscale)
                part = {"type": "image_url", "image_url": {"url": url}}
                naives.append(naive_estimate(part))
                exacts.append(exact_tokens(part))
            if naives:
                naive_mean = statistics.mean(naives)
                exact = exacts[0]                # constant for a fixed grid+upscale

        rows.append({
            "game": game,
            "trial": stem.rsplit("_p", 1)[1] if "_p" in stem else "",
            "images_in_request": images,
            "user_messages": users,
            "message_count": message_count,
            "log_bytes": log_path.stat().st_size,
            "naive_tokens_per_image": round(naive_mean, 1) if naive_mean else "",
            "exact_tokens_per_image": exact if exact else "",
            "wasted_tokens_per_image": round(naive_mean - exact, 1) if naive_mean and exact else "",
            "wasted_tokens_in_request": round((naive_mean - exact) * images, 1) if naive_mean and exact else "",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
