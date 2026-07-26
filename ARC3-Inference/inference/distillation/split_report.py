"""Report distillation progress split by train vs held-out games.

The rollout job plays all 25 official games; only the 20 train games feed the
teacher and the optimizer. This splits a run's per-game scores accordingly, so
the held-out mean -- the number that actually says whether distillation
generalized -- is never mixed with games the student trained on.

  python -m inference.distillation.split_report [--run RUN_DIR ...]

With no --run it reports every runs/distill-iteration-* directory in order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAMES_FILE = REPO_ROOT / "configs" / "distill.games.json"


def _per_game_scores(run_dir: Path) -> dict[str, float]:
    """Mean score per game id, from whichever summary the run wrote."""
    for name in ("score.json", "benchmark.json"):
        path = run_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        games = payload.get("games")
        if not isinstance(games, dict):
            continue
        scores: dict[str, float] = {}
        for game_id, entry in games.items():
            if isinstance(entry, dict) and "score" in entry:
                value = entry["score"]
                if isinstance(value, dict):
                    value = sum(value.values()) / len(value) if value else 0.0
                scores[game_id] = float(value)
        if scores:
            return scores
    return {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def report(run_dirs: list[Path], games_file: Path) -> None:
    split = json.loads(games_file.read_text(encoding="utf-8"))
    train_ids, eval_ids = set(split["train"]), set(split["eval"])
    reference = split.get("eval_reference_scores", {})

    print(f"{'run':34}{'train':>8}{'held-out':>10}{'n':>8}")
    for run_dir in run_dirs:
        scores = _per_game_scores(run_dir)
        if not scores:
            print(f"{run_dir.name:34}{'(no score data)':>26}")
            continue
        train = [s for g, s in scores.items() if g in train_ids]
        held = [s for g, s in scores.items() if g in eval_ids]
        print(
            f"{run_dir.name:34}{_mean(train):8.2f}{_mean(held):10.2f}"
            f"{f'{len(train)}/{len(held)}':>8}"
        )

    if run_dirs:
        last = _per_game_scores(run_dirs[-1])
        held = {g: s for g, s in last.items() if g in eval_ids}
        if held:
            print("\nheld-out games (baseline = 10-pass base model):")
            for game_id in sorted(held, key=lambda g: -held[g]):
                base = reference.get(game_id)
                delta = f"{held[game_id] - base:+.2f}" if base is not None else "n/a"
                base_text = f"{base:.2f}" if base is not None else "n/a"
                print(f"  {game_id:22}{held[game_id]:7.2f}  base {base_text:>6}  {delta:>7}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=None)
    parser.add_argument("--games-file", type=Path, default=DEFAULT_GAMES_FILE)
    args = parser.parse_args()
    runs = args.run or sorted((REPO_ROOT / "runs").glob("distill-iteration-*"))
    if not runs:
        raise SystemExit("no distillation runs found")
    report(runs, args.games_file)


if __name__ == "__main__":
    main()
