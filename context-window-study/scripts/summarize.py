"""Aggregate the per-trial extract into the numbers a window change needs.

Reports the images-per-request distribution overall and per game, because the
retained-step count is not a constant: games whose steps produce long tool
output evict history sooner and hold fewer boards than games with terse turns.
A single mean hides a spread that directly scales the wasted-token estimate.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--top", type=int, default=8,
                        help="games to show at each end of the spread")
    args = parser.parse_args()

    with open(args.csv_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]

    images = [int(r["images_in_request"]) for r in rows]
    wasted = [float(r["wasted_tokens_in_request"]) for r in rows if r["wasted_tokens_in_request"]]
    per_img = [float(r["wasted_tokens_per_image"]) for r in rows if r["wasted_tokens_per_image"]]

    print(f"trials: {len(rows)}   games: {len({r['game'] for r in rows})}")
    print()
    print("images per request")
    print(f"  mean   {statistics.mean(images):6.2f}")
    print(f"  median {statistics.median(images):6.1f}")
    print(f"  stdev  {statistics.stdev(images):6.2f}")
    print(f"  range  {min(images)} - {max(images)}")
    hist = defaultdict(int)
    for n in images:
        hist[n] += 1
    print("  histogram:")
    for n in sorted(hist):
        print(f"    {n:2d} images  {hist[n]:3d}  {'#' * hist[n]}")

    print()
    print("wasted tokens per image (naive estimate - exact)")
    print(f"  mean {statistics.mean(per_img):7.1f}   range {min(per_img):.0f} - {max(per_img):.0f}")
    print()
    print("wasted budget per request (images x per-image waste)")
    print(f"  mean   {statistics.mean(wasted):7.0f}")
    print(f"  median {statistics.median(wasted):7.0f}")
    print(f"  range  {min(wasted):.0f} - {max(wasted):.0f}")

    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(int(r["images_in_request"]))
    means = sorted(((statistics.mean(v), g, len(v)) for g, v in by_game.items()))
    print()
    print(f"per-game images/request -- lowest {args.top}")
    for m, g, n in means[:args.top]:
        print(f"  {g:20s} {m:5.2f}  (n={n})")
    print(f"per-game images/request -- highest {args.top}")
    for m, g, n in means[-args.top:]:
        print(f"  {g:20s} {m:5.2f}  (n={n})")

    spread = means[-1][0] - means[0][0]
    print()
    print(f"between-game spread: {spread:.2f} images "
          f"({spread * statistics.mean(per_img):.0f} tokens of budget)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
