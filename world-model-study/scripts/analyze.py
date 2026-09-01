#!/usr/bin/env python3
"""Report world-model heading frequency, headline unit = per GAME-RUN.

Reads  data/games.csv, data/turns.csv     (source data, written by extract.py)
Writes data/runs.csv     <- READ THIS. 4 rows x 5 cols.
       data/by_game.csv  <- READ THIS. 100 rows x 4 cols.

Everything else (per-field rates, position-within-level, distributions) prints
to the console; it is derivable from data/games.csv and not worth a file.
"""
import collections, csv, os, statistics as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIELDS = ["World model", "Goal model", "Action model",
          "Recent findings", "Open questions", "Plan", "Cross-level notes"]
INT = ["turns", "rounds", "turns_any_field", "turns_model_talk", "turns_with_prose"] + FIELDS
POS = [("first_turn_of_level", lambda s: s == 0), ("turn_2_5", lambda s: 1 <= s < 5),
       ("turn_6plus", lambda s: s >= 5)]

games = list(csv.DictReader(open(f"{DATA}/games.csv")))
for r in games:
    for k in INT:
        r[k] = int(r[k])
    r["score"] = float(r["score"]) if r["score"] else 0.0
    r["levels_completed"] = float(r["levels_completed"] or 0)
runs = sorted({r["run"] for r in games})
short = {n: n[4:8] + "-" + n[9:15] for n in runs}

# position-within-level, the one thing that needs turn granularity
pos = collections.defaultdict(lambda: [0, 0])
for r in csv.DictReader(open(f"{DATA}/turns.csv")):
    b = next(name for name, p in POS if p(int(r["turns_since_level_start"])))
    pos[(r["run"], b)][0] += 1
    pos[(r["run"], b)][1] += int(r["any_field"])


def sel(run=None, game=None):
    return [r for r in games if (run is None or r["run"] == run) and (game is None or r["game"] == game)]


def block(rs):
    """The heading stats for a set of game-runs."""
    v = sorted(r["turns_any_field"] for r in rs)
    t = sum(r["turns"] for r in rs)
    return {"game_runs": len(rs), "turns": t,
            "headings_per_game_run": round(st.mean(v), 3),
            "median": st.median(v), "p90": v[int(0.9 * (len(v) - 1))], "max": max(v),
            "pct_game_runs_zero": round(100 * sum(1 for x in v if x == 0) / len(v), 2),
            "pct_game_runs_3plus": round(100 * sum(1 for x in v if x >= 3) / len(v), 2),
            "turns_any_field": sum(v), "pct_turns_any_field": round(100 * sum(v) / t, 3),
            "mean_turns_per_game_run": round(t / len(rs), 2)}


# ---------------- runs.csv : 5 columns ----------------
# prompt_sha is kept because it is the only way to spot replicate runs, which
# set the noise floor for every comparison in this table.
run_rows = []
for n in runs:
    rs = sel(run=n)
    b = block(rs)
    run_rows.append({"run": n, "prompt_sha": rs[0]["prompt_sha"],
                     "score": round(st.mean(st.mean(r["score"] for r in sel(n, g))
                                            for g in sorted({r["game"] for r in rs})), 3),
                     "headings_per_game_run": b["headings_per_game_run"],
                     "pct_game_runs_zero": b["pct_game_runs_zero"]})
with open(f"{DATA}/runs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, list(run_rows[0])); w.writeheader(); w.writerows(run_rows)

# ---------------- by_game.csv : 4 columns ----------------
game_rows = []
for n in runs:
    for g in sorted({r["game"] for r in sel(run=n)}):
        rs = sel(n, g)
        game_rows.append({"run": n, "game": g,
                          "score": round(st.mean(r["score"] for r in rs), 2),
                          "headings_per_game_run": block(rs)["headings_per_game_run"]})
with open(f"{DATA}/by_game.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, list(game_rows[0])); w.writeheader(); w.writerows(game_rows)

# ---------------- console report ----------------
print("=== per run (score read from evaluation.json, asserted in extract.py) ===")
print(f"{'run':50s} {'score':>6s} {'per game-run':>13s} {'%0':>6s} {'%3+':>6s} {'turns/gr':>9s}")
for r, n in zip(run_rows, runs):
    b = block(sel(run=n))
    print(f"{r['run'][:50]:50s} {r['score']:6.3f} {r['headings_per_game_run']:13.2f} "
          f"{r['pct_game_runs_zero']:5.1f}% {b['pct_game_runs_3plus']:5.1f}% "
          f"{b['mean_turns_per_game_run']:9.1f}")
p = block(games)
print(f"{'POOLED':50s} {'':>6s} {p['headings_per_game_run']:13.2f} "
      f"{p['pct_game_runs_zero']:5.1f}% {p['pct_game_runs_3plus']:5.1f}% "
      f"{p['mean_turns_per_game_run']:9.1f}")

print("\n=== distribution of headings per game-run (1000 game-runs) ===")
hist = collections.Counter(r["turns_any_field"] for r in games)
cum = 0
for k in sorted(hist):
    if k > 6:
        break
    cum += hist[k]
    print(f"  {k} headings {hist[k]:4d}  {100*hist[k]/len(games):5.1f}%  cum {100*cum/len(games):5.1f}%")
print(f"  7+         {len(games)-cum:4d}  {100*(len(games)-cum)/len(games):5.1f}%  cum 100.0%")

print("\n=== per field: % of turns (pooled) ===")
t = sum(r["turns"] for r in games)
for f in sorted(FIELDS, key=lambda f: -sum(r[f] for r in games)):
    c = sum(r[f] for r in games)
    print(f"  {f:20s} {c:5d}  {100*c/t:5.2f}%")

print("\n=== position within level (pooled) ===")
for b, _ in POS:
    tt = sum(pos[(n, b)][0] for n in runs); aa = sum(pos[(n, b)][1] for n in runs)
    print(f"  {b:22s} turns={tt:6d}  labeled={aa:5d}  ({100*aa/tt:5.2f}%)")

print("\n=== headings per game-run vs score ===")
for name, pred in [("0", lambda v: v == 0), ("1-2", lambda v: 1 <= v <= 2), ("3+", lambda v: v >= 3)]:
    s2 = [r for r in games if pred(r["turns_any_field"])]
    print(f"  {name:4s} game-runs={len(s2):5d}  score={st.mean(r['score'] for r in s2):6.2f}"
          f"  levels={st.mean(r['levels_completed'] for r in s2):5.2f}"
          f"  turns={st.mean(r['turns'] for r in s2):6.1f}")
x = [r["headings_per_game_run"] for r in game_rows]; y = [r["score"] for r in game_rows]
print(f"  across {len(game_rows)} run x game cells: pearson r(headings/game-run, score) = {st.correlation(x,y):+.3f}")

print("\n=== replicate runs (identical prompt_sha) ===")
by_sha = collections.defaultdict(list)
for r in run_rows:
    by_sha[r["prompt_sha"]].append(r)
for sha, grp in by_sha.items():
    if len(grp) < 2:
        continue
    print(f"  prompt_sha {sha}")
    for r in grp:
        print(f"    {r['run'][:52]:52s} score={r['score']:6.3f}  per game-run={r['headings_per_game_run']:5.2f}")
    print(f"    -> same prompt: delta score {grp[1]['score']-grp[0]['score']:+.3f}, "
          f"delta per game-run {grp[1]['headings_per_game_run']-grp[0]['headings_per_game_run']:+.2f}")

print("\nwrote data/runs.csv (4 rows) and data/by_game.csv (100 rows)")
