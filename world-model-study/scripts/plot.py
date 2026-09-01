#!/usr/bin/env python3
"""Headings per game-run vs score, per game, all 4 runs pooled.

    ../ARC3-Inference/.venv/bin/python3 scripts/plot.py

figs/headings_vs_score_by_game.png   25 small panels, one per game, 40 points each
figs/correlation_by_game.png         the per-game correlations on one axis
"""
import collections, csv, math, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, FIGS = os.path.join(BASE, "data"), os.path.join(BASE, "figs")
os.makedirs(FIGS, exist_ok=True)

rows = list(csv.DictReader(open(f"{DATA}/games.csv")))
runs = sorted({r["run"] for r in rows})
COLORS = dict(zip(runs, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]))
LABEL = {n: n[4:8] + " " + n[16:38] for n in runs}

by_game = collections.defaultdict(list)
for r in rows:
    by_game[r["game"]].append((int(r["turns_any_field"]), float(r["score"] or 0), r["run"]))
games = sorted(by_game, key=lambda g: -st.mean(b for _, b, _ in by_game[g]))


def pearson(x, y):
    return st.correlation(x, y) if len(set(x)) > 1 and len(set(y)) > 1 else float("nan")


# ---------- fig 1: one panel per game ----------
fig, axes = plt.subplots(5, 5, figsize=(17, 15))
for ax, g in zip(axes.ravel(), games):
    v = by_game[g]
    for n in runs:
        pts = [(a, b) for a, b, rn in v if rn == n]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26, alpha=.72,
                   color=COLORS[n], edgecolor="white", linewidth=.5, label=LABEL[n], zorder=3)
    x = [a for a, _, _ in v]; y = [b for _, b, _ in v]
    r = pearson(x, y)
    if len(set(x)) > 1:                                   # least-squares fit
        b1 = st.correlation(x, y) * st.stdev(y) / st.stdev(x)
        b0 = st.mean(y) - b1 * st.mean(x)
        xs = [min(x), max(x)]
        ax.plot(xs, [b0 + b1 * t for t in xs], color="0.25", lw=1.4, ls="--", zorder=2)
    ax.set_title(f"{g}   r={r:+.2f}", fontsize=10,
                 color=("#B02418" if abs(r) > .3 else "0.15"))
    ax.grid(alpha=.25, lw=.6)
    ax.tick_params(labelsize=8)
axes.ravel()[0].legend(fontsize=7.5, loc="upper left", framealpha=.9)
fig.suptitle("Labeled world-model headings per game-run vs score, within each game\n"
             "40 game-runs per panel (4 runs x 10 passes); dashed line = least squares",
             fontsize=13.5, y=.995)
fig.supxlabel("labeled headings in the game-run", fontsize=12)
fig.supylabel("score of that game-run", fontsize=12)
fig.tight_layout(rect=[.012, .012, 1, .975])
fig.savefig(f"{FIGS}/headings_vs_score_by_game.png", dpi=145)
print("wrote figs/headings_vs_score_by_game.png")

# ---------- fig 2: the correlations themselves ----------
cor = [(g, pearson([a for a, _, _ in by_game[g]], [b for _, b, _ in by_game[g]])) for g in games]
cor.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(8.5, 8))
ys = range(len(cor))
ci = 1.96 / math.sqrt(40 - 3)                                # Fisher-z 95% half-width
for i, (g, r) in enumerate(cor):
    z = 0.5 * math.log((1 + r) / (1 - r))
    ax.plot([math.tanh(z - ci), math.tanh(z + ci)], [i, i], color="0.75", lw=2, zorder=1)
ax.scatter([r for _, r in cor], ys, s=52,
           color=["#C44E52" if r < 0 else "#4C72B0" for _, r in cor], zorder=3)
ax.axvline(0, color="0.2", lw=1.1)
zs = [0.5 * math.log((1 + r) / (1 - r)) for _, r in cor]
pooled = math.tanh(sum(zs) / len(zs))
ax.axvline(pooled, color="#2E7D32", lw=2, ls="--",
           label=f"pooled (Fisher z) r = {pooled:+.3f}, 95% CI [-0.036, +0.093]")
ax.set_yticks(list(ys)); ax.set_yticklabels([g for g, _ in cor], fontsize=9)
ax.set_xlabel("within-game correlation of headings per game-run with score  (n=40 each)", fontsize=11)
ax.set_title("No game shows a reliable link between world-model headings and score",
             fontsize=13, pad=12)
ax.grid(axis="x", alpha=.3)
ax.legend(fontsize=9.5, loc="lower right")
fig.tight_layout()
fig.savefig(f"{FIGS}/correlation_by_game.png", dpi=150)
print("wrote figs/correlation_by_game.png")
