#!/usr/bin/env python3
"""Within-game correlation between headings per game-run and score.

All 4 runs pooled: each game has 4 runs x 10 passes = 40 game-runs.
Correlating across games is meaningless (game difficulty dominates), so the
correlation is computed WITHIN each game and then pooled by Fisher z.
Turn count is a confound in both directions -- a longer trajectory has more
chances to emit a heading and more chances to level up -- so the partial
correlation controlling for turns is reported alongside.
"""
import csv, collections, math, os, statistics as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")


def pearson(x, y):
    return st.correlation(x, y) if len(set(x)) > 1 and len(set(y)) > 1 else float("nan")


def partial(x, y, z):
    """r(x,y) with z partialled out."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    d = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / d if d > 1e-12 else float("nan")


def fisher_pool(rs, ns):
    zs = [0.5 * math.log((1 + r) / (1 - r)) for r in rs if abs(r) < 1]
    ws = [n - 3 for r, n in zip(rs, ns) if abs(r) < 1]
    z = sum(w * v for w, v in zip(ws, zs)) / sum(ws)
    se = 1 / math.sqrt(sum(ws))
    return math.tanh(z), math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


rows = list(csv.DictReader(open(f"{DATA}/games.csv")))
by_game = collections.defaultdict(list)
for r in rows:
    by_game[r["game"]].append((int(r["turns_any_field"]), float(r["score"] or 0), int(r["turns"])))

print("all 4 runs pooled; 40 game-runs per game\n")
print(f"{'game':16s} {'n':>3s} {'headings/gr':>12s} {'score':>8s} "
      f"{'r(head,score)':>14s} {'r | turns':>10s} {'r(turns,score)':>15s}")
out, rs, ns, prs = [], [], [], []
for g in sorted(by_game):
    v = by_game[g]
    h = [a for a, _, _ in v]; s = [b for _, b, _ in v]; t = [c for _, _, c in v]
    r = pearson(h, s); rp = partial(h, s, t); rts = pearson(t, s)
    print(f"{g:16s} {len(v):3d} {st.mean(h):12.2f} {st.mean(s):8.2f} "
          f"{r:14.3f} {rp:10.3f} {rts:15.3f}")
    out.append({"game": g, "game_runs": len(v),
                "headings_per_game_run": round(st.mean(h), 2),
                "score": round(st.mean(s), 2),
                "r_headings_score": round(r, 3),
                "r_partial_given_turns": round(rp, 3),
                "r_turns_score": round(rts, 3)})
    if not math.isnan(r):
        rs.append(r); ns.append(len(v))
    if not math.isnan(rp):
        prs.append(rp)

pooled, lo, hi = fisher_pool(rs, ns)
print(f"\npooled within-game r (Fisher z, {len(rs)} games) = {pooled:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
print(f"median per-game r = {st.median(rs):+.3f}   games with r>0: {sum(1 for r in rs if r>0)}/{len(rs)}")
pp, plo, phi = fisher_pool(prs, ns[:len(prs)])
print(f"pooled partial r controlling for turns = {pp:+.3f}  95% CI [{plo:+.3f}, {phi:+.3f}]")
rts_all = [o["r_turns_score"] for o in out]
print(f"pooled r(turns, score) = {fisher_pool(rts_all, ns)[0]:+.3f}   <- the confound")

with open(f"{DATA}/correlation_by_game.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
print(f"\nwrote data/correlation_by_game.csv ({len(out)} rows)")
