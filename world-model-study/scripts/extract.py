#!/usr/bin/env python3
"""
Count labeled world-model headings emitted by the solver, per GAME-RUN.

Outputs (data/):
  games.csv   source data: one row per game-run (run x game x pass): score,
              levels, turns, and how many turns carried each labeled heading.
  turns.csv   source data: per-turn detail, feeds the within-level breakdown.

analyze.py turns these into the two files meant for reading, runs.csv and
by_game.csv.

Two parsing facts, verified against the artifacts on 2026-09-01:

1. TURN vs ANALYSIS EVENT.  `*_events.jsonl` can hold several `type=analysis`
   events sharing one `analysis_step`; those are successive LLM rounds inside
   one observe-plan-act cycle.  `inference/tools/eval.py:_turn_count` defines a
   turn as the MAX `analysis_step`, so a turn is the unique `analysis_step`.
   Counting events instead inflates the denominator by ~31%.

2. WHERE HEADINGS ARE COUNTED.  Every transcript embeds the system prompt,
   which itself lists `World model:`, `Action model:`, ... verbatim, so a naive
   grep reports ~100%.  Only `[ASSISTANT]` sections are scanned.

SCORES ARE NOT RECOMPUTED.  Game-run scores are read verbatim from
evaluation.json `games[g].trial_scores[<run>/pass-N]`.  Per eval.py, a game's
score is the mean of its trial_scores and a run's score is the unweighted mean
of the per-game means; both are re-derived here and asserted against the
evaluation.json values.  (Do NOT use `mean score` from passes/<n>/summary.txt:
each pass dir is one GPU's half of the run.)
"""
import argparse, collections, csv, glob, hashlib, json, os, re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

FIELDS = ["World model", "Goal model", "Action model",
          "Recent findings", "Open questions", "Plan", "Cross-level notes"]
MARK = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|ASSISTANT"
                  r"|TOOL CALL:[^\]]*|TOOL RESULT:[^\]]*|ANALYZER STATUS)\]\s*$", re.M)
FPAT = {f: re.compile(r"^[ \t]*(?:[-*]\s*)?(?:\*\*|##\s*)?" + re.escape(f) + r"\**[ \t]*:", re.M | re.I)
        for f in FIELDS}
LOOSE = re.compile(r"world model|my model|the model is|model:|goal (?:is|seems|appears)"
                   r"|hypothes|mechanic|confirm(?:ed|s)?\b|theory|i control|controls? the", re.I)


def assistant_text(transcript):
    marks = list(MARK.finditer(transcript))
    return "\n".join(transcript[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(transcript))]
                     for i, m in enumerate(marks) if m.group(1) == "ASSISTANT")


def scan_game_run(events_path):
    """Return (per-turn rows, aggregated game-run row) for one events file."""
    turns = collections.OrderedDict()
    with open(events_path) as fh:
        for line in fh:
            if '"type":"analysis"' not in line and '"type": "analysis"' not in line:
                continue
            ev = json.loads(line)
            if ev.get("type") != "analysis":
                continue
            step = int(ev["analysis_step"])
            rec = turns.setdefault(step, {"turn": step, "level": int(ev.get("level") or 0),
                                          "rounds": 0, "asst_chars": 0, "loose": 0,
                                          **{f: 0 for f in FIELDS}})
            text = assistant_text(ev.get("transcript") or "")
            rec["rounds"] += 1
            rec["asst_chars"] += len(text.strip())
            rec["loose"] |= bool(LOOSE.search(text))
            for f, pat in FPAT.items():
                rec[f] += len(pat.findall(text))
    rows, prev, since = [], None, 0
    for step in sorted(turns):
        r = turns[step]
        since = 0 if r["level"] != prev else since + 1
        prev = r["level"]
        r["turns_since_level_start"] = since
        r["any_field"] = int(any(r[f] for f in FIELDS))
        rows.append(r)
    agg = {"turns": len(rows), "rounds": sum(r["rounds"] for r in rows),
           "max_level_seen": max((r["level"] for r in rows), default=0),
           "turns_any_field": sum(r["any_field"] for r in rows),
           "turns_model_talk": sum(int(r["loose"]) for r in rows),
           "turns_with_prose": sum(1 for r in rows if r["asst_chars"] > 40),
           **{f: sum(1 for r in rows if r[f]) for f in FIELDS}}
    return rows, agg


def load_eval(run_dir):
    path = os.path.join(run_dir, "evaluation.json")
    return json.load(open(path)) if os.path.exists(path) else None


def system_prompt_sha(run_dir):
    for ev in sorted(glob.glob(os.path.join(run_dir, "passes", "*", "artifacts", "*_events.jsonl")))[:1]:
        for line in open(ev):
            if '"type":"analysis"' not in line and '"type": "analysis"' not in line:
                continue
            t = json.loads(line)["transcript"]
            sp = t[t.find("[SYSTEM PROMPT]") + 15:t.find("[USER PROMPT]")].strip()
            return hashlib.sha1(sp.encode()).hexdigest()[:12]
    return ""


def git_commit(run_dir):
    p = os.path.join(run_dir, "git_info.txt")
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("commit:"):
                return line.split(":", 1)[1].strip()[:12]
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    game_cols = (["run", "commit", "prompt_sha", "game", "pass", "pass_dir", "trial",
                  "score", "levels_completed", "total_levels", "state", "turns", "actions",
                  "rounds", "max_level_seen", "turns_any_field", "pct_turns_any_field",
                  "turns_model_talk", "turns_with_prose"] + FIELDS)
    turn_cols = ["run", "game", "pass", "turn", "level", "turns_since_level_start",
                 "rounds", "asst_chars", "loose", "any_field"] + FIELDS
    gfh = open(os.path.join(DATA, "games.csv"), "w", newline="")
    tfh = open(os.path.join(DATA, "turns.csv"), "w", newline="")
    gw, tw = csv.DictWriter(gfh, game_cols), csv.DictWriter(tfh, turn_cols)
    for w in (gw, tw):
        w.writeheader()

    for run_dir in args.run_dirs:
        run = os.path.basename(run_dir.rstrip("/"))
        ev = load_eval(run_dir)
        commit, sha = git_commit(run_dir), system_prompt_sha(run_dir)
        per_game_scores, rows_out = collections.defaultdict(list), []
        tot = collections.Counter()
        for events in sorted(glob.glob(os.path.join(run_dir, "passes", "*", "artifacts", "*_events.jsonl"))):
            pass_dir = int(events.split(os.sep + "passes" + os.sep)[1].split(os.sep)[0])
            stem = os.path.basename(events).replace("_events.jsonl", "")
            game, _, trial = stem.rpartition("_p")
            trial = int(trial)
            # passes/<d> holds one GPU's five trials; global pass index = dir + trial
            gpass = pass_dir + trial
            key = f"{run}/pass-{gpass}"
            turns, agg = scan_game_run(events)
            g = (ev or {}).get("games", {}).get(game, {})
            # the join is self-checking: eval's turn count must match ours
            ev_turns = (g.get("turns") or {}).get(key)
            if ev_turns is not None and ev_turns != agg["turns"]:
                raise SystemExit(f"turn mismatch {key} {game}: ours={agg['turns']} eval={ev_turns} "
                                 f"-- the pass_dir+trial -> global pass mapping is wrong")
            score = (g.get("trial_scores") or {}).get(key)
            if score is not None:
                per_game_scores[game].append(score)
            rec = {"run": run, "commit": commit, "prompt_sha": sha, "game": game,
                   "pass": gpass, "pass_dir": pass_dir, "trial": trial,
                   "score": score if score is not None else "",
                   "levels_completed": (g.get("levels_completed") or {}).get(key, ""),
                   "total_levels": g.get("total_levels", ""),
                   "state": (g.get("states") or {}).get(key, ""),
                   "actions": (g.get("actions") or {}).get(key, ""),
                   "pct_turns_any_field": round(100 * agg["turns_any_field"] / agg["turns"], 3) if agg["turns"] else "",
                   **{k: v for k, v in agg.items()}}
            rows_out.append(rec)
            tot["turns"] += agg["turns"]; tot["any"] += agg["turns_any_field"]
            tot["actions"] += rec["actions"] if isinstance(rec["actions"], int) else 0
            for r in turns:
                tw.writerow({"run": run, "game": game, "pass": gpass,
                             **{c: r[c] for c in turn_cols[3:]}})
        gw.writerows(rows_out)

        # reproduce evaluation.json's score with eval.py's own method
        game_means = {g: sum(v) / len(v) for g, v in per_game_scores.items() if v}
        check = sum(game_means.values()) / len(game_means) if game_means else float("nan")
        if ev is not None and abs(check - ev["score"]) > 1e-9:
            raise SystemExit(f"{run}: score check {check} != evaluation.json {ev['score']}")
        print(f"  {run:58s} {len(rows_out):4d} game-runs  score={ev['score']:.3f} (check OK)")
    for fh in (gfh, tfh):
        fh.close()
    print(f"wrote {DATA}/games.csv, {DATA}/turns.csv  (run analyze.py for runs.csv + by_game.csv)")


if __name__ == "__main__":
    main()
