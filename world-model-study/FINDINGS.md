# World-model heading frequency (4 local runs, 2026-08-30 → 08-31)

## What was counted

The system prompt's "Per-turn protocol" block says *"restate your working world
model"* and offers optional prefixes `World model:`, `Goal model:`,
`Action model:`, `Recent findings:`, `Open questions:`, `Plan:`,
`Cross-level notes:`. This counts how often the solver actually emits them.

`scripts/extract.py` → `data/games.csv` (one row per game-run), `data/runs.csv`,
`data/turns.csv` (per-turn detail, only for the within-level breakdown).
`scripts/analyze.py` → `data/by_game.csv`, `data/field_counts.csv`,
`data/by_level_position.csv`. `scripts/prompt_diff.py` → `data/prompts/`.

Two traps the parser handles, both verified against the artifacts:

- **Grep trap.** Every transcript embeds the system prompt, which lists all seven
  strings verbatim. Grepping a transcript reports ~100%. Only `[ASSISTANT]`
  sections are scanned (`[SYSTEM PROMPT]`, `[USER PROMPT]`, `[THINKING]` excluded).
- **Turn trap.** `*_events.jsonl` can hold several `type=analysis` events sharing
  one `analysis_step` — successive LLM rounds in one observe-plan-act cycle.
  `eval.py:_turn_count` defines a turn as the max `analysis_step`. Counting
  events instead inflates the denominator by ~31%. extract.py raises if its turn
  count disagrees with `evaluation.json usage.per_run.turns` for any game-run;
  all 1000 game-runs match.

**Scores are read verbatim from `evaluation.json`,** never recomputed and never
taken from `passes/<n>/summary.txt` (each pass dir is one GPU's half of the run,
so its `mean score` is over 5 of 10 passes and is not the run score). extract.py
re-derives the run score with eval.py's own method — per-game score = mean of
`trial_scores` over 10 passes, run score = unweighted mean of the 25 per-game
means — and asserts it equals `evaluation.json["score"]`.

## Result

| run | commit | score | turns | labeled | % of turns | per game-run |
|---|---|---|---|---|---|---|
| 0830_104347 prompt-static-to-system | b7f05fe | 6.492 | 9642 | 415 | 4.30% | 1.66 |
| 0830_110533 game-priors | 4c1d5ba | 5.837 | 8994 | 336 | 3.74% | 1.34 |
| 0831_083143 remove_model_headings | 4c1d5ba | 6.773 | 8493 | 432 | 5.09% | 1.73 |
| 0831_094025 improve-game-prior-prompts | 5971285 | 6.427 | 9468 | 415 | 4.38% | 1.66 |

Per field, pooled over 36,597 turns:

| field | turns | % |
|---|---|---|
| `Plan:` | 1041 | 2.84% |
| `World model:` | 523 | 1.43% |
| `Open questions:` | 138 | 0.38% |
| `Goal model:` | 134 | 0.37% |
| `Recent findings:` | 89 | 0.24% |
| `Action model:` | 54 | 0.15% |
| `Cross-level notes:` | 7 | 0.02% |
| **any** | **1598** | **4.37%** |

A game-run averages ~37 turns, so a whole trajectory carries **~1.7 labeled
headings**. `Action model:` fires once per ~680 turns; `Cross-level notes:`
7 times in 36,597 turns.

## Three things that change the reading

**1. The headings fire at re-grounding moments, not uniformly.**

| position in level | turns | labeled |
|---|---|---|
| first turn of a level | 2453 | **20.14%** |
| turns 2–5 | 8846 | 4.94% |
| turn 6+ | 25298 | 2.64% |

**2. The instruction is obeyed; only the formatting is skipped.** 86–92% of turns
carry assistant prose (median ~300 chars, held in a 20-message history so it does
carry forward), and 32–35% contain explicit model language in unlabeled prose.
Example (`ar25` p0): *"Model: the black piece is controlled by arrow keys (3 cells
per press); the gray is its live mirror; the yellow Γ … is the goal position"* —
a full restatement that scores 0 on the heading count.

**3. Two of these four runs are a replicate pair, and that sets the noise floor.**
`20260830_110533_game-priors` and `20260831_083143_remove_model_headings_and_game_priors`
ran the **same commit (4c1d5ba) and a byte-identical system prompt**
(`prompt_sha 179de17880f4`) — the heading removal the second name advertises never
landed in the run. Same prompt, different outcome:

- Δ score **+0.937** (5.837 → 6.773)
- Δ labeled rate **+1.35pp** (3.74% → 5.09%)

That gap is larger than every prompt-to-prompt gap in the table. None of the four
runs is distinguishable from another on this evidence.

## Bearing on "should the scaffolding be removed"

There is **no A/B**: `prompt_diff.py` confirms all four prompts still carry
`restate your working world model` and the full seven-label list. The score
spread is attributable to the game-priors edits plus replicate noise, not to this
block. Its cost is ~103 tokens inside a ~3.9k-token system prefix that is
prefix-cached every turn, so removal saves essentially nothing.

### Within-game correlation: none

Pooling all four runs gives 40 game-runs per game (4 runs × 10 passes).
Correlating *across* games is meaningless — game difficulty dominates both axes —
so `scripts/correlate.py` computes the correlation **within** each game:

- pooled within-game **r = +0.029, 95% CI [−0.036, +0.093]** (Fisher z, 25 games)
- median per-game r = +0.038; 14 of 25 games positive, 11 negative
- controlling for turn count: partial **r = +0.018** [−0.047, +0.082]
- the suspected confound is absent too: within-game r(turns, score) = −0.005

No single game reaches significance either. The extremes are sc25 (+0.38) and
re86 (−0.31), both with 95% CIs that straddle 0 at n=40 (`figs/correlation_by_game.png`).

The earlier bucketed gradient — 0 headings → score 5.78, 3+ → 7.06 — is a
between-game artifact: games that draw many headings are not the same games that
score high. It disappears once the comparison is made within a game.

Figures: `figs/headings_vs_score_by_game.png` (25 panels, 40 points each),
`figs/correlation_by_game.png` (per-game r with CIs).

Defensible trim: drop `Action model:` and `Cross-level notes:` from the label
list (rates indistinguishable from zero), keep the `restate your working world
model` bullet, which is demonstrably obeyed in prose. Removing the restatement
bullet itself would remove the only cross-turn memory this branch has and needs
its own A/B — and given a ±0.94 replicate gap, that A/B needs more than one run
per arm.
