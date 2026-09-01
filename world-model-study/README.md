# world-model-study

How often the solver actually emits the labeled world-model headings that the
system prompt offers (`World model:`, `Goal model:`, `Action model:`, …).

    python3 scripts/extract.py ../ARC3-Inference/runs/local/<run> [<run> ...]
    python3 scripts/analyze.py
    python3 scripts/correlate.py     # within-game r(headings, score), all runs pooled
    python3 scripts/prompt_diff.py ../ARC3-Inference/runs/local/<run> [<run> ...]

    # plotting needs matplotlib, which lives in the project venv
    ../ARC3-Inference/.venv/bin/python3 scripts/plot.py

`data/` — **read these two:**

| file | rows × cols | columns |
|---|---|---|
| `runs.csv` | 4 × 5 | `run, prompt_sha, score, headings_per_game_run, pct_game_runs_zero` |
| `by_game.csv` | 100 × 4 | `run, game, score, headings_per_game_run` |

`prompt_sha` is there for one reason: identical shas mean the runs shipped the
same prompt, and their gap is the noise floor for every other comparison.

Everything else — per-field rates, position within level, distributions — prints
to the console. Source data, regenerate rather than read: `games.csv` (1000
game-runs, everything aggregates from it), `turns.csv` (per-turn, feeds only the
position breakdown), `prompts/` (the system prompt each run actually sent).

`figs/` — `headings_vs_score_by_game.png` (25 panels, 40 game-runs each),
`correlation_by_game.png` (per-game r with 95% CIs).

See `FINDINGS.md`.
