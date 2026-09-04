# context-window-study

Measures how much of the analyzer's context budget was being spent on
over-estimated board images before `ba48881`, so a compensating change to
`LOCAL_ANALYZER_CONTEXT_WINDOW` can be sized from data instead of a rule of
thumb.

- `scripts/extract.py` -- per-trial image counts (from prompt logs) joined to
  per-game image token cost priced both ways (from real boards in events.jsonl)
- `scripts/summarize.py` -- distribution, per-game spread, budget implication
- `data/` -- extracted CSVs, one per analysed run
- `FINDINGS.md` -- results and what they mean for the window

See `world-model-study/` for the same shape applied to prompt variants.
