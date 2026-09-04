# How many board images ride in a request, and what the image-token fix freed

Run analysed: `runs/local/20260901_092257_even-simpler-user-prompt-and-game-prior-dial-back`
(commit `f38fadf` = `ba48881^`, the last commit before the exact image estimate;
`LOCAL_ANALYZER_CONTEXT_WINDOW=32768`, `MULTIMODAL_UPSCALE=4`, 25 games x 5 trials).

## Headline

| quantity | mean | median | range |
|---|---|---|---|
| board images per request | **6.85** | 6 | 3 - 18 |
| wasted tokens per image (naive - exact) | **459** | - | 280 - 722 |
| wasted budget per request | **3,147** | 2,866 | 1,176 - 7,515 |

The working assumption of "~5 steps, so 5 images" undercounts by ~37%. History
messages are never stripped of their images -- `_persistent_history_messages`
keeps prior user messages whole, and the fork's `_strip_images_from_message`
died with `d448d40` -- so a request carries one image per retained step, and
the trimmer retains more steps than the rule of thumb suggests.

## The count is not a constant across games

Per-game means run from **5.00** (`s5i5`, `tu93`) to **10.20** (`sc25`), a
spread of 5.2 images = ~2,400 tokens of budget. Games with terse turns keep
twice the visual history of games whose tool output is verbose enough to evict
it, because the trimmer evicts on estimated tokens and knows nothing about
steps.

Lowest: `s5i5` 5.00, `tu93` 5.00, `cn04` 5.20, `g50t` 5.60, `sk48` 5.60
Highest: `sc25` 10.20, `tn36` 10.00, `m0r0` 8.00, `wa30` 7.80, `sp80` 7.80

The per-image waste varies too (280 - 722 tokens), because the naive estimate
priced the base64 payload and PNG compressibility tracks board complexity --
busy boards were charged more than sparse ones for identical real cost.

## Implication for the context window

`ba48881` handed the trimmer back a mean of ~3,150 tokens per request, which it
now spends on real conversation history. To restore the pre-fix prompt length:

    32,768 - 3,147 ~= 29,600

A window of 30,000 shrinks by 2,768, recovering ~88% of the mean correction --
close enough, and it is a rounder number to reason about. There is no
power-of-two constraint: the value is consumed once, as
`max(1024, WINDOW - reply_reserve - safety_margin)` at `tool_agent.py:945`.

Note the knob is denominated in *estimated* tokens while text is deliberately
over-counted at 3 chars/token against a real ~4, so a 3,150-token cut removes
roughly 2,360 *real* prompt tokens.

**No single window restores parity for every game.** A flat cut over-corrects
`s5i5`/`tu93` (which wasted ~2,300) and under-corrects `sc25` (~4,700). The
pre-fix behaviour charged more budget where more images were present, i.e. it
implicitly kept less history in image-heavy games; a flat window replaces that
coupling with a uniform one. That is arguably the better shape, but it means
the change is not score-neutral per game even if it is neutral on average.

## Caveat on the sample

A prompt log is a "LATEST MODEL CALL SNAPSHOT" -- `_write_prompt_log` opens it
with mode `"w"` on every model call, so only the final request of each
game-run survives. All 125 rows are therefore late-game observations, taken
when context is fullest. Treat the numbers as a near-steady-state ceiling
rather than an average over every turn of the run.

## Reproduce

    cd ARC3-Inference
    uv run --no-sync python ../context-window-study/scripts/extract.py \
      runs/local/<run> --upscale 4 \
      --out ../context-window-study/data/<run>.csv
    uv run --no-sync python ../context-window-study/scripts/summarize.py \
      ../context-window-study/data/<run>.csv
