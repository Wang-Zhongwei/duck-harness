1. fail to understand the game holistically
`hmm, let me reconsider. In level 1:
- LEFT: dark gray moves right
- RIGHT: dark gray moves left
- DOWN: both shapes move down
In level 2:
- LEFT: dark gray moves left (and divider moves left)
- RIGHT: dark gray moves right (and divider moves right)
- UP/DOWN: no effect on shapes
So the movement mechanics are different between levels.`

2. fail to notice all changes


3. fail to use history object: if model can reference past actions given the same game state

first let's compare whether history makes a difference by comparing `ARC3-Inference/runs/20260719_184917/score.json` and `ARC3-Inference/runs/20260719_233640_action-goal-model-api/score.json`

score wise baseline is better. Token wise, similar tokens, expected for same compute resources, step/turn wise baseline has more steps/turns. The prompt structure might leads action-goal-model have more tokens and thus less turns and reduce the score. But here is a nuance. I introduce other changes too. And that change effect can be seen by comparing `20260719_184917` and `20260719_230325_all_games`
no we can't see anything because it's a full run and there are way less tokens per turn. I submitted another run called `ar25-10-pass` this can compare whether the prompt change actually make a gain or loss. 


┌──────────────────────┬────────────────────────────────────────────────────────────┬───────────────┐
│      prediction      │                        measurement                         │    verdict    │
├──────────────────────┼────────────────────────────────────────────────────────────┼───────────────┤
│ (empty) scaffold     │ slot mentions in thinking: 0.13 vs 0.09/block; slots       │               │
│ causes rumination    │ essentially never filled in either run (content_chars      │ refuted       │
│                      │ 2,530 vs 387 total — the cd82 broken-carry pattern)        │               │
├──────────────────────┼────────────────────────────────────────────────────────────┼───────────────┤
│ boundary/holes       │ 'boundary'/'holes' fields in tool results: 0 in both runs; │ refuted       │
│ fatten output        │  24 code refs in 481 turns                                 │               │
├──────────────────────┼────────────────────────────────────────────────────────────┼───────────────┤
│ compactness deletion │ prints/code-block 3.15→4.45; chars/call 592→736            │ supported     │
│  → bigger output     │                                                            │ (output only) │
├──────────────────────┼────────────────────────────────────────────────────────────┼───────────────┤
│ level-up line →      │ levelup-talk 3.4 vs 0.31/block with vs without the line    │ topic yes,    │
│ retrospection        │ (11×) — but block length identical (2,298 vs 2,332)        │ length no     │
└──────────────────────┴────────────────────────────────────────────────────────────┴───────────────┘

4. unwilling to update models and notes