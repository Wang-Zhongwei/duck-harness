# Dual-model training: world_model + planner

Status: draft v2 — updated with smoke-test results (2026-08-02, `dual_model/smoke/`).

## 1. Motivation

RL-finetuning one policy on per-game score is the wrong objective here: every
game is different, and the Kaggle hidden set (110 games) shares no specifics
with our 25 offline games. What must be learned is the **adaptation
procedure** — memorize important details, update beliefs from the latest
observation, plan under uncertainty — i.e. the meta-RL regime: weights encode
*how to learn a game*; the per-game learning happens in context.

Two measured failure modes of the current single-model agent motivate the
split (see memory/analysis notes):

- belief upkeep is an *optional tool call* the 27B makes on only ~4–6% of
  turns, and level-transition wipes clear model fields;
- planning failures are hypothesis-bookkeeping failures (no hypothesis kill,
  64-action timer misread as progress bar, repeated no-op actions).

Compute reality: a 27B rollout is ~1 h/game (H100, ~49 tok/s batch=1). A
small model (gemma4-class) decodes >500 tok/s with 1.4k tok/s prefill →
~8–10 min/episode, and leaves headroom for 8–16 concurrent episodes per H100.
Small-model score signal on the 25 games is tiny but nonzero → trainable with
dense rewards + curriculum, not with score-only RL.

## 2. Architecture

Two small models, "two parts of the brain", per-turn protocol:

1. `world_model(B_{t-1}, a_{t-1}, o_t, ô_t)` → mismatch analysis → revised
   belief `B_t` → (after the planner commits `a_t`) prediction `ô_{t+1}`.
2. `planner(B_t, o_t, score/level, budget)` → hypothesis being tested +
   expected outcome + `a_t`.

Design points:

- Belief update is an **unconditional pipeline stage**, not a tool call — the
  4–6%-of-turns problem is fixed structurally, not by prompting.
- The planner sees the belief + the *current* frame only; episode history
  lives compressed inside `B`. That bottleneck is what forces `B` to be a
  real memory.
- `B` schema (interface contract, to be frozen before any training):
  entities/objects; dynamics rules each with {statement, status, evidence
  turns}; controls map (per-action effect); level-goal hypothesis; open
  questions; active hypothesis + kill criteria. Format: structured markdown
  or JSON — whatever the checker can parse.
- **Competing hypotheses are first-class** (design decision, 2026-08-02).
  When evidence underdetermines a rule, `B` carries the live alternatives,
  not a point estimate. A single committed belief is the measured failure
  mode (ar25 timer-as-progress-bar never revisited; cd82's orbit mechanic
  never even enumerated across 10 trials): when it breaks, the model must
  invent an alternative mid-episode and rationalizes instead, and with one
  hypothesis information gain is undefined — "explore" has no target. With
  ≥2 live hypotheses the planner gets a computable objective: the
  discriminating experiment — cheapest action whose predicted outcomes
  *differ* across hypotheses (an action they agree on teaches nothing about
  that split). Bounds that keep `B` compact:
  1. scope alternatives to *open questions* (factored belief: confirmed core
     + per-question branches, K ≤ 2–4) — never K parallel world models;
  2. every rule carries status `confirmed(evidence turns)` /
     `competing{H1,…}` / `ruled-out(turn, evidence)`; the ruled-out
     graveyard prevents zombie hypotheses and re-proposal after level wipes;
  3. value-of-information filter: spell out alternatives only when they'd
     change a near-term decision, else log an open question;
  4. collapse on resolution: winner → confirmed, losers → graveyard.
- Two separate checkpoints (no interference, independently updatable);
  LoRA is fine for training but **merge before serving** (LoRA-on-GDN vLLM
  decode was 2.8 tok/s — the iter-0001 lesson).

## 3. Training recipes

| | world_model | planner |
|---|---|---|
| Stages | SFT → STaR self-improve → single-turn GRPO (verifiable reward) | SFT cold-start → expert iteration → (GRPO only if EI plateaus) |
| On/off-policy | off-policy throughout (env dynamics are policy-independent) | off-policy cold start, on-policy after |
| Env in the loop? | never (logged + branched-replay transitions) | yes, composed rollouts (~2 h/iteration) |
| Existing rollouts | transitions are free labels; beliefs via hindsight relabeling | success-filtered decisions, relabeled |

### 3.1 world_model (never touches the slow game loop)

- **W1 — SFT**, three tasks multi-tasked:
  1. belief update `(B_{t-1}, a_{t-1}, o_t) → B_t` from hindsight labels,
     incl. mismatch→localized-revision cases;
  2. prediction `(B_t, a_t) → next-obs delta` (target format: §5.2), labels
     free from logs, replay-verified;
  3. cold-start belief from the first frames — target is honest uncertainty
     plus what-to-find-out.
- **W2 — self-improvement**: sample the WM's own belief updates on logged
  contexts; reward = revised belief correctly predicts the next *k*
  logged/replayed transitions. STaR (keep-correct-traces SFT) first, then
  GRPO where one "episode" = one turn against ground truth. Replay
  verification runs at ~130 env-steps/s single-process (measured) — the
  reward check is free relative to generation. Where `B` holds competing
  hypotheses, score predictions as a **proper scoring rule over the
  hypothesis set** (log-loss/Brier-style), not 0/1 on a single guess:
  keeping both alternatives live under insufficient evidence must beat a
  lucky commitment (that's calibration — the transferable skill), while
  confirmed-rule predictions stay committed so permanent hedging doesn't
  pay. Mechanics: one GRPO group = G sampled traces of the *same* context
  (within-group variance is what GRPO differentiates); k > 1 so a rule must
  be real, not a memorized frame; the model reasons freely before the
  structured block, but reward attaches only to the block.
- **Prediction contract — explicit, judge-free** (decision, 2026-08-02):
  `ŝ` is an explicit machine-expandable delta — per-rule/object claims +
  "everything else unchanged" — deterministically expanded to cells and
  scored as exact-match bonus + precision/recall on the **changed-cell
  set**. Never whole-board accuracy: "predict no change" scores ~97% of
  ar25's cells while scoring 0 changed-set recall. Rules-as-code
  (WorldCoder/EWM) is the ceiling — use it when the model can produce it;
  the delta language is the floor; both are machine-checkable with dense
  partial credit. Monolithic per-turn program synthesis is rejected:
  all-or-nothing frame reproduction is sparse reward until near-perfect.
  **LLM-as-judge never sits in the RL reward loop** — a judged reward is
  hackable (GRPO optimizes the judge's soft spots; cf. the summed-KL EOS
  collapse for what a gameable objective buys) and costs a model call per
  sample vs ~µs for cell comparison. Its only place is offline SFT-label
  curation. `B` is likewise explicit (schema-parsed; the planner and the
  filters consume it) but never reward-judged directly — belief quality is
  measured only through downstream prediction accuracy.
- **W3 — refresh** during planner RL on the planner's fresh states, so the
  planner cannot camp on WM blind spots.

Rationale: "predict the next observation" trained naively (obs→obs′
regression) learns texture-copying shortcuts and transfers nowhere. The
prediction is therefore always **conditioned on explicitly stated rules** and
used as the *verifier* of those rules (EWM contract, cf. baseline1 /
arXiv 2605.05138: executable world model + exact-replay verifier, cd82 6/6).
The transferable skill is the mismatch branch: notice violation → localize
the failed rule → revise.

### 3.2 planner

- **P1 — SFT** from: (a) decisions at relabeled-belief states on *successful*
  stretches of logged episodes; (b) ChatGPT re-justifications constrained to
  "derivable from B_t alone"; (c) synthetic drills targeting measured
  pathologies — kill-the-dead-hypothesis, never repeat a no-op action,
  test one variable at a time (the ar25 stuck-loop failures), and
  choose-the-discriminating-experiment: given `B` with competing H1/H2,
  pick the cheapest action whose predicted outcomes differ (plus the dual:
  recognize when a candidate action is uninformative because all live
  hypotheses predict the same outcome).
- **P2 — expert iteration**: roll out the composed pair (25 games × 8
  rollouts ÷ 12–16 concurrent ≈ 2 h/iteration at small-model speeds), filter
  by reward, SFT on winners, repeat. GRPO only if EI plateaus — EI is more
  stable and reuses the SFT infra.
- **Reward**: per-*level* credit primary (score.json v3 tracks per-trial
  levels; a score-0 group still differentiates). Dense tie-breakers, kept
  small (Goodhart): new rules *confirmed* in B (not raw prediction accuracy —
  standing still is perfectly predictable), hypothesis posited→resolved,
  penalty for repeating an action with unchanged board (`board_changed=false`
  is logged; 2–35 no-ops/episode measured, so the signal exists).
- Curriculum: start on games with nonzero small-model signal; mix zero-score
  games in dense-reward-only; expand as the frontier moves.

## 4. Data sources

### 4.1 Existing rollouts (measured, ar25-10-pass run)

Every pass writes `passes/<J>/artifacts/<game>_p<N>_events.jsonl`:

- event types `initial` / `action` / `analysis`;
- `action` events carry the full 64×64 post-action board (`board`,
  `board_ascii`), `action_name`, `action_display`, `board_changed`,
  `level_completed`, `game_over`, `score` (=levels completed), `level`;
- `analysis` events embed the complete LLM transcript (system prompt,
  thinking, python tool calls/results) — ~1.0–1.7 M chars/episode: the
  hindsight-relabeling context;
- ACTION6 click coords exist **only** inside
  `action_display: "MOUSE(row=R, col=C)"` → parse from there; engine data is
  `{"x": col, "y": row}`. (Harness improvement, non-blocking: also log
  structured `action_data`.)
- `p<N>` restarts at 0 in each job dir → episode ids must include the job
  dir (`job5/p2` ≠ `job0/p2`).

One 10-pass ar25 run alone yields **1,230 verified transitions** + 14 M chars
of reasoning context. The full-set runs (25 games) multiply this out with no
harness changes.

### 4.2 Deterministic branched replay (verified)

`arc_agi.Arcade(OFFLINE, environments_dir=…)` + `ONLY_RESET_LEVELS=true` set
*after* `arcade.make()` (mirror `taaf.game_api`) reproduces logged episodes
**exactly**: 10/10 episodes, 1,230/1,230 actions, bit-exact boards, including
RESETs, clicks, game_overs, level transitions. Consequences:

- unlimited new transitions: replay any logged prefix (~130 steps/s), branch
  with scripted/random actions — no LLM, no annotation;
- hindsight-label verification is free: any stated rule/prediction can be
  checked cell-exactly against ground truth;
- WM training needs zero new agent rollouts.

### 4.3 ChatGPT hindsight labeling (distillation at annotation prices)

Two-pass, filtered:

1. **Dossier pass** — one call per episode, full hindsight: mechanics,
   controls, win condition, citing turn evidence. Transcripts are ~1.5 M
   chars/episode → feed the *event stream* (boards/diffs + actions), not raw
   transcripts; chunk if needed.
2. **Per-turn pass** — beliefs/decisions at turn t conditioned on the dossier
   but constrained to observations ≤ t (dossier makes uncertainty statements
   accurate without leaking future facts).
3. **Filters, cheap→expensive**: schema validation → prediction check (B_t
   must predict the logged a_t outcome; compare to logged o_{t+1} — free) →
   hindsight-leak check → k-step rule-consistency spot check via replay.
   The competing-hypotheses requirement makes the leak check *mechanical*:
   a rule marked `confirmed` whose cited evidence turns precede the first
   discriminating observation is by definition a hindsight leak — at
   underdetermined turns the label must carry the live alternatives, not
   the (true) winner.
4. Prioritize level-completing episodes and surprise-rich segments. For games
   no annotator understands, keep prediction-only data, skip belief labels.
   For cd82, baseline1's EWM trajectories are a solved-6/6 source.
5. Budget: ~100 episodes × ~30 turns × ~5 k tokens ≈ 15 M tokens — tens of
   dollars; noise next to GPU time. Check Kaggle external-data rules +
   OpenAI ToS before committing.

## 5. Smoke-test findings → design revisions

`dual_model/smoke/replay_smoke.py`, run
`20260720_134304_ar25-10-pass` (10 episodes, 1,230 actions). Results in
`dual_model/smoke/results/ar25-10-pass.json`.

### 5.1 Confirmed

| Claim | Result |
|---|---|
| transitions extractable from logs | 1,230/1,230, incl. click coords (via `action_display` parse) |
| offline replay deterministic | **10/10 episodes bit-exact**, initial board included |
| replay cheap | ~130 steps/s single process → branching + label-verification ≈ free |
| belief-relevant signal in logs | no-ops 2–35/ep, game_overs 6/10 eps, levels 0–3/ep, per-action `board_changed` |
| relabel context exists | full transcripts embedded in `analysis` events |

### 5.2 Falsified → revised

**"Frame-diffs are tiny" is false on ar25.** Measured cells-changed/action:
median ~108–370, mean 83–268, p90 up to ~550 (of 4,096); raw cell-diff text
is 1.1–3.6 k chars vs ~4.2 k for the full ascii board — only ~1.2–3.8×
compression. Cause: large deterministic *background* dynamics (the 64-action
timer bar repaints every step; animations).

Revision — the WM prediction target is **rule-factored, not cell-level**:

- persistent background rules stated once in `B` ("timer bar loses one cell
  per action") and thereafter *assumed*, not re-emitted;
- per-step prediction = object-level foreground delta ("player moves 1 left;
  orb at (r,c) collected") + "background per rules";
- verification stays cell-exact by *executing* the stated rules (EWM-style —
  the harness already runs a python tool in analysis; rules-as-code is the
  natural form); text-only rules with no machine-expandable delta are
  curation-time material, not RL-reward material (§3.1 prediction contract).

This strengthens the meta-RL story: the trainable skill becomes "compress
dynamics into rules that reproduce the frame", not "paint cells".

### 5.3 Still unverified (next smokes)

- Replay determinism on the other 24 games (sweep the full-set runs; any
  game with true RNG breaks branched replay and needs flagging).
- One hindsight-label round trip: dossier → per-turn labels → filters →
  measured yield/leak rates (needs API access from hpc1).
- gemma4-class protocol-following under the two-model prompt (one harness
  smoke run) — rollout-based training is wasted until format compliance is
  boring.

## 6. Evaluation & controls

- **Held-out games**: ~5 of 25 never annotated, never RL'd — the meta-test
  that the *procedure* was learned. (Same principle as the distillation
  plan's held-out control.)
- Grade the halves separately so composed failures are attributable:
  WM = prediction accuracy on held-out transitions + belief-quality rubric;
  planner = decision accuracy on held-out annotated states.
- Composed = levels/game on the 25 (score.json v3), vs the 27B baseline and
  the gemma4 single-model baseline (run `20260715_210108_gemma4-e4b-fa-bf16`).
- Loss hygiene from the distillation collapse: per-token-mean with length
  control, never summed per-token losses with γ=1 and no baseline.

## 7. Risks

- **Thin meta-train distribution** (25 games): procedural perturbation
  (palette shuffles, grid transforms, rule-parameter tweaks where env source
  permits — we have the env `.py` files) is likely worth more than any
  objective tweak.
- Planner exploiting a frozen WM's errors → W3 refresh cadence.
- Process-reward Goodharting → keep dense terms small; primary reward stays
  per-level.
- Hindsight labels confidently wrong on never-understood games → replay
  filters + skip-belief fallback (§4.3.4).
- Small-model protocol compliance unknown → smoke first (§5.3).

## 8. Next steps

1. Freeze the `B` schema + per-turn protocol (blocks all data generation).
2. Replay-determinism sweep across all 25 games' logged runs.
3. Extraction pipeline: events.jsonl → transition/decision datasets with
   job-qualified episode ids.
4. One-game hindsight-label round trip with filters; measure yield.
5. Two-model harness mode (compose two served models; smoke vanilla gemma4).
6. W1 SFT data build → train → held-out transition eval.
7. P1 SFT → composed eval vs baselines → EI loop.
