# Dual-model training: world_model + planner

## 1. Goal

Learn the *adaptation procedure* — memorize details, update beliefs from the
latest observation, plan under uncertainty — not per-game policies. The
Kaggle hidden set (110 games) shares no specifics with our 25 offline games,
so game-specific weights are dead weight. Meta-RL: weights encode how to
learn a game; the learning itself happens in context.

Motivating failures of the single-model agent: belief upkeep is an optional
tool call the 27B makes on ~4–6% of turns (and level wipes clear it); the
planning failures are hypothesis-bookkeeping failures (no hypothesis kill,
timer misread as progress bar, repeated no-ops).

Compute: 27B ≈ 1 h/game (H100, ~49 tok/s batch=1); gemma4-class ≈ 8–10
min/episode at >500 tok/s, 8–16 concurrent per H100. Small-model score signal
on the 25 games is tiny but nonzero → needs dense rewards + curriculum, not
score-only RL.

## 2. Architecture

Per-turn protocol over two small models:

1. `world_model(B_{t-1}, a_{t-1}, s_t, ŝ_t)` → mismatch analysis → revised
   belief `B_t` → (after the planner commits `a_t`) prediction `ŝ_{t+1}`
2. `planner(B_t, s_t, score/level, budget)` → hypothesis under test +
   expected outcome + `a_t`

- Belief update is an unconditional pipeline stage, not a tool call.
- The planner sees `B` + the current frame only; history lives compressed in
  `B`. That bottleneck is what makes `B` a real memory.
- Two checkpoints, independently updatable. LoRA for training, but **merge
  before serving** (LoRA-on-GDN vLLM decode = 2.8 tok/s).

### 2.1 Belief `B` (freeze the schema before generating data)

Entities; dynamics rules; controls map; level-goal hypothesis; open
questions; active hypothesis + kill criteria. Markdown or JSON — whatever the
checker parses.

Every rule carries a status: `confirmed(evidence turns)` / `competing{H1,…}`
/ `ruled-out(turn, evidence)`.

**Competing hypotheses are first-class**: under underdetermined evidence `B`
holds the live alternatives, not a point estimate. This gives the planner a
computable exploration target — the **discriminating experiment**, the
cheapest action whose predicted outcomes differ across live hypotheses (an
action they all agree on teaches nothing). Bounds:

1. alternatives scoped per open question (K ≤ 2–4), never K world models;
2. the `ruled-out` graveyard prevents zombie hypotheses and re-proposal after
   level wipes;
3. spell out alternatives only when they'd change a near-term decision;
4. collapse on resolution: winner → confirmed, losers → graveyard.

## 3. Training

| | world_model | planner |
|---|---|---|
| Stages | SFT → STaR → single-turn GRPO | SFT → expert iteration → GRPO only if EI plateaus |
| On/off-policy | off-policy throughout | off-policy cold start, on-policy after |
| Env in the loop | never | yes, composed rollouts (~2 h/iteration) |

### 3.1 world_model — never touches the game loop

**W1 SFT**, multi-tasked: (a) belief update `(B_{t-1}, a_{t-1}, s_t) → B_t`
from hindsight labels, including mismatch→localized-revision cases; (b)
prediction `(B_t, s_t, a_t) → ŝ_{t+1}` teacher-forced on the logged next
state; (c) cold-start belief from the first frames, where the target is
honest uncertainty plus what to find out.

**W2 self-improvement**: sample the WM's own belief updates on logged
contexts; reward = the revised belief correctly predicts the next *k*
transitions. STaR first, then GRPO with one turn as the "episode". Mechanics:
a GRPO group is G traces of the *same* context; k > 1 so a rule must be real
rather than a memorized frame; free reasoning before the structured block,
reward on the block only. Replay verification is ~130 steps/s — free relative
to generation.

With competing hypotheses live, score predictions as a **proper scoring rule
over the hypothesis set**, not 0/1 on one guess: keeping alternatives under
insufficient evidence beats a lucky commitment (calibration), while
confirmed-rule predictions stay committed so permanent hedging doesn't pay.

**W3 refresh** on the planner's fresh states during planner RL, so the
planner can't camp on WM blind spots.

Prediction is always *conditioned on explicitly stated rules* and serves as
the **verifier** of those rules (EWM contract, cf. baseline1 / arXiv
2605.05138, cd82 6/6). Naive obs→obs′ regression learns texture-copying
shortcuts that transfer nowhere. The transferable skill is the mismatch
branch: notice violation → localize the failed rule → revise.

### 3.2 Prediction contract — explicit, judge-free

`ŝ` is a machine-expandable **delta**: per-rule/object claims plus
"everything else unchanged", deterministically expanded to cells.

- Score = exact-match bonus + precision/recall on the **changed-cell set**.
  Never whole-board accuracy: "no change" scores ~97% of ar25's cells at 0
  changed-set recall.
- Rules-as-code (WorldCoder/EWM) is the ceiling, the delta language the
  floor; both machine-checkable with dense partial credit. Monolithic
  per-turn program synthesis is rejected — all-or-nothing frame reproduction
  is sparse reward until near-perfect.
- Persistent background dynamics (ar25's timer repaints ~108 cells every
  step) are stated once in `B` and thereafter assumed, not re-emitted.
- **No LLM-as-judge in the RL reward loop**: judged rewards are hackable
  (GRPO optimizes the judge's soft spots) and cost a model call per sample vs
  ~µs for cell comparison. Judges are for offline label curation only.
- `B` is explicit so the planner and filters can parse it, but never
  reward-judged directly — belief quality is measured only through downstream
  prediction accuracy.

### 3.3 planner

**P1 SFT** from: decisions at relabeled-belief states on *successful*
stretches of logged episodes; ChatGPT re-justifications constrained to
"derivable from `B_t` alone"; synthetic drills for the measured pathologies —
kill-the-dead-hypothesis, never repeat a no-op, test one variable at a time,
choose-the-discriminating-experiment (and its dual: recognize an
uninformative action when all live hypotheses predict the same outcome).

**P2 expert iteration**: roll out the composed pair (25 games × 8 rollouts ÷
12–16 concurrent ≈ 2 h/iteration), filter by reward, SFT on winners, repeat.
EI is more stable than GRPO here and reuses the SFT infra.

**Reward**: per-*level* credit primary (score.json v3 has per-trial levels, so
a score-0 group still differentiates). Small dense tie-breakers: new rules
*confirmed* in `B` (not raw prediction accuracy — standing still is perfectly
predictable), hypothesis posited→resolved, penalty for repeating an action
with unchanged board (`board_changed=false` is logged; 2–35 no-ops/episode).

**Curriculum**: start on games with nonzero small-model signal; mix in
zero-score games under dense reward only; expand as the frontier moves.

## 4. Data

### 4.1 Logged rollouts

`passes/<J>/artifacts/<game>_p<N>_events.jsonl`, event types `initial` /
`action` / `analysis`:

- `action` events carry the full 64×64 post-action board (`board`,
  `board_ascii`), `action_name`, `action_display`, `board_changed`,
  `level_completed`, `game_over`, `score` (= levels completed), `level`;
- `analysis` events embed the full LLM transcript (~1.0–1.7 M chars/episode)
  — the hindsight-relabeling context.

Extraction gotchas: ACTION6 click coords exist **only** inside
`action_display: "MOUSE(row=R, col=C)"` (engine data is `{"x": col, "y":
row}`), and `p<N>` restarts per job dir, so episode ids must be job-qualified
(`job5/p2` ≠ `job0/p2`).

One 10-pass ar25 run = 1,230 verified transitions + ~14 M chars of context;
full-set runs multiply this out with no harness changes.

### 4.2 Deterministic branched replay

`arc_agi.Arcade(OFFLINE, environments_dir=…)` with `ONLY_RESET_LEVELS=true`
set *after* `arcade.make()` (mirroring `taaf.game_api`) reproduces logged
episodes exactly — 10/10 episodes, 1,230/1,230 actions, bit-exact, including
RESETs, clicks, game_overs, level transitions, at ~130 steps/s. Therefore:

- unlimited new transitions: replay any prefix, branch with scripted/random
  actions — no LLM, no annotation;
- hindsight labels verify for free, cell-exactly, against ground truth;
- WM training needs zero new agent rollouts.

Re-verify per game: any game with true RNG breaks branched replay and must be
flagged.

### 4.3 ChatGPT hindsight labeling

ChatGPT labels only `B_t` and a per-episode dossier; `s_t`, `a_t`, `s_{t+1}`
come free from the logs.

1. **Dossier pass** — one call per episode with full hindsight: mechanics,
   controls, win condition, citing turn evidence. Feed the event stream
   (boards/diffs + actions), not raw transcripts.
2. **Per-turn pass** — beliefs/decisions at turn t conditioned on the dossier
   but constrained to observations ≤ t, so uncertainty statements are
   accurate without leaking future facts.
3. **Filters, cheap→expensive**: schema validation → prediction check (`B_t`
   must predict the logged `a_t` outcome against logged `s_{t+1}` — free) →
   hindsight-leak check → k-step rule-consistency spot check via replay.
   Competing hypotheses make the leak check mechanical: a rule marked
   `confirmed` whose evidence turns precede the first discriminating
   observation is by definition a leak.
4. Prioritize level-completing and surprise-rich segments. Where no annotator
   understands the game, keep prediction-only data and skip belief labels;
   for cd82, baseline1's EWM trajectories are a solved-6/6 source.
5. ~100 episodes × ~30 turns × ~5 k tokens ≈ 15 M tokens — tens of dollars.
   Check Kaggle external-data rules + OpenAI ToS first.

## 5. Evaluation

- **~5 held-out games**: never annotated, never RL'd — the meta-test that the
  procedure was learned rather than the games.
- Grade halves separately so composed failures are attributable: WM =
  prediction accuracy on held-out transitions + belief rubric; planner =
  decision accuracy on held-out annotated states.
- Composed = levels/game on the 25 (score.json v3) vs the 27B baseline and
  the gemma4 single-model baseline (`20260715_210108_gemma4-e4b-fa-bf16`).
- Loss hygiene: per-token-mean with length control; never summed per-token
  losses with γ=1 and no baseline.

## 6. Risks

- **Thin meta-train distribution** (25 games): procedural perturbation
  (palette shuffles, grid transforms, rule-parameter tweaks — we have the env
  `.py` files) is likely worth more than any objective tweak.
- Planner exploiting a frozen WM's errors → W3 refresh cadence.
- Process-reward Goodharting → keep dense terms small, primary stays
  per-level.
- Hindsight labels confidently wrong on never-understood games → replay
  filters + skip-belief fallback.
- Small-model protocol compliance is unverified → smoke it before building
  the rollout loop.

## 7. Next steps

1. Freeze the `B` schema + per-turn protocol (blocks all data generation).
2. Replay-determinism sweep across all 25 games.
3. Extraction pipeline: events.jsonl → transition/decision datasets.
4. One-game hindsight-label round trip; measure yield and leak rates.
5. Two-model harness mode; smoke vanilla gemma4 protocol-following.
6. W1 SFT → held-out transition eval.
7. P1 SFT → composed eval vs baselines → EI loop.

Smoke test backing §4.1–4.2: `dual_model/smoke/replay_smoke.py`, results in
`dual_model/smoke/results/`.
