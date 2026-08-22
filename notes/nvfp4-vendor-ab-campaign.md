# NVFP4 vendor A/B — vLLM×unsloth vs SGLang×RadixArk

## 0. Brief (as specified)

Test on Kaggle's RTX PRO 6000 servers.

1. **vLLM + unsloth Qwen3.8 NVFP4**. Docs: `https://unsloth.ai/docs/models/qwen3.8`.
   No existing notebook.
2. **SGLang + `RadixArk/Qwen3.8-27B-NVFP4`**. Setup docs: SGLang cookbook
   `.../Qwen/Qwen3.8-27B#hw=rtx6000&...&quant=nvfp4&...&spec=dspark&...`.
   Existing notebook `taaf-qwen38-nvfp4-sglang` — push the next version.

> **Correction to the brief, confirmed.** `taaf-qwen38-nvfp4-sglang` is the **full TAAF
> agent benchmark**, not a probe: it loads `deploy_target.pkl`, plays the 25 games and
> writes competition results. It has no throughput measurement and no sanity battery, so
> neither of the two tests below can run inside it. It is also SGLang-only, so it could
> never host the vendor comparison. It stays in the plan as a **downstream** vehicle
> (§10), gated on the probe. The two tests run in the dedicated probe kernel
> `nvfp4-vendor-ab`, which loads **both** vendors in **one** kernel.

Other parameters stay at the optimal combination: **fp8 KV cache, triton attention
backend, MTP=3**.

Two kinds of test for each arm:

1. **Throughput** — realistic: C=16, ~32k context, generate many tokens.
2. **Sanity** — like `sglang arith check` but more diverse. Confirm the model is not
   visibly degraded by a wrong setup.

## 1. Why the vendor is the variable

The SGLang cookbook pins RadixArk, not unsloth. The two checkpoints differ in exactly
one structural place, visible in `config.json` without loading anything:

| | `unsloth/Qwen3.8-27B-NVFP4` | `RadixArk/Qwen3.8-27B-NVFP4` |
|---|---|---|
| `lm_head` | **FP8** (`group_0`, 8-bit) | **NVFP4** (`group_1`, 4-bit) |
| `lm_head` tensors | `weight`, `weight_scale` | + `weight_scale_2`, `input_scale` |
| layers 56-63 MLP | FP8 | NVFP4 |
| tensors / mtp | 1968 / 15 | 2194 / 15 |

`weight_scale_2` is the NVFP4 double-scale; its absence means the head is FP8. Unsloth
documents the consequence outright: NVFP4 is *"vLLM only for now (SGLang is not
supported)"* because *"SGLang cannot load the FP8 lm_head"*.

`lm_head` projects to vocab logits, so corrupting it degrades **token selection** while
the layers beneath still compute correctly. That is precisely the observed signature:
short answers right (arith check got 6/6 on "51" at 3 tokens), long generations
degenerate with digits dropped and empty `\frac{ }{ }` groups. Both vendors leave `mtp`
unquantized (15 tensors each), so MTP=3 works on either with no separate draft model.

**Hypothesis H1**: the SGLang NVFP4 degeneration was a checkpoint/engine mismatch, not a
property of NVFP4 or of SGLang. Pairing each engine with its vendor's checkpoint fixes it.

## 2. Arms

| arm | engine | checkpoint | status |
|---|---|---|---|
| **A** | vLLM | unsloth | to measure |
| **B** | SGLang | RadixArk | to measure |
| (—) | SGLang | unsloth | **already measured broken** — 5-gram diversity 0.109–0.385 |
| C *(opt-in)* | vLLM | RadixArk | completes the 2×2 |
| F *(opt-in)* | SGLang | RadixArk | arm B with flashinfer attention — §4 |
| K *(opt-in)* | SGLang | RadixArk | **cookbook recipe verbatim**, unmatched ceiling — §3 |

A-vs-B changes engine **and** checkpoint together, so a difference between them cannot be
attributed to either alone. That is acceptable here because H1 is about the *pairing*, and
the third row already pins one off-diagonal cell. Arm **C** is one extra server launch and
would isolate the checkpoint effect within a single engine; it is wired up and opt-in via
`VENDOR_AB_ORDER`, not run by default.

Run **all arms in one kernel**, interleaved `A,B,A,B`, so hardware drift and warm-up
cannot masquerade as an arm effect. One kernel also guarantees identical hardware, which
separate kernels do not.

## 3. Matched configuration

| knob | value | vLLM | SGLang |
|---|---|---|---|
| attention backend | triton | `--attention-backend TRITON_ATTN` | `--attention-backend triton` |
| KV cache dtype | fp8_e4m3 | `--kv-cache-dtype fp8_e4m3` | `--kv-cache-dtype fp8_e4m3` |
| MTP depth | 3 | `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` |
| concurrency | 16 | client-side | client-side |
| context length | 32768 | `--max-model-len 32768` | `--context-length 32768` |
| generation cap | 16384 | `max_tokens` | `max_tokens` |
| sampling | temp 1.0, top_p 0.95, top_k 20 | request | request + `--sampling-defaults openai` |
| tensor parallel | 1 | — | `--tp-size 1` |

**Deviations from the cookbook are defaults, not deletions.** Every cookbook setting we
do not use in the matched arms is still implemented and reachable — matching is a
*default*, and the vendor's own recipe stays runnable as arm **K**.

| cookbook setting | matched arms | reachable via |
|---|---|---|
| `--attention-backend flashinfer` | triton (§4) | arm **F**, arm **K** |
| `--speculative-algorithm DSPARK` + draft path | built-in MTP=3 | arm **K** |
| `--mamba-full-memory-ratio 5.61` | omitted | arm **K** (`mamba=True`) |
| `--mamba-radix-cache-strategy extra_buffer_lazy` | omitted | arm **K** |
| `--mamba-ssm-dtype bfloat16` | omitted | arm **K** |
| `--chunked-prefill-size 2048`, `--mem-fraction-static 0.85` | engine defaults | arm **K** |

The reason to default to matched values is narrow and worth stating precisely: DSPARK has
no vLLM equivalent, and the mamba knobs tune the linear-attention half of the hybrid, so
enabling either in arm B alone would mean a vLLM-vs-SGLang difference could be caused by
the spec stack or the cache tuning rather than by the vendor. That argues for keeping them
out of the **matched comparison** — it does not argue for removing them from the probe.

**Arm K is the cookbook recipe verbatim** (flashinfer + DSPARK + mamba tuning) and is
deliberately *unmatched*. It answers a different and genuinely useful question: *is
matching costing us performance that SGLang would otherwise deliver?* If arm B is healthy
but slow, K tells us whether that is SGLang or our configuration of it — without K, a slow
B would be indistinguishable from a mis-tuned B. Any flag the installed build does not
report in `--help` is skipped with a printed note rather than crashing the launch.

Requires the draft checkpoint `RadixArk/Qwen3.8-27B-DSpark` (2.72 GB, staged). Arms not
listed in `VENDOR_AB_ORDER` do not have their checkpoints validated, so the default 2-arm
run does not need that dataset attached.

Three properties of arm K, read from the draft's `config.json` and the installed SGLang,
that make it *more* unmatched than the flag list alone suggests:

- **The draft is unquantized BF16** — no `quantization_config`, 5 layers, `dtype
  bfloat16`. It is 2.72 GB of resident weights that the matched arms do not pay for, and
  it shrinks the KV pool. So arm K trades context headroom for acceptance rate; at 32k
  context that trade is not obviously favourable and is part of what K measures.
- **DSPARK speculates deeper than MTP=3.** `block_size: 7` and SGLang's
  `--speculative-dspark-block-size` help text states the verify window is gamma + 1, so K
  runs **8 draft tokens against our 4**. Any K-vs-B throughput gap therefore mixes the
  backend, the draft model, the mamba tuning *and* roughly double the speculation depth.
  K is a ceiling, not an attribution.
- **`DSPARK` is a valid algorithm in the installed build** (`server_args.py:2026` lists
  `EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK`), and the draft loads through
  `auto_map -> dspark.DSparkDraftModel`, so `--trust-remote-code` is required. It is
  already passed. Checking this mattered: the launch guard asserts the *flag* exists, not
  that the *value* is accepted, so an unsupported algorithm would have failed at runtime
  mid-run rather than at argument parsing.

## 4. Why triton, not the cookbook's flashinfer

**The attention backend and the NVFP4 numerics are separate axes.** Choosing triton does
*not* move NVFP4 off its supported path, because SGLang imports FlashInfer for the FP4
linear layers regardless of `--attention-backend`:

- `srt/layers/quantization/fp4_utils.py` binds `fp4_quantize` to `flashinfer.fp4_quantize`
  at import, with `backend="cute-dsl" if is_sm100_supported() else "cuda"`.
- `initialize_fp4_gemm_config()` under the default `auto` resolves: `is_sm100_supported()`
  → `flashinfer_cutedsl`; `(10,0) > cap >= (8,0)` → `marlin`; **else → `flashinfer_cutlass`**.
- `is_sm100_supported` is `device_capability_majors=[10]` — **exactly major 10**. The
  RTX PRO 6000 is major 12, so it takes neither the first branch nor marlin.

So on this GPU the NVFP4 GEMM is `flashinfer_cutlass` and the quantizer is FlashInfer's
`"cuda"` backend, whichever attention backend we pass. `--attention-backend` only picks
the kernel for QK^T/PV over the KV cache — which NVFP4 does not touch, since it quantizes
*linear-layer weights*. The mechanism behind the measured triton win is therefore
unaffected by the move from FP8 to NVFP4 weights.

Consequence worth stating: **FlashInfer must work in arm B anyway.** Its JIT dies with
`/usr/bin/ld: cannot find -lcuda` unless `LIBRARY_PATH=/usr/local/nvidia/lib64`. Picking
triton does *not* avoid this. The probe sets it in both `sglang_env()` and `vllm_env()`.

**Measured on this exact GPU** (`sm120-kv-headroom-probe`, vLLM, FP8 weights, MTP=3,
identical KV capacity of 356,800 tokens): TRITON_ATTN **220.14** vs FLASHINFER **161.56**
agg gen tok/s — 1.36×. FlashInfer's Blackwell edge is the trtllm-gen kernel family, gated
to SM100 and closed on SM120; without it it falls back to generic kernels while still
paying a `PIECEWISE` cudagraph downgrade under spec-decode, where triton keeps
`AttentionCGSupport.ALWAYS`. At MTP=3 that is paid per draft step.

**Limits**: measured under **vLLM**, n=1, no warm-up, prefill-dominated. vLLM's FLASHINFER
backend is not SGLang's, and we have zero SGLang backend numbers on sm_120. Optional arm
**F** (= arm B with `--attention-backend flashinfer`, changing nothing else) settles it on
SGLang; opt-in via `VENDOR_AB_ORDER`.

Holding the backend constant also protects the vendor comparison: an effect that applies
to both arms shifts the *level*, not the A−B *difference*.

## 5. Test 2 — sanity / anti-degeneration (primary gate)

This gates test 1: **an arm that fails here gets no throughput number**, because tokens
per second on degenerate output is not a measurement of anything.

The predecessor `sglang arith check` was too narrow — it judged health from ~2-token
answers, which is exactly the regime where the broken checkpoint looks fine. Diversity
here means both *longer* and *broader*: **6 items**, each generating up to **16,384
tokens** — `long_explain` (arithmetic with digit echo), `long_narrative`, `long_arc`,
`count_to_60`, `structured_json`, `code_gen` — plus a separate tool-calling check.

| metric | threshold | what it catches |
|---|---|---|
| 5-gram diversity | > 0.70 | repetition loops |
| max 5-gram repeat | < 20 | stuck loops |
| digit retention | > 0.95 | dropped operands (the observed NVFP4 signature) |
| empty-LaTeX count | 0 | `\frac{ }{ }` with the operands gone |
| stop-vs-length frac | > 0.5 | never emitting EOS |
| tool-call validity | 100% | fluent prose with malformed calls still scores 0 in the real harness |
| JSON array valid | must parse | objective correctness, not just fluency: 12 objects, ids 1..12, `square == id²` |

The 0.70 gate is set from measurement, not taste: it separates every broken NVFP4 row
(0.109–0.385) from every healthy FP8 row (0.873–0.984) on the same server.

**Gate 2 — loader sanity.** SGLang reporting `"quantization": null` is how the original
failure hid: the loader silently ignored the quant config. Every arm records
`/get_server_info` (SGLang) or the resolved engine config (vLLM), asserts `quantization`
is not null, and logs `fp4_gemm_runner_backend` — expected `flashinfer_cutlass` per §4,
which confirms the branch analysis on real hardware rather than on my reading of it.

## 6. Test 1 — throughput

C=16 matched load, 32k context, 4096-token generations. Reported: aggregate generation
tok/s, per-request tok/s, TTFT, KV cache tokens, and spec-decode accept rate.

**Caveat to apply when reading the result**: this is a *level* comparison across two
different engines, so it confounds vendor with engine scheduler, kernel maturity, and KV
accounting. It is a deployment-choice number, not evidence about NVFP4 itself. The
sanity result is what tests H1.

## 7. Phases

| phase | what | kill condition |
|---|---|---|
| **0 — feasibility** | each server starts, passes Gate 2, emits a coherent 2k sample | if vLLM cannot load NVFP4 on sm_120 at all, arm A is dead — report and stop |
| **1 — sanity** | full 16k battery, all Gate-1 metrics | an arm failing Gate 1 is excluded from phase 2 |
| **2 — throughput** | C=16 matched load | — |
| **3 — agreement** *(opt)* | temp-0 cross-arm token agreement | — |
| **4 — backend** *(opt)* | arm F: arm B with flashinfer attention | skip if arm B failed Gate 1 |
| **5 — cookbook** *(opt)* | arm K: the vendor's full recipe, unmatched | skip if arm B failed Gate 1 |

## 8. Risks

1. **vLLM wheelhouse may predate NVFP4 support on sm_120.** Mitigated: the upstream fix
   for vllm#40756 was validated on Qwen3.8-27B-NVFP4 / sm_120 / vLLM 0.27.1 with 4096
   requests at concurrency 256, zero errors — direct evidence for arm A's exact pairing.
   The probe reports `vllm_version` and `mtp_patch` so a negative is diagnosable.
2. **MTP=3 CUDA-graph race** (vllm#40756) — sticky device-side assert. `scripts/patch_vllm_mtp_race.py`
   is applied; it needed `--site-packages` because Kaggle's `pip --target` layout is flat
   and the venv globs would have matched nothing and patched no arms silently.
3. **SGLang EAGLE ≠ vLLM MTP** in scheduling detail even at the same depth. Accept-rate is
   recorded per arm so a depth mismatch is visible rather than silent.
4. **FlashInfer JIT** — see §4; `LIBRARY_PATH` is set in both envs.
5. **Kaggle mount paths** are `/kaggle/input/datasets/<owner>/<slug>`, not `/kaggle/input/<slug>`.
6. **One kernel, four+ model loads** — runtime budget. Interleaving is worth it; drop
   optional arms before dropping interleaving.

## 9. Decision rule

| outcome | reading | action |
|---|---|---|
| A healthy, B healthy | H1 confirmed — it was the vendor mismatch | pick on throughput; push the RadixArk next version of `taaf-qwen38-nvfp4-sglang` |
| A healthy, B broken | vendor mismatch was not the whole story; SGLang NVFP4 suspect on sm_120 | stay on vLLM; run arm C to check whether the checkpoint alone explains it |
| A broken, B healthy | unsloth NVFP4 unusable here despite being the vLLM-targeted build | move to SGLang×RadixArk |
| both broken | NVFP4 is not viable on sm_120 at 16k generations | stay on FP8 (3.26×, not 4.21×) and close the line |

## 10. Vehicles

- **Probe kernel** `jonathanwang2022/nvfp4-vendor-ab` (`ARC3-Inference/scripts/kaggle_nvfp4_vendor_ab_probe.py`) — both arms,
  both tests, one kernel, interleaved. This is what produces the measurements.
- **Agent notebook** `jonathanwang2022/taaf-qwen38-nvfp4-sglang` — **a full benchmark, not
  a probe**, and SGLang-only. It cannot measure throughput or sanity and cannot compare
  vendors; it answers a different question (does the agent score) and is the vehicle for
  *acting on* the probe's verdict, not for reaching it. Its next version points at
  RadixArk. Its launch path lives in
  `ARC3-Inference/inference/framework/kaggle.py`; the checkpoint is overridable at runtime
  via `KAGGLE_MODEL_DATASET_SOURCE` / `KAGGLE_SERVED_MODEL_NAME`, so no code edit is needed
  to repoint it — but `DEFAULT_QWEN_NVFP4_MODEL_DATASET_SOURCE` carries a
  "do not point the model dataset here" warning written from the *unsloth* failure, and
  that comment must be corrected before it misleads the next reader.

Gate the notebook push on the probe: it is the benchmark, so running it before the probe
says which pairing is coherent spends a full benchmark slot to re-learn what a 20-minute
probe answers.

## 11. Provenance

- Arith check `jonathanwang2022/sglang-arith-check` — short answers correct, long
  generations degenerate with operands dropped.
- `jonathanwang2022/sglang-quality-matrix` — NVFP4 5-gram diversity 0.109–0.385 vs FP8
  0.873–0.984; spec-decode OFF made it *worse* (0.031); bf16 KV did not help.
- `jonathanwang2022/sm120-kv-headroom-probe` — triton vs flashinfer, fp8 KV block model.
- `unsloth.ai/docs/models/qwen3.8` — "vLLM only for now (SGLang is not supported)".
- Both `config.json` / `model.safetensors.index.json` read directly from HF.
- SGLang FP4 dispatch read from the installed tree at
  `ARC3-Inference/.venv-sglang/.../sglang/srt/layers/quantization/fp4_utils.py`.

Everything above is offline-checkable and has been checked. **Nothing about how either
engine behaves on Blackwell has been verified**, because it cannot be verified here — the
cluster is H100 (sm_90) and NVFP4 needs Blackwell. That is what the Phase 0 run is for.

## 12. Run 1 results (kernel `nvfp4-vendor-ab` v1, 2026-08-21, ~45 min)

**H1 CONFIRMED.** Arm B (SGLang × RadixArk) generates coherent text, reproducibly:

| metric | B rep1 | B rep3 | broken NVFP4 (unsloth) |
|---|---|---|---|
| mean 5-gram diversity | 0.805 | 0.811 | 0.109–0.385 |
| digit retention | 1.0 | 1.0 | operands dropped |
| empty LaTeX | 0 | 0 | present |
| stop (not length) | 1.0 | 1.0 | — |
| JSON array valid | true | true | — |
| longest coherent gen | 15,837 tok @ 0.950 | 14,340 tok | — |

The vendor mismatch was the cause of the original degeneration.

**Arm A (vLLM × unsloth) died at engine-core init, both reps** — `vllm_version 0.19.0`,
`mtp_patch applied`, startup 135 s then 30 s. **Root cause unknown and NOT fixed**: all
arms shared one log path, the last arm overwrote it, and the 40-line tail captured ends at
`RuntimeError: Engine core initialization failed. See root cause above.` The cause was
above the tail.

### Defects in the probe that run 1 exposed

| # | defect | consequence | fix |
|---|---|---|---|
| 1 | load prompt was 7,936 tok | C=16 ran at ~12k, not the ~32k in the brief | 776 rows = 27,952 tok, peak 32,048 |
| 2 | tool-call check had 256 tok under `reasoning_effort xhigh` | thinking ate the budget, 0/4 calls, healthy arm failed the gate | 4,096 tok + per-request thinking off |
| 3 | 5-gram split on whitespace only | `count_to_60` (comma-separated) scored diversity 0.0 | split on `\w+` |
| 4 | `quant_ok` used as a gate | arm B was coherent with `quantization: null` | demoted to informational |
| 5 | one shared server log; 40-line **tail** | a dead arm's root cause is unrecoverable | per-arm logs + `first_error()` |
| 6 | tool-call check recorded only a pass count | 0/4 was undiagnosable | per-attempt finish/tokens/snippet |

### Findings that change the plan

- **`max_running_requests` was capped to 17**, not the 48 requested, *"by the mamba state
  cache (max_mamba_cache_size=85, 5 state slots per request)"*. SGLang's own suggested
  remedies are `--mamba-full-memory-ratio` / `--mamba-ssm-dtype bfloat16` — **the cookbook
  flags §3 had removed**. At C=16 the run sat at 16 of 17 slots. The cookbook's `5.61` vs
  the default `0.9` is roughly 6× the state-cache headroom. These flags gate concurrency on
  this hybrid model; they are not cosmetic.

  The arithmetic is explicit in the log: `max_mamba_cache_size=85, 5 state slots per
  request` -> 85/5 = 17. Three levers move it: `--mamba-full-memory-ratio` (0.9 default vs
  the cookbook's 5.61), `--max-mamba-cache-size` (slots directly), or
  `--mamba-ssm-dtype bfloat16` (halves per-state size, ~2x slots).

  **Raising it is not free.** The mamba state cache and the KV cache draw on the same GPU
  memory, so a higher ratio buys concurrency slots by taking memory from KV -- which at 32k
  context reduces how many long requests fit. There is a real optimum and it is not "as
  high as possible". Arm K is what would locate it; nothing here is measured, since the
  throughput test never executed.

  **Two different pools both land near 16, and they are not the same mechanism.** C=16 was
  chosen from the *KV* budget (measured Kaggle `GPU KV cache size: 171,200 tokens`, 9.13x
  at 65k -> ~18x at 32k). The 17 cap came from the *mamba state cache*, governed by
  `--mamba-full-memory-ratio`. Both descend from the same 96 GB, so they were never going
  to be far apart -- but **the KV-based estimate would not have predicted 17**, and the
  binding constraint on run 1 was the pool we had not modelled. Nor is the trade strictly
  zero-sum against KV: `--mamba-ssm-dtype bfloat16` roughly doubles state slots by halving
  per-state size, without taking KV memory.

  Counter-evidence against simply pushing concurrency up: the KV-starvation work found 16
  was *too high* on H100 nodes -- same aggregate server tok/s, but per-request decode
  halved (59->28), prefix cache hit rate collapsed (47.7% -> 13.0%), and 240 s timeouts
  went 54 -> 367 with 32% of generated tokens discarded. Different GPU tier and the agent
  harness rather than this probe, so it does not transfer directly, but it means the
  honest ceiling is empirical and "raise the ratio, raise C" may well regress.
- **`quantization: null` is not a health signal** — see defect 4. Gate 2's original premise
  was wrong.
- **Risk 1 was under-weighted.** It anticipated "the wheelhouse may predate NVFP4 support on
  sm_120", then largely dismissed it because upstream validated NVFP4/sm_120 on vLLM
  **0.27.1**. Our wheelhouse is **0.19.0**, and the dataset is named
  `arc3-vllm-h100-wheelhouse-v3` — built for H100 / sm_90. Evidence about a much newer vLLM
  was allowed to reassure us about a much older one.

### Single-stream throughput (C=1, arm B, prefill included)

No C=16 number exists: throughput is gated on health, and health failed on defect 2.

| trial | rep1 tok/s | rep3 tok/s |
|---|---|---|
| long_narrative (~7k) | 77.4 | 78.9 |
| long_arc (~15k) | 70.9 | 73.0 |
| code_gen (~9k) | 100.7 | 93.3 |
| **pooled** | **73.5** | **83.5** |
## 13. Run 2 (kernel v2) — died at install, 75 s

Not a GPU result. The vLLM 0.27.1 wheelhouse was **resolved on the login node's python
3.11.13**, and vLLM 0.27.1 gates two requirements on `python_version > "3.11"`:

```
six>=1.16.0;                 python_version > "3.11"
setuptools<81.0.0,>=77.0.3;  python_version > "3.11"
```

Both evaluated **false** locally, so pip silently dropped them. `--python-version 3.12`
steers wheel-**tag** selection only; it does not drive **marker** evaluation. On Kaggle
(python 3.12, markers true) this produced two faults, the second of which would have bitten
immediately after fixing the first:

1. `six` absent → `No matching distribution found` → install aborted, kernel ERROR at 75 s.
2. `setuptools` had resolved to **84.0.0** via another package's unconstrained requirement —
   violating vLLM's own `<81.0.0`, because that pin was marker-excluded.

**Fix**: resolve with a real python 3.12 interpreter (`ARC3-Inference/scripts/build_vllm_wheelhouse.sh`). Result:
`six==1.17.0`, `setuptools==80.10.2`, 195 wheels.

### The check that should have existed

`scripts/verify_vllm_wheelhouse.sh` resolves the lock exactly as Kaggle will, with no GPU minute spent.
Both conditions are required or it proves nothing:

- **python 3.12 interpreter** — markers evaluate as Kaggle evaluates them (what v2 got wrong)
- **`--platform` overrides** — this login node is glibc 2.17, so `manylinux_2_28` wheels are
  not candidates locally and every one of them reads as a spurious "no matching version"

`scripts/verify_vllm_flags.py` covers the adjacent failure class: an unrecognised CLI flag is
`SystemExit(2)` with no server and no useful log. The probe already parses `--help` for
SGLang; vLLM had no equivalent guard. All 16 flags the probe passes resolve in 0.27.1, and
`TRITON_ATTN` / `fp8_e4m3` are still valid values.

### Diagnosability, and why the "trivial" change mattered

Run 1's output was unreadable in practice: `SITE_PACKAGES` lived under `/kaggle/working`, so
`kaggle kernels output` tried to pull ~10 GB of torch + CUDA before reaching the results, and
the arm-A log was never retrieved. Moving it to `/tmp` made run 2's output **57 KB including
the kernel log** — which is the only reason the marker bug was diagnosed in one pass rather
than by re-running blind.

### Lesson, stated generally

A dependency resolution performed under a different interpreter than the target is not a
resolution — it is a guess that usually looks identical. Resolve on the target's python
version, and prove it by resolving the lock again under those exact constraints.

## 14. Run 5 — CONCLUSIVE. vLLM wins, ~2.5x

Both engines now serve NVFP4 on sm_120, each with its own vendor's checkpoint. H1 confirmed
in full: **the vendor mismatch was the whole of the original problem.**

| | vLLM 0.27.1 × unsloth | SGLang × RadixArk |
|---|---|---|
| C=16 aggregate | **544.8 / 578.6 tok/s** | 225.5 / 229.1 tok/s |
| per-request decode | **37.2 / 38.0** | 14.8 / 15.5 |
| wall clock, 65,536 tok | **120.3 / 113.3 s** | 290.6 / 286.0 s |
| requests ok | 16/16 | 16/16 |
| MTP accept length | 2.54 / 2.54 | 2.65 / 2.62 |
| sanity | pass, diversity 1.000 | pass, 0.898 / 0.962 |

Matched and verified from the engine log, not assumed: both arms generated **exactly 65,536
tokens**; vLLM shows `SpeculativeConfig(method='mtp', num_spec_tokens=3)`, `Resolved
architecture: Qwen3_5MTP`, `Loading drafter model...`, `quantization=compressed-tensors`,
`TRITON_ATTN`, `fp8_e4m3`, `max_model_len 32768`.

**The interesting shape: SGLang has the slightly better acceptance length and is still 2.5x
slower.** The gap is raw execution speed, not speculation quality.

### Runs 3-5, the three vLLM failures and their fixes

| run | arm A outcome | cause | fix |
|---|---|---|---|
| 3 | died | `No supported CUDA architectures found for major versions [12]` — no nvcc, so FlashInfer's `_normalize_cuda_arch` raised and an `except Exception` swallowed it, leaving the arch list empty | `CUDA_HOME`, nvcc on `PATH`, `FLASHINFER_CUDA_ARCH_LIST=12.0f` |
| 4 | died | `/usr/bin/ld: cannot find -lcudart` — wheels ship only `libcudart.so.13`; `LIBRARY_PATH` (link time, not `LD_LIBRARY_PATH`) lacked the wheel lib dirs | 42 unversioned soname aliases + wheel lib dirs on `LIBRARY_PATH` |
| 5 | **ok** | — | — |

The unifying lesson: **`--attention-backend TRITON_ATTN` does not keep FlashInfer out of the
NVFP4 path.** vLLM routes the FP4 GEMM through `flashinfer_mm_fp4 -> cutlass ->
get_gemm_sm120_module_cutlass_fp4()`, which JIT-compiles at engine init. Attention backend
and FP4 GEMM are separate axes — the same split that holds in SGLang.

### An instrumentation bug that produced an impossible number

Run 5 first reported vLLM `accept_len_mean 25.98` — impossible when `num_spec_tokens=3` caps
the length at 4. The regex missed vLLM's `acceptance length:` wording, fell through to
`acceptance rate:`, and averaged **two different units from one file** (`46.7` percent
alongside `0.667` fraction). True values 2.54 / 2.65. Fixed to read the length only.

### Decision

**Kaggle engine: vLLM 0.27.1 + `unsloth/Qwen3.8-27B-NVFP4`.** Full migration procedure in
`VLLM-KAGGLE-SETUP.md`. `0820_report.md` rows 5 and 8 superseded.

Untested: SGLang may win at low concurrency — every number here is C=16.
Still confounded: A-vs-B moves engine and checkpoint together; arm C would separate them.
