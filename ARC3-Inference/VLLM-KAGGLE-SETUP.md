# vLLM + unsloth NVFP4 on Kaggle RTX PRO 6000 — the complete setup

**Status: this is the current Kaggle engine decision. It supersedes the SGLang decision in
`0820_report.md` §1 row 8.**
**Date: 2026-08-22 · every number below is measured on the target GPU.**

**Wired on `main` as of 2026-08-21.** `configs/inference.json` `kaggle.backend` is `vllm`,
pointing at `vllm-0271-wheelhouse-sm120` + `qwen38-27b-nvfp4-unsloth`, and every trap below
is handled in `inference/framework/kaggle.py`: `/tmp` install (§5), soname aliases and exec
bits in `repair_vllm_install()` (§6), the FP4 JIT toolchain in `vllm_env()` (§7), the argv
in `start_vllm_server()` (§8), and the MTP race patch against **both** target files
(Trap 6). One item from §11 is deliberately NOT applied: `concurrent_jobs` /
`max_runtime_minutes` are still the SGLang-derived 16 / 71.

Audience: an agent migrating the Kaggle submission onto vLLM. Everything here was learned by
failing on a billed GPU. **Read §2 before changing anything** — five of the six traps produce
error messages that point at the wrong cause.

---

## 1. The decision and its evidence

Kernel `jonathanwang2022/nvfp4-vendor-ab` v5, one kernel, arms interleaved `A,B,A,B` so
hardware drift cannot masquerade as an arm effect.

| | **vLLM 0.27.1 × unsloth NVFP4** | SGLang 0.5.17 × RadixArk NVFP4 |
|---|---|---|
| C=16 aggregate | **544.8 / 578.6 tok/s** | 225.5 / 229.1 tok/s |
| per-request decode | **37.2 / 38.0 tok/s** | 14.8 / 15.5 tok/s |
| wall clock for 65,536 tok | **120.3 / 113.3 s** | 290.6 / 286.0 s |
| requests ok | 16/16 | 16/16 |
| MTP accept length | 2.54 / 2.54 | 2.65 / 2.62 |
| sanity gate | pass (diversity 1.000) | pass (0.898 / 0.962) |

**≈2.5x, reproducible across both reps.** Matched: both arms generated *exactly* 65,536
tokens, MTP=3, triton attention, `fp8_e4m3` KV, 32,768 context, C=16, same GPU, same kernel.

Note the shape of the result: **SGLang has the slightly better acceptance length and is
still 2.5x slower.** The gap is raw execution speed, not speculation quality.

**Untested and plausible:** SGLang may win at low concurrency. Every number here is C=16.
Nothing in this document licenses a claim about C=1..4.

**Still confounded:** A-vs-B moved engine *and* checkpoint together. Arm C
(vLLM × RadixArk) separates them and has not been run.

---

## 2. The six traps, each with the misleading error it produces

### Trap 1 — resolving the wheelhouse on the wrong python **silently drops packages**

vLLM 0.27.1 gates two requirements on `python_version > "3.11"`:

```
six>=1.16.0;                 python_version > "3.11"
setuptools<81.0.0,>=77.0.3;  python_version > "3.11"
```

The login node is python **3.11.13**, so both evaluate FALSE and pip omits them. **`pip
download --python-version 3.12` does NOT fix this** — it steers wheel-*tag* selection, not
*marker* evaluation. On Kaggle (3.12) you then get:

```
ERROR: No matching distribution found for six>=1.16.0
```

…which reads as "that version doesn't exist". It exists. And fixing only `six` fails again
immediately, because `setuptools` had resolved to **84.0.0** through another package's
unconstrained requirement — violating vLLM's own `<81.0.0`, whose pin was marker-excluded.

**Fix:** resolve with a real python 3.12 interpreter. `scripts/build_vllm_wheelhouse.sh`
does this.

### Trap 2 — a narrow `--platform` list reports as a missing **version**

Three separate packages ship tags outside the obvious set:

| package | required tag |
|---|---|
| `llguidance` | `manylinux_2_31` |
| `nvidia-cudnn-cu13` | `manylinux_2_27` |
| `nvidia-nccl-cu13` | `manylinux_2_18` |

Each omission surfaced as `No matching distribution found for <pkg>==<ver>`. **Enumerate the
whole range with `seq 17 40`; do not hand-list it** — a hand-written list skipped 2_18..2_23
and reproduced the failure a third time. You also cannot drop `--platform` and resolve
natively: this node is glibc 2.17 and would pick wheels far too old for Kaggle.

### Trap 3 — the FP4 GEMM goes through FlashInfer, whatever the attention backend

This is the one that looks most wrong. **`--attention-backend TRITON_ATTN` does not keep
FlashInfer out of the picture.** vLLM 0.27.1 dispatches the NVFP4 dense GEMM as:

```
vllm/utils/flashinfer.py:584 flashinfer_mm_fp4
  -> flashinfer mm_fp4, backend "cutlass"
  -> get_gemm_sm120_module_cutlass_fp4()   # JIT-compiles at engine init
```

Attention and the NVFP4 GEMM are **separate axes**. Triton is the attention backend; it never
touches this path. (SGLang behaves the same way — `fp4_utils.py` binds `fp4_quantize` to
flashinfer and `initialize_fp4_gemm_config()` resolves to `flashinfer_cutlass` on sm_120.)

Consequence: **the FP4 JIT toolchain must work, or the engine core dies at startup.**

### Trap 4 — no nvcc ⇒ an empty arch list ⇒ a error naming the wrong thing

```
RuntimeError: No supported CUDA architectures found for major versions [12].
```

This does *not* mean sm_120 is unsupported. FlashInfer's `_normalize_cuda_arch` **raises**
`"SM 12.x requires CUDA >= 12.9"` when it cannot determine the CUDA version, and the
enclosing `except Exception` in `compilation_context.py` swallows that raise and only logs a
warning — leaving `TARGET_CUDA_ARCHS` empty. The real cause is that the environment had no
`nvcc` on `PATH` and no `CUDA_HOME`.

**Fix:** set `CUDA_HOME`, put `nvidia/cu13/bin` on `PATH`, and set
`FLASHINFER_CUDA_ARCH_LIST=12.0f` to short-circuit both the device probe and the normaliser.
(`12.0f` is exactly what the normaliser emits for SM 12.0 under CUDA ≥ 12.9;
`compilation_context.py:96-97` respects a supplied suffix as-is.)

### Trap 5 — pip's CUDA wheels ship **no unversioned soname**, so the JIT link fails

```
/usr/bin/ld: cannot find -lcudart
ninja: build stopped: subcommand failed.
```

`nvidia-cuda-runtime` ships `libcudart.so.13` and **not** `libcudart.so`, which is what
`ld -lcudart` resolves. Two distinct things are needed and they are easy to conflate:

- **`LD_LIBRARY_PATH`** — run time, for loading
- **`LIBRARY_PATH`** — **link** time, for `ld`. This is the one that was missing.

`/usr/local/nvidia/lib64` carries `libcuda` but **not** `libcudart`, so having it alone is
not enough.

**Fix:** create the unversioned aliases (42 of them in practice) and put the wheel lib dirs
on `LIBRARY_PATH`. This is the same soname repair the SGLang path already did for its blob.

### Trap 6 — the MTP race patch moved file between vLLM releases

`vllm#40756`: the sequential MTP draft loop writes CUDA-graph input buffers with no barrier,
giving a sticky device-side assert (error 710). It previously cost a 10.8 h run.

- vLLM **0.19.0**: the buffer writes are in `vllm/v1/spec_decode/eagle.py`
- vLLM **0.27.1**: `EagleProposer` is a 22-line subclass; the loop moved to
  **`vllm/v1/spec_decode/llm_base_proposer.py`** (anchor at lines 713-714, no `synchronize()`
  anywhere in its 1,886 lines)

Patching only `eagle.py` reports "anchor absent" and **protects nothing**. Target both files,
and make an unrecognised layout report `MTP UNPROTECTED` rather than something reassuring.

---

## 3. Build the wheelhouse

`ARC3-Inference/scripts/build_vllm_wheelhouse.sh`

```bash
PY=<a real python 3.12>/bin/python          # NOT the 3.11 login-node python
PLATS=(); for v in $(seq 17 40); do PLATS+=(--platform "manylinux_2_${v}_x86_64"); done
PLATS+=(--platform manylinux2014_x86_64 --platform linux_x86_64 --platform any)

"$PY" -m pip download "vllm==0.27.1" -d "$D" --only-binary=:all: "${PLATS[@]}"
```

A 3.12 interpreter can be made from any 3.12 on the box: `python3.12 -m venv /tmp/py312`
(the repo's `.venv-sglang` is 3.12 but has no pip; `-m venv` from it works).

Then generate the lock from the filenames:

```python
rows = sorted({'{}=={}'.format(*Path(f).stem.split('-')[:2]) for f in glob('*.whl')})
Path('requirements.lock').write_text('\n'.join(rows) + '\n')
```

Result: **195 wheels, 3.5 GB**, `vllm==0.27.1 torch==2.13.0 flashinfer-python==0.6.16.post3
six==1.17.0 setuptools==80.10.2`.

Upload as a Kaggle dataset. **The Kaggle CLI under-reports large uploads** — it printed 4
"Upload successful" lines for 195 files, and the dataset size column read `0` for minutes
afterwards. Verify server-side with
`kaggle datasets files <ref> --page-size 500 | grep -c '\.whl'` before concluding anything.

---

## 4. Verify before spending a GPU minute

Both live in `ARC3-Inference/scripts/`, beside `kaggle_nvfp4_vendor_ab_probe.py` (the
kernel that produced every number above). Run them after any wheelhouse change.

**`scripts/verify_vllm_wheelhouse.sh`** — resolves the lock exactly as Kaggle will. Both conditions are
required or it proves nothing:

- a **python 3.12** interpreter → markers evaluate as Kaggle evaluates them
- the **`--platform` overrides** → otherwise this glibc-2.17 node rejects every
  `manylinux_2_28` wheel as a phantom "no matching version"

Success looks like `Would install ... vllm-0.27.1 ... six-1.17.0 setuptools-80.10.2`.

**`scripts/verify_vllm_flags.py`** — an unrecognised CLI flag is `SystemExit(2)` with no server and
no useful log. Confirms all 16 flags resolve in 0.27.1 and that `TRITON_ATTN` / `fp8_e4m3`
are still valid *values*, not just accepted names.

---

## 5. Install location

**Install to `/tmp`, never `/kaggle/working`.**

The tree expands to ~10 GB of torch + CUDA. Under `/kaggle/working` it counts against the
output budget and ships in `kaggle kernels output` — which made run 1's results effectively
unreachable behind a multi-GB download, and is why that run's failure cause was never
recovered. With `/tmp`, run 2's output was **57 KB including the kernel log**.

```python
SITE_PACKAGES = Path('/tmp/vllm-site-packages')
```

Install command (`--target` gives a FLAT layout — see §6 and Trap 6):

```
pip install --no-index --find-links <wheelhouse> --requirement <wheelhouse>/requirements.lock
            --target /tmp/vllm-site-packages --upgrade --ignore-installed
            --only-binary :all: --no-compile --disable-pip-version-check --no-warn-conflicts
```

---

## 6. Post-install repair (required — the JIT fails without it)

```python
# 1. unversioned soname aliases, or the FP4 JIT link dies on -lcudart
for lib in sorted((SITE_PACKAGES / 'nvidia').glob('*/lib/*.so.*')):
    alias = lib.parent / (lib.name.split('.so.')[0] + '.so')
    if not alias.exists():
        alias.symlink_to(lib.name)          # ~42 aliases

# 2. exec bit on the CUDA binaries; a lost bit is a silent build failure
for b in (SITE_PACKAGES / 'nvidia/cu13/bin').glob('*'):
    b.chmod(b.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

# 3. MTP race patch — BOTH targets, llm_base_proposer.py FIRST
MTP_PATCH_TARGETS = ('vllm/v1/spec_decode/llm_base_proposer.py',
                     'vllm/v1/spec_decode/eagle.py')
# insert after the two buffer writes:
#     self.input_ids[:batch_size] = input_ids
#     self.hidden_states[:batch_size] = hidden_states
#     torch.accelerator.current_stream().synchronize()   <-- the fix
```

`scripts/patch_vllm_mtp_race.py` does #3 standalone and needs `--site-packages` for the flat
`pip --target` layout: its default venv globs (`lib/python3.*/site-packages`) match **nothing**
there and it would patch no files while reporting success.

---

## 7. The environment

Every entry below is load-bearing. Removing any one reproduces a documented failure.

```python
def vllm_env() -> dict:
    e = os.environ.copy()
    sp = str(SITE_PACKAGES)
    e['PYTHONPATH'] = sp                                   # flat --target layout

    # run time
    libs  = [f'{sp}/torch/lib']
    libs += sorted(str(x) for x in (SITE_PACKAGES / 'nvidia').glob('*/lib'))
    libs += ['/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu']
    e['LD_LIBRARY_PATH'] = ':'.join(libs + [e.get('LD_LIBRARY_PATH', '')]).strip(':')

    # FP4 JIT toolchain  (Traps 3 + 4)
    cu13 = SITE_PACKAGES / 'nvidia/cu13'
    e['CUDA_HOME'] = str(cu13)
    e['PATH']      = f'{cu13}/bin:' + e.get('PATH', '')
    e.setdefault('FLASHINFER_CUDA_ARCH_LIST', '12.0f')
    e['NVCC_APPEND_FLAGS'] = (e.get('NVCC_APPEND_FLAGS', '') +
                              ' -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK').strip()

    # LINK time — distinct from LD_LIBRARY_PATH  (Trap 5)
    linkdirs = sorted(str(x) for x in (SITE_PACKAGES / 'nvidia').glob('*/lib'))
    e['LIBRARY_PATH'] = ':'.join(linkdirs + ['/usr/local/nvidia/lib64',
                                             e.get('LIBRARY_PATH', '')]).strip(':')

    e.update({'USE_TF': '0', 'TRANSFORMERS_NO_TF': '1',
              'TRANSFORMERS_NO_TORCHVISION': '1', 'VLLM_NO_USAGE_STATS': '1',
              'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    return e
```

`NVCC_APPEND_FLAGS`: nvcc reports 13.4 while `cuda_runtime_api.h` says 13000; CCCL
hard-errors on the skew and kills the JIT. SGLang needed the same workaround.

> **When editing these two builders, beware:** `sglang_env()` and `vllm_env()` share several
> identical lines (`NVCC_APPEND_FLAGS` among them). A `str.replace(old, new, 1)` over the
> whole file hits `sglang_env` first. That mistake pointed the *SGLang* arm at
> `/tmp/vllm-site-packages` and would likely have gone unnoticed, because
> `/usr/local/nvidia/lib64` survived in the string and the arm kept working. Edit each
> function's own body slice and assert which function the change landed in.

---

## 8. Server flags

```
python -m vllm.entrypoints.openai.api_server
  --model <checkpoint dir>
  --served-model-name Qwen/Qwen3.8-27B-NVFP4
  --host 127.0.0.1 --port 1234
  --tensor-parallel-size 1
  --max-model-len 32768
  --attention-backend TRITON_ATTN            # 2.70x over FA2, 1.36x over FLASHINFER on sm_120
  --kv-cache-dtype fp8_e4m3                  # 2x KV capacity
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --generation-config vllm                   # ignore the checkpoint's generation_config.json
  --enable-prefix-caching
  --trust-remote-code
  --default-chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "xhigh"}'
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
  --enable-prompt-tokens-details             # or the image smoke test cannot see the image
```

**The probe never sent an image.** Its sanity battery and C=16 load were text-only, so the
multimodal path is the one thing the A/B could not validate -- and it is where kernel
`taaf-qwen38-nvfp4-vllm` v2 died. The engine was sighted (vision encoder attention
initialised, encoder cache profiled with image items, and the model answered
"Checkerboard." to a checkerboard) but `usage.prompt_tokens_details` came back `None`, and the
smoke gate read `image_tokens` from it. Two separate facts:

- `prompt_tokens_details` is **off by default in every vLLM release** (`cli_args.py`
  `enable_prompt_tokens_details: bool = False` in both 0.19.0 and 0.27.1).
- the per-modality count is `multimodal_tokens: {"image": N}` in 0.27.1; 0.19.0 reported no
  per-modality count at all. `image_tokens` is **SGLang's** field name -- the gate had only
  ever run against SGLang.

The launcher now passes the flag, reads both shapes, and also sends the same prompt without
the image part: a positive `prompt_tokens` delta proves the image was consumed whatever the
details field is called next release.

Confirm in the engine log that all of these actually took effect:

```
Resolved architecture: Qwen3_5MTP
SpeculativeConfig(method='mtp', num_spec_tokens=3)
Loading drafter model...
Detected MTP model. Sharing target model embedding weights with the draft model.
quantization=compressed-tensors            <-- NVFP4 recognised
kv_cache_dtype=torch.float8_e4m3fn       <-- the RESOLVED torch dtype, not the CLI spelling
```

That last line is a trap of its own: the CLI takes `fp8_e4m3` but the log reports
`torch.float8_e4m3fn`. Kernel `taaf-qwen38-nvfp4-vllm` v1 asserted the two were equal and
shut down a healthy server 7.5 minutes into startup. Normalise before comparing, and warn
rather than assert -- the smoke test is the gate.

**`quantization=compressed-tensors` is the check that matters.** Do *not* reuse the old
"`quantization: null` means broken" heuristic — that was wrong. SGLang reported
`quantization: null` while generating perfectly coherent 16k text; the field reports the
server *arg*, which we never set.

---

## 9. Checkpoint: use **unsloth**, not RadixArk, with vLLM

The two NVFP4 checkpoints are **not interchangeable**, and the difference is visible in
`config.json` without loading anything:

| | `unsloth/Qwen3.8-27B-NVFP4` | `RadixArk/Qwen3.8-27B-NVFP4` |
|---|---|---|
| `lm_head` | **FP8** (`group_0`, 8-bit) | **NVFP4** (`group_1`, 4-bit) |
| `lm_head` tensors | `weight`, `weight_scale` | + `weight_scale_2`, `input_scale` |
| tensors / mtp | 1968 / 15 | 2194 / 15 |
| engine | **vLLM** | SGLang |

`weight_scale_2` is the NVFP4 double-scale; its absence means the head is FP8. Fastest
possible vendor check: fetch `model.safetensors.index.json` and grep the `lm_head.*` keys.

Unsloth states it outright: NVFP4 is *"vLLM only for now (SGLang is not supported)"* because
*"SGLang cannot load the FP8 lm_head"*. Pairing SGLang with unsloth is what produced the
original degeneration — 5-gram diversity 0.109-0.385 against 0.873-0.984 for FP8.

Kaggle datasets: `jonathanwang2022/qwen38-27b-nvfp4-unsloth` (use this),
`jonathanwang2022/qwen38-27b-nvfp4-radixark` (SGLang only),
`jonathanwang2022/vllm-0271-wheelhouse-sm120`.

---

## 10. Measuring it afterwards — two instrumentation bugs to avoid

Both produced confidently wrong numbers in this campaign.

1. **Never average an acceptance *rate* as if it were a *length*.** vLLM logs the rate in two
   units in the same file (`acceptance rate: 46.7` as a percent and `acceptance rate: 0.667`
   as a fraction). Averaging across them yielded `accept_len_mean 25.98` — impossible when
   `num_spec_tokens=3` caps the length at 4. Read the length only:
   SGLang logs `accept len:`, vLLM logs `acceptance length:`.
2. **Do not tokenize n-grams on whitespace alone.** A comma-separated answer with no spaces
   scores 2 "words", 0 five-grams, diversity 0.000 — indistinguishable from total collapse on
   a perfectly correct answer. Split on `\w+`. This understated a healthy arm as 0.805 when
   it was really 0.966.

And one harness rule earned the hard way: **a health gate must not be able to withhold the
measurement.** Run 1 produced no throughput number at all because a gate failed on a probe
bug (a tool-call check given 256 tokens while the server ran `reasoning_effort: xhigh`, so
reasoning consumed the budget before any call was emitted). Record health, attach a warning
to a suspect number, but always emit the number.

---

## 11. Open items

- **Arm C (vLLM × RadixArk)** — separates engine from checkpoint. One extra server launch.
- **Low-concurrency behaviour** — SGLang may well win at C=1..4; untested.
- **`CONCURRENT_JOBS` / `MAX_RUNTIME_MINUTES`** — `0820_report.md` set 16 / 71 against
  *SGLang* throughput. vLLM is ~2.5x faster per request at C=16, so that pairing should be
  re-derived rather than inherited.
