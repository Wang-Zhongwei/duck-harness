#!/usr/bin/env python
"""NVFP4 vendor A/B: does the checkpoint vendor, not NVFP4/SGLang/sm_120, explain the
degeneration we saw?

See CAMPAIGN.md next to this file. In one line: unsloth quantizes lm_head to FP8 and
their docs say SGLang cannot load it; RadixArk quantizes lm_head to NVFP4 and is what
SGLang's own cookbook pins for hw=rtx6000. lm_head is the projection to vocab logits,
so a broken one degrades token SELECTION while everything upstream still works -- which
is exactly what we saw (LaTeX scaffolding survived, the digits inside it did not).

  arm A  vLLM   x unsloth    (unsloth's supported pairing)
  arm B  SGLang x RadixArk   (SGLang's supported pairing)
  arm C  vLLM   x RadixArk   (completes the 2x2; the 4th cell is already measured broken)

Every arm gets the SAME knobs: attention=triton, KV=fp8_e4m3, MTP=3, ctx 32768.
Arms run interleaved A,B,C,B,A so drift and warm-up cannot masquerade as an arm effect.

Design rules paid for in earlier runs:
  * A short sample cannot judge coherence. 17x3 -> "51" passed 6/6 on a server that
    looped for 4096 tokens on any prompt inviting prose. Health is scored over ~16k
    tokens or not at all.
  * Throughput measured on degenerate output is not throughput. An arm that fails the
    health gate gets NO tok/s number reported. accept_rate 1.00 on a looping arm is a
    consequence of repetitive text being trivially predictable, not a win.
  * A server can be perfectly healthy and still score 0, because the agent acts ONLY
    through tool calls. Tool-call validity is a gate, not a nice-to-have.
  * "quantization": null on a quantized checkpoint means the loader ignored the quant
    config. That is a failure, not a detail.
"""

import json, os, re, shutil, signal, stat, subprocess, sys, tarfile, threading, time, traceback
import urllib.request
from collections import Counter
from pathlib import Path

WORKING = Path('/kaggle/working')
RESULTS = WORKING / 'vendor_ab_results.json'
rec = {'arms': [], 'phase0': {}, 'agreement': {}}
def save(): RESULTS.write_text(json.dumps(rec, indent=2), encoding='utf-8')

# ---------------------------------------------------------------- configuration

HOST, PORT = '127.0.0.1', 1234
BASE = f'http://{HOST}:{PORT}/v1'
SERVED = 'Qwen/Qwen3.8-27B-NVFP4'

CONTEXT_LEN = 32768         # matched across arms
HEALTH_MAX_TOKENS = 16384   # the long generation that actually exposes looping
LOAD_CONCURRENCY = 16
LOAD_MAX_TOKENS = 4096      # bounded for wall-clock; identical across arms
SPEC_STEPS = 3              # MTP=3
ATTENTION = 'triton'
KV_DTYPE = 'fp8_e4m3'
TEMPERATURE, TOP_P, TOP_K = 1.0, 0.95, 20
START_TIMEOUT = 2400

SGLANG_TREE = Path('/tmp/sglang-sp')
SITE_PACKAGES = Path('/tmp/vllm-site-packages')
SERVER_LOG = WORKING / 'inference-server.log'
SERVER_PID = WORKING / 'inference-server.pid'
SERVER_PGID = WORKING / 'inference-server.pgid'

# vLLM 0.27.1 / torch 2.13.0 / flashinfer 0.6.16.post3, built 2026-08-21 for Kaggle
# py3.12 x86_64. Replaces driessmit1/arc3-vllm-h100-wheelhouse-v3 (vLLM 0.19.0, torch
# 2.10.0) -- that dataset is named for H100/sm_90 and arm A died at vLLM engine-core init
# on the Blackwell GPU in run 1. 0.27.1 is the version upstream validated for NVFP4 on
# sm_120 (Qwen3.8-27B-NVFP4, 4096 requests @ C=256, zero errors).
WHEELHOUSE_REF = ('jonathanwang2022', 'vllm-0271-wheelhouse-sm120')
SGLANG_REF = ('jonathanwang2022', 'sglang-0517-sp-blob')
UNSLOTH_REF = ('jonathanwang2022', 'qwen38-27b-nvfp4-unsloth')
RADIXARK_REF = ('jonathanwang2022', 'qwen38-27b-nvfp4-radixark')
DSPARK_REF = ('jonathanwang2022', 'qwen38-27b-dspark-radixark')

# Health gates. Reference points from kernel sglang-quality-matrix on this same GPU:
# healthy FP8 scored 0.873-0.984 mean 5-gram diversity, broken NVFP4 scored 0.109-0.385.
# 0.70 sits clear of both, so the verdict does not hinge on where exactly it is drawn.
GATE_DIVERSITY = 0.70
GATE_MAX_REPEAT = 20
GATE_DIGIT_RETENTION = 0.95
GATE_STOP_FRAC = 0.5

# v2 is throughput-focused: run 1 spent ~7.5 min per arm on the 16k battery and then
# discarded the throughput result anyway. One short sanity check is kept -- a throughput
# number measured on degenerate output is worthless, and the check costs well under a
# minute against a ~10 min arm. Set VENDOR_AB_FULL_HEALTH=1 for the full battery.
FOCUS_THROUGHPUT = os.getenv('VENDOR_AB_FULL_HEALTH', '') not in ('1', 'true', 'yes')
QUICK_MAX_TOKENS = 2048


def resolve_dataset(owner: str, slug: str) -> Path:
    raw = os.getenv('TAAF_KAGGLE_INPUT_PATHS', '').strip()
    if raw:
        mapped = json.loads(raw).get(f'{owner}/{slug}')
        if mapped:
            return Path(mapped)
    # Kaggle mounts datasets at /kaggle/input/datasets/<owner>/<slug>, NOT /kaggle/input/<slug>.
    for p in (Path('/kaggle/input/datasets') / owner / slug, Path('/kaggle/input') / slug):
        if p.exists():
            return p
    return Path('/kaggle/input') / slug


def resolve_model_dir(root: Path) -> Path:
    if (root / 'config.json').exists():
        return root
    for cfg in sorted(root.rglob('config.json')):
        if list(cfg.parent.glob('*.safetensors')):
            return cfg.parent
    return root


# ---------------------------------------------------------------- preflight

def assert_gpu():
    assert shutil.which('nvidia-smi'), 'nvidia-smi missing'
    out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True)
    names = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    assert names, 'no CUDA GPU'
    # NVFP4 needs Blackwell. On anything else this whole campaign is meaningless, and
    # failing here is much cheaper than failing after two 8-minute model loads.
    assert any('rtx pro 6000' in n.lower() or 'b200' in n.lower() for n in names), \
        f'NVFP4 requires Blackwell; found {names}'
    print(f'GPU ok: {names}', flush=True)


def checkpoint_fingerprint(model_dir: Path) -> dict:
    """Record which group lm_head landed in. This is the whole hypothesis, so read it
    off the checkpoint rather than trusting the dataset name."""
    fp = {'path': str(model_dir)}
    try:
        cfg = json.loads((model_dir / 'config.json').read_text())
        qc = cfg.get('quantization_config') or {}
        fp['format'] = qc.get('format')
        for gname, g in (qc.get('config_groups') or {}).items():
            targets = g.get('targets') or []
            if any('lm_head' in str(t) for t in targets):
                fp['lm_head_group'] = gname
                fp['lm_head_bits'] = (g.get('weights') or {}).get('num_bits')
                fp['lm_head_format'] = g.get('format')
    except Exception as exc:
        fp['config_error'] = repr(exc)[:200]
    try:
        idx = json.loads((model_dir / 'model.safetensors.index.json').read_text())
        wm = idx.get('weight_map', {})
        fp['n_tensors'] = len(wm)
        fp['lm_head_tensors'] = sorted(k for k in wm if 'lm_head' in k)
        # weight_scale_2 is the NVFP4 double-scale signature; its absence means FP8.
        fp['lm_head_is_nvfp4'] = any('weight_scale_2' in k for k in fp['lm_head_tensors'])
        fp['n_mtp_tensors'] = sum(1 for k in wm if k.startswith('mtp'))
    except Exception as exc:
        fp['index_error'] = repr(exc)[:200]
    return fp


# ---------------------------------------------------------------- runtimes

def prepare_sglang_tree() -> Path:
    assert sys.version_info[:2] == (3, 12), (
        f'blob is a cp312 tree, interpreter is {sys.version_info.major}.{sys.version_info.minor}; '
        'the Kaggle base image changed -- rebuild the blob')
    root = resolve_dataset(*SGLANG_REF)
    blobs = sorted((p for p in root.rglob('*') if p.is_file() and p.stat().st_size > 1_000_000_000),
                   key=lambda p: p.stat().st_size, reverse=True)
    assert blobs, f'no SGLang blob under {root}'
    hits = sorted(SGLANG_TREE.rglob('sglang/srt')) if SGLANG_TREE.exists() else []
    if hits and (SGLANG_TREE / '.unpacked').exists():
        sp = hits[0].parent.parent
        print(f'reusing SGLang tree {sp}', flush=True)
    else:
        free = shutil.disk_usage('/tmp').free / 2 ** 30
        assert free >= 8.0, f'only {free:.1f} GiB free on /tmp, need >= 8'
        shutil.rmtree(SGLANG_TREE, ignore_errors=True)
        SGLANG_TREE.mkdir(parents=True, exist_ok=True)
        t = time.time()
        with tarfile.open(blobs[0], 'r:*') as a:
            a.extractall(SGLANG_TREE, filter='tar')
        print(f'untar {time.time() - t:.0f}s', flush=True)
        hits = sorted(SGLANG_TREE.rglob('sglang/srt'))
        assert hits, 'blob extracted but no sglang/srt'
        sp = hits[0].parent.parent
        (SGLANG_TREE / '.unpacked').write_text('ok')
    # sglang 0.5.17 ships nvidia/cu13 with lib/ but no lib64/, and only versioned
    # sonames, so every JIT link dies with 'ld: cannot find -lcudart'.
    cu13 = sp / 'nvidia' / 'cu13'
    lib = cu13 / 'lib'
    assert lib.is_dir(), f'{lib} missing; JIT links will all fail'
    if not (cu13 / 'lib64').exists():
        (cu13 / 'lib64').symlink_to('lib')
    for so in sorted(lib.glob('*.so.*')):
        bare = lib / (so.name.split('.so.')[0] + '.so')
        if not bare.exists():
            try:
                bare.symlink_to(so.name)
            except OSError:
                pass
    for b in list((sp / 'triton/backends/nvidia/bin').glob('*')) + list((cu13 / 'bin').glob('*')):
        try:
            b.chmod(b.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
    need = {'torch ext': sp / 'torch/_C.cpython-312-x86_64-linux-gnu.so',
            'sglang': sp / 'sglang/srt', 'libnvrtc': cu13 / 'lib/libnvrtc.so.13',
            'cutlass DSL': sp / 'nvidia_cutlass_dsl/dsl_packages/cutlass',
            'nvcc': cu13 / 'bin/nvcc', 'ptxas': sp / 'triton/backends/nvidia/bin/ptxas'}
    missing = [f'{k} ({v})' for k, v in need.items() if not v.exists()]
    assert not missing, 'incomplete SGLang tree: ' + '; '.join(missing)
    print(f'SGLang preflight ok {sp}', flush=True)
    return sp


def sglang_env(sp: Path) -> dict:
    s = str(sp)
    libs = [f'{s}/torch/lib', f'{s}/nvidia/cu13/lib', f'{s}/nvidia/cudnn/lib',
            f'{s}/nvidia/nccl/lib', f'{s}/nvidia/cusparselt/lib', f'{s}/nvidia/nvshmem/lib',
            '/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu']
    libs += sorted(str(p) for p in (sp / 'nvidia').glob('*/lib'))
    e = os.environ.copy()
    # .pth files do not execute for PYTHONPATH entries and the GDN layers import
    # cutlass.cute, so the CuTe DSL dir must be added by hand.
    e['PYTHONPATH'] = f'{s}:{s}/nvidia_cutlass_dsl/dsl_packages'
    e['LD_LIBRARY_PATH'] = ':'.join(libs + [e.get('LD_LIBRARY_PATH', '')]).strip(':')
    e['LIBRARY_PATH'] = ':'.join(['/usr/local/nvidia/lib64', e.get('LIBRARY_PATH', '')]).strip(':')
    e['CUDA_HOME'] = f'{s}/nvidia/cu13'
    e['PATH'] = f'{s}/nvidia/cu13/bin:' + e.get('PATH', '')
    e['PYTHONNOUSERSITE'] = '1'
    e['TRITON_CACHE_DIR'] = '/tmp/triton'
    e['HF_HUB_OFFLINE'] = e['TRANSFORMERS_OFFLINE'] = '1'
    # nvcc reports 13.4 while cuda_runtime_api.h says 13000; CCCL hard-errors on the skew.
    e['NVCC_APPEND_FLAGS'] = (e.get('NVCC_APPEND_FLAGS', '') +
                              ' -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK').strip()
    e.pop('PYTHONHOME', None)
    return e


def vllm_env() -> dict:
    e = os.environ.copy()
    prev = e.get('PYTHONPATH', '')
    e['PYTHONPATH'] = str(SITE_PACKAGES) if not prev else f'{SITE_PACKAGES}{os.pathsep}{prev}'
    sp = str(SITE_PACKAGES)
    libs = [f'{sp}/torch/lib'] + sorted(str(x) for x in (SITE_PACKAGES / 'nvidia').glob('*/lib'))
    libs += ['/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu']
    e['LD_LIBRARY_PATH'] = ':'.join(libs + [e.get('LD_LIBRARY_PATH', '')]).strip(':')

    # --- arm A's run-3 death, fixed ------------------------------------------------
    # vLLM 0.27.1 routes the NVFP4 dense GEMM through FlashInfer on sm_120:
    #   vllm/utils/flashinfer.py flashinfer_mm_fp4 -> flashinfer mm_fp4 backend "cutlass"
    #   -> get_gemm_sm120_module_cutlass_fp4 -> JIT compile.
    # That JIT needs nvcc. sglang_env() supplied CUDA_HOME/PATH; vllm_env() did not, so
    # FlashInfer could not resolve the CUDA version, _normalize_cuda_arch raised
    # "SM 12.x requires CUDA >= 12.9", the enclosing `except Exception` swallowed it, and
    # TARGET_CUDA_ARCHS was left EMPTY -> "No supported CUDA architectures found for major
    # versions [12]" and the engine core died. Both reps, run 3.
    cu13 = SITE_PACKAGES / 'nvidia/cu13'
    e['CUDA_HOME'] = str(cu13)
    e['PATH'] = f'{cu13}/bin:' + e.get('PATH', '')
    # Belt and braces: an explicit entry short-circuits BOTH the device probe and
    # _normalize_cuda_arch (compilation_context.py:96-97 respects a suffix as given).
    # "0f" is what normalize itself produces for SM 12.0 under CUDA >= 12.9.
    e.setdefault('FLASHINFER_CUDA_ARCH_LIST', '12.0f')
    # Same nvcc/CCCL header skew sglang_env works around (nvcc says 13.4, the runtime
    # header says 13000); CCCL hard-errors on it and would kill the JIT.
    e['NVCC_APPEND_FLAGS'] = (e.get('NVCC_APPEND_FLAGS', '') +
                              ' -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK').strip()
    # The FP4 JIT's LINK step needs -lcudart, which is resolved from LIBRARY_PATH (link
    # time), not LD_LIBRARY_PATH (run time). Run 4 got past the arch gate and then died at
    # `/usr/bin/ld: cannot find -lcudart` because LIBRARY_PATH held only
    # /usr/local/nvidia/lib64, which carries libcuda but not libcudart.
    linkdirs = sorted(str(x) for x in (SITE_PACKAGES / 'nvidia').glob('*/lib'))
    e['LIBRARY_PATH'] = ':'.join(linkdirs + ['/usr/local/nvidia/lib64',
                                             e.get('LIBRARY_PATH', '')]).strip(':')
    e.update({'USE_TF': '0', 'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_TORCHVISION': '1',
              'VLLM_NO_USAGE_STATS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
              # FlashInfer JIT dies with 'ld: cannot find -lcuda' without this. We run
              # TRITON_ATTN, which needs no JIT, but vLLM may still probe FlashInfer.
              # LIBRARY_PATH is set above with the wheel CUDA lib dirs; do not clobber it.
              })
    return e


def install_vllm() -> None:
    wh = resolve_dataset(*WHEELHOUSE_REF)
    reqs = wh / 'requirements.lock'
    assert reqs.exists(), f'missing wheelhouse lock {reqs}'
    if (SITE_PACKAGES / 'vllm' / '__init__.py').exists():
        print('reusing vLLM install', flush=True)
    else:
        SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                        '--find-links', str(wh), '--requirement', str(reqs),
                        '--target', str(SITE_PACKAGES), '--upgrade', '--ignore-installed',
                        '--only-binary', ':all:', '--no-compile',
                        '--disable-pip-version-check', '--no-warn-conflicts'], check=True)
    # FlashInfer's FP4 JIT shells out to these; a lost exec bit is a silent build failure.
    for b in (SITE_PACKAGES / 'nvidia/cu13/bin').glob('*'):
        try:
            b.chmod(b.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
    # pip's CUDA wheels ship ONLY versioned sonames (libcudart.so.13). `ld -lcudart`
    # resolves `libcudart.so`, so without these aliases the FP4 JIT link fails. This is
    # the same repair prepare_sglang_tree() already does for the SGLang blob.
    aliased = 0
    for lib in sorted((SITE_PACKAGES / 'nvidia').glob('*/lib/*.so.*')):
        alias = lib.parent / (lib.name.split('.so.')[0] + '.so')
        if not alias.exists():
            try:
                alias.symlink_to(lib.name)
                aliased += 1
            except OSError:
                pass
    rec['soname_aliases'] = aliased
    cudart = SITE_PACKAGES / 'nvidia/cu13/lib/libcudart.so'
    rec['libcudart_so'] = cudart.exists()
    print(f'soname aliases created: {aliased}; libcudart.so present: {cudart.exists()}',
          flush=True)

    nvcc = SITE_PACKAGES / 'nvidia/cu13/bin/nvcc'
    rec['nvcc_present'] = nvcc.exists()
    print('nvcc for FlashInfer FP4 JIT:', nvcc if nvcc.exists() else 'MISSING', flush=True)

    ver = subprocess.run([sys.executable, '-c', 'import vllm;print(vllm.__version__)'],
                         env=vllm_env(), capture_output=True, text=True)
    rec['vllm_version'] = (ver.stdout or ver.stderr).strip()[:80]
    print('vLLM version:', rec['vllm_version'], flush=True)
    apply_mtp_patch()


# vLLM moved the draft loop between releases. 0.19.0 had the buffer writes in eagle.py;
# 0.27.1 refactored EagleProposer down to a 22-line subclass and the loop now lives in
# llm_base_proposer.py (verified by reading the 0.27.1 wheel: eagle.py has neither the fix
# nor the anchor, llm_base_proposer.py has the anchor at lines 713-714 and no synchronize()
# anywhere). Targeting only eagle.py would have reported ANCHOR NOT FOUND and left MTP=3
# unprotected -- the exact failure that cost a 10.8h run.
MTP_PATCH_TARGETS = ('vllm/v1/spec_decode/llm_base_proposer.py',
                     'vllm/v1/spec_decode/eagle.py')


def apply_mtp_patch() -> None:
    """MTP>=2 dies with a sticky CUDA device-side assert (error 710) without this.
    vllm#40756: the sequential MTP draft loop writes the CUDA-graph input buffers with no
    barrier before the draft model reads them."""
    marker = 'torch.accelerator.current_stream().synchronize()'
    anchor = ('            self.input_ids[:batch_size] = input_ids\n'
              '            self.hidden_states[:batch_size] = hidden_states\n')
    results, patched_any, seen_any = [], False, False
    for rel in MTP_PATCH_TARGETS:
        f = SITE_PACKAGES / rel
        if not f.exists():
            results.append(f'{rel}: absent')
            continue
        seen_any = True
        text = f.read_text()
        if marker in text:
            results.append(f'{rel}: already patched')
            patched_any = True
        elif anchor in text:
            f.write_text(text.replace(anchor, anchor + '            ' + marker + '\n', 1))
            results.append(f'{rel}: APPLIED')
            patched_any = True
        else:
            results.append(f'{rel}: anchor absent')
    if not seen_any:
        rec['mtp_patch'] = 'NO TARGET FILE FOUND -- MTP unprotected'
    elif patched_any:
        rec['mtp_patch'] = '; '.join(results)
    else:
        # Refuse to guess. A wrong patch is worse than no patch -- but say so loudly,
        # because run 1 recorded a reassuring "applied" that meant nothing.
        rec['mtp_patch'] = 'ANCHOR NOT FOUND IN ANY TARGET -- MTP UNPROTECTED: ' + \
                           '; '.join(results)
    print('MTP race patch:', rec['mtp_patch'], flush=True)


# ---------------------------------------------------------------- arm lifecycle

def stop_arm():
    # SGLang forks scheduler/detokenizer children that survive a SIGTERM to the launcher
    # and keep the 96 GB. Signal the whole process group or the next arm OOMs.
    if SERVER_PGID.exists():
        try:
            os.killpg(int(SERVER_PGID.read_text().strip()), signal.SIGTERM)
        except Exception:
            pass
    for _ in range(90):
        try:
            urllib.request.urlopen(f'http://{HOST}:{PORT}/health', timeout=2)
            time.sleep(2)
        except Exception:
            break
    time.sleep(25)


def _spawn(cmd, env) -> subprocess.Popen:
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    lf = SERVER_LOG.open('w', encoding='utf-8')
    p = subprocess.Popen(cmd, env=env, cwd='/tmp', stdout=lf, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    SERVER_PID.write_text(str(p.pid))
    SERVER_PGID.write_text(str(os.getpgid(p.pid)))
    return p


def start_sglang(model_dir: Path, sp: Path, attention: str = ATTENTION,
                 mamba: bool = False, dspark: Path | None = None) -> subprocess.Popen:
    env = sglang_env(sp)
    # Ask the installed argparse what it accepts. An unrecognised option is SystemExit(2)
    # with no server and no useful log -- on a real submission that costs the whole run.
    helped = subprocess.run([sys.executable, '-m', 'sglang.launch_server', '--help'],
                            env=env, cwd='/tmp', capture_output=True, text=True, timeout=600)
    flags = set(re.findall(r'--[a-z0-9][a-z0-9-]+', (helped.stdout or '') + (helped.stderr or '')))
    assert '--model-path' in flags, 'could not read SGLang argument table'
    cmd = [sys.executable, '-m', 'sglang.launch_server',
           '--model-path', str(model_dir), '--served-model-name', SERVED,
           '--host', HOST, '--port', str(PORT), '--tp-size', '1',
           '--context-length', str(CONTEXT_LEN), '--trust-remote-code', '--log-level', 'info',
           '--attention-backend', attention, '--kv-cache-dtype', KV_DTYPE,
           '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3',
           '--default-chat-template-kwargs', '{"preserve_thinking":true,"reasoning_effort":"xhigh"}',
           # vLLM runs --generation-config vllm (ignore the checkpoint's generation_config.json).
           # SGLang defaults the other way and would silently override our sampling params.
           '--sampling-defaults', 'openai',
           ]
    if dspark is not None:
        # The cookbook's own spec stack. Unmatched against vLLM by construction -- there is
        # no vLLM equivalent of DSPARK -- so this is a ceiling reference, not an A/B arm.
        cmd += ['--speculative-algorithm', 'DSPARK',
                '--speculative-draft-model-path', str(dspark),
                '--speculative-draft-attention-backend', attention]
    else:
        cmd += ['--speculative-algorithm', 'EAGLE',
                '--speculative-num-steps', str(SPEC_STEPS),
                '--speculative-eagle-topk', '1',
                '--speculative-num-draft-tokens', str(SPEC_STEPS + 1)]
    if mamba:
        # Hybrid-model memory/cache knobs from the cookbook. They tune the linear-attention
        # (mamba) half of Qwen3.8 and are independent of vendor, so they are off by default
        # to keep the arms matched -- but they are the vendor's tuned values, so an arm that
        # is healthy-but-slow should be re-checked with these before concluding anything.
        for flag, value in (('--mamba-full-memory-ratio', '5.61'),
                            ('--mamba-radix-cache-strategy', 'extra_buffer_lazy'),
                            ('--mamba-ssm-dtype', 'bfloat16'),
                            ('--chunked-prefill-size', '2048'),
                            ('--mem-fraction-static', '0.85')):
            if flag in flags:
                cmd += [flag, value]
            else:
                print(f'  note: {flag} not supported by this SGLang build, skipped', flush=True)
    need = {'--attention-backend', '--kv-cache-dtype', '--tool-call-parser',
            '--reasoning-parser', '--sampling-defaults', '--speculative-algorithm'}
    absent = sorted(need - flags)
    assert not absent, f'SGLang build does not report {absent}; launching would silently differ'
    print('launch sglang:', ' '.join(cmd[3:]), flush=True)
    return _spawn(cmd, env)


def start_vllm(model_dir: Path) -> subprocess.Popen:
    cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
           '--model', str(model_dir), '--served-model-name', SERVED,
           '--host', HOST, '--port', str(PORT), '--tensor-parallel-size', '1',
           '--max-model-len', str(CONTEXT_LEN),
           '--attention-backend', 'TRITON_ATTN',
           '--kv-cache-dtype', KV_DTYPE,
           '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
           '--reasoning-parser', 'qwen3', '--generation-config', 'vllm',
           '--enable-prefix-caching', '--trust-remote-code',
           '--default-chat-template-kwargs',
           '{"preserve_thinking": true, "reasoning_effort": "xhigh"}',
           '--speculative-config',
           json.dumps({'method': 'mtp', 'num_speculative_tokens': SPEC_STEPS})]
    print('launch vllm:', ' '.join(cmd[3:]), flush=True)
    return _spawn(cmd, vllm_env())


def wait_ready(proc, timeout=START_TIMEOUT) -> bool:
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f'{BASE}/models', timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def log_tail(n=40) -> str:
    if not SERVER_LOG.exists():
        return ''
    return '\n'.join(SERVER_LOG.read_text(errors='replace').splitlines()[-n:])


def first_error(max_chars=4000) -> str:
    """Return the FIRST traceback / error block in the log, not the last.

    vLLM's engine-core failure ends with "Engine core initialization failed. See root cause
    above." -- so a tail capture is guaranteed to miss the cause. On 2026-08-21 that is
    exactly what happened to arm A: it died twice and left no diagnosable evidence.
    """
    if not SERVER_LOG.exists():
        return ''
    lines = SERVER_LOG.read_text(errors='replace').splitlines()
    marks = ('Traceback (most recent call last)', 'ERROR', 'Error:', 'raise ',
             'Exception', 'CUDA error', 'not supported', 'Unsupported')
    for i, line in enumerate(lines):
        if any(m in line for m in marks):
            return '\n'.join(lines[max(0, i - 5):i + 120])[:max_chars]
    return '\n'.join(lines[:60])[:max_chars]


def gate2_loader_sanity(engine: str) -> dict:
    """An FP4/KV misconfiguration does not raise -- it produces fluent garbage. Check what
    the server actually resolved. Our failed run reported "quantization": null on a
    quantized checkpoint, i.e. the loader ignored the quant config entirely."""
    info = {}
    if engine == 'sglang':
        try:
            info = json.loads(urllib.request.urlopen(
                f'http://{HOST}:{PORT}/get_server_info', timeout=60).read())
            info = {k: info.get(k) for k in
                    ('attention_backend', 'kv_cache_dtype', 'quantization', 'context_length',
                     'max_total_num_tokens', 'max_running_requests', 'speculative_algorithm',
                     'speculative_num_steps',
                     # The NVFP4 GEMM path is independent of --attention-backend and is
                     # what actually decides NVFP4 numerics; on sm_120 it should auto-
                     # resolve to flashinfer_cutlass (fp4_utils.initialize_fp4_gemm_config).
                     'fp4_gemm_runner_backend')}
        except Exception as exc:
            info = {'error': repr(exc)[:200]}
    else:
        # vLLM has no /get_server_info; read the resolved config off the engine log.
        txt = SERVER_LOG.read_text(errors='replace') if SERVER_LOG.exists() else ''
        for key in ('quantization', 'kv_cache_dtype', 'attention_backend', 'max_model_len'):
            m = re.search(rf'{key}=([^,\s\)]+)', txt)
            info[key] = m.group(1) if m else None
        m = re.search(r'GPU KV cache size: ([\d,]+) tokens', txt)
        info['kv_cache_tokens'] = m.group(1) if m else None
    quant = str(info.get('quantization'))
    # INFORMATIONAL ONLY -- not a gate. Arm B reported quantization=null on the 2026-08-21
    # run while scoring 0.805 diversity and 1.0 digit retention over 16k-token generations,
    # so a null here does NOT mean the quant config was ignored. It reports the server ARG,
    # which we never set, not the format resolved from the checkpoint.
    info['quant_ok'] = quant not in ('None', 'null', 'none', '')
    return info


# ---------------------------------------------------------------- generation + scoring

def chat(prompt, max_tokens, temperature=TEMPERATURE, tools=None, timeout=1800,
         no_think=False):
    p = {'model': SERVED, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False,
         'temperature': temperature, 'top_p': TOP_P, 'top_k': TOP_K, 'max_tokens': max_tokens}
    if tools:
        p['tools'] = tools
        p['tool_choice'] = 'auto'
    if no_think:
        # The server is launched with reasoning_effort xhigh, which is right for the health
        # prompts and wrong for a tool-call check: xhigh spent the entire budget thinking
        # and emitted no call, failing a healthy arm. Override per-request.
        p['chat_template_kwargs'] = {'preserve_thinking': False, 'reasoning_effort': 'low'}
    req = urllib.request.Request(BASE + '/chat/completions', data=json.dumps(p).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def json_array_ok(text) -> bool:
    """Objective correctness for the structured_json item.

    Looks for the first bracketed span and requires it to parse as a 12-object array whose
    ids are 1..12 and whose "square" fields are the ids squared. A degenerate decoder can
    emit JSON-looking text that never parses, or parses but drops entries -- both fail here
    without anyone reading the output."""
    i, j = text.find('['), text.rfind(']')
    if i < 0 or j <= i:
        return False
    try:
        arr = json.loads(text[i:j + 1])
    except Exception:
        return False
    if not isinstance(arr, list) or len(arr) != 12:
        return False
    try:
        return sorted(int(o['id']) for o in arr) == list(range(1, 13)) and all(
            int(o['square']) == int(o['id']) ** 2 for o in arr)
    except Exception:
        return False


def score(text, expect_digits=None) -> dict:
    # Split on punctuation as well as whitespace. Splitting on whitespace alone scored the
    # comma-separated `count_to_60` output as 2 words / 0 grams / diversity 0.0 -- a false
    # degeneration signal on a perfectly correct answer.
    w = re.findall(r'\w+', text)
    grams = [' '.join(w[i:i + 5]) for i in range(max(0, len(w) - 4))]
    out = {'words': len(w),
           'gram5_diversity': round(len(set(grams)) / len(grams), 3) if grams else 0.0,
           'max_gram5_repeat': Counter(grams).most_common(1)[0][1] if grams else 0,
           # The specific corruption seen on the broken config: LaTeX groups with the
           # operand missing from inside them.
           'empty_latex': len(re.findall(r'\\\(\s*\\\)|\\frac\{\s*\}\{\s*\}', text)),
           'sample': text[:200]}
    if expect_digits:
        hit = sum(1 for d in expect_digits if d in text)
        out['digit_retention'] = round(hit / len(expect_digits), 3)
        out['digits_missing'] = [d for d in expect_digits if d not in text]
    return out


TAKE_ACTION_TOOL = [{
    'type': 'function',
    'function': {'name': 'take_action',
                 'description': 'Take one action in the ARC puzzle.',
                 'parameters': {'type': 'object',
                                'properties': {'action': {'type': 'string',
                                                          'description': 'One of ACTION1..ACTION6 or RESET'}},
                                'required': ['action']}}}]
VALID_ACTIONS = {f'ACTION{i}' for i in range(1, 7)} | {'RESET'}

# Prompts that invite LONG output -- the only kind that exposes a repetition loop -- and
# that carry specific digits, so we can measure whether numeric tokens survive.
HEALTH_PROMPTS = [
    ('long_explain',
     'Explain in detail, step by step, how to multiply 17 by 3, then 48 by 26, then '
     '135 by 7. Show every intermediate result and state each final answer clearly.',
     ['51', '1248', '945']),
    ('long_narrative',
     'Write a detailed multi-paragraph essay about the history of the number zero, '
     'covering at least six distinct historical periods.', None),
    ('long_arc',
     'You are shown a 10x10 grid puzzle. Describe, at length, a systematic strategy for '
     'discovering the rule that maps the input grid to the output grid. Enumerate at '
     'least 12 distinct hypotheses you would test and how you would falsify each.', None),
    ('count_to_60',
     'Count from 1 to 60, separated by commas. Then count back down from 60 to 1.',
     ['13', '27', '41', '58']),
    # Structured output. Degeneration shows up here as unparseable JSON long before it is
    # obvious in prose, and it is objectively checkable rather than eyeballed.
    ('structured_json',
     'Output ONLY a JSON array (no prose, no code fence) of exactly 12 objects, each with '
     'keys "id" (integer 1..12), "name" (a distinct colour), and "square" (id squared). '
     'The last object must have id 12 and square 144.',
     ['144', '121', '100']),
    # Code. Exercises a token distribution -- identifiers, punctuation, indentation --
    # that prose prompts never reach.
    ('code_gen',
     'Write a complete Python function `rle_encode(grid)` that run-length encodes a 2D '
     'list of integers row by row, plus its exact inverse `rle_decode`. Include a '
     'docstring, type hints, and three worked examples in comments showing the encoded '
     'form of [[1,1,2],[3,3,3]].', None),
]


def quick_coherence(arm_label) -> dict:
    """One sanity check, ~30-60 s. Deliberately NOT a gate.

    Run 1 withheld the throughput number because a gate failed on a probe bug -- the single
    most expensive thing that happened. So this records health and lets throughput run
    regardless; a failing arm gets a warning attached to its number, not a missing one.

    The prompt targets the known NVFP4 failure mode specifically: it must echo digits back
    (dropped operands were the signature) and sustain enough length for a loop to appear.
    """
    out = {'mode': 'quick', 'trials': {}}
    name = 'quick_arith'
    prompt = ('Multiply 17 by 3, then 48 by 26, then 135 by 7. Show every intermediate '
              'step, state each final answer, then briefly explain why long multiplication '
              'works.')
    digits = ['51', '1248', '945']
    try:
        t = time.time()
        r = chat(prompt, QUICK_MAX_TOKENS)
        m = r['choices'][0]['message']
        text = (m.get('content') or '') or (m.get('reasoning_content') or '')
        sc = score(text, digits)
        sc['finish'] = r['choices'][0].get('finish_reason')
        sc['completion_tokens'] = (r.get('usage') or {}).get('completion_tokens')
        sc['seconds'] = round(time.time() - t, 1)
        out['trials'][name] = sc
    except Exception as exc:
        out['trials'][name] = {'error': repr(exc)[:300]}
    print('    ' + name + ' ' + json.dumps(out['trials'][name])[:240], flush=True)

    tr = out['trials'][name]
    out['mean_diversity'] = tr.get('gram5_diversity', 0.0)
    out['worst_repeat'] = tr.get('max_gram5_repeat', 0)
    out['mean_digit_retention'] = tr.get('digit_retention')
    out['empty_latex_total'] = tr.get('empty_latex', 0)
    out['stop_frac'] = 1.0 if tr.get('finish') == 'stop' else 0.0
    out['toolcall_valid_frac'] = None      # not exercised in quick mode
    fails = []
    if out['mean_diversity'] < GATE_DIVERSITY:
        fails.append('diversity ' + str(out['mean_diversity']) + ' < ' + str(GATE_DIVERSITY))
    if out['worst_repeat'] >= GATE_MAX_REPEAT:
        fails.append('repeat ' + str(out['worst_repeat']) + ' >= ' + str(GATE_MAX_REPEAT))
    dr = out['mean_digit_retention']
    if dr is not None and dr < GATE_DIGIT_RETENTION:
        fails.append('digit_retention ' + str(dr) + ' < ' + str(GATE_DIGIT_RETENTION))
    if out['empty_latex_total']:
        fails.append('empty_latex ' + str(out['empty_latex_total']))
    out['health_fails'] = fails
    out['health_pass'] = not fails
    print('  QUICK ' + str(arm_label) + ': pass=' + str(out['health_pass'])
          + ' diversity=' + str(out['mean_diversity']) + ' digits=' + str(dr)
          + '  [NON-BLOCKING]', flush=True)
    return out


def health_battery(arm_label) -> dict:
    out = {'trials': {}}
    stops = []
    for name, prompt, digits in HEALTH_PROMPTS:
        try:
            t = time.time()
            r = chat(prompt, HEALTH_MAX_TOKENS)
            m = r['choices'][0]['message']
            text = (m.get('content') or '') or (m.get('reasoning_content') or '')
            s = score(text, digits)
            s['finish'] = r['choices'][0].get('finish_reason')
            s['completion_tokens'] = (r.get('usage') or {}).get('completion_tokens')
            s['seconds'] = round(time.time() - t, 1)
            if name == 'structured_json':
                s['json_array_ok'] = json_array_ok(text)
            stops.append(s['finish'] == 'stop')
            out['trials'][name] = s
        except Exception as exc:
            out['trials'][name] = {'error': repr(exc)[:300]}
            stops.append(False)
        print(f'    {name:16s} ' + json.dumps(out['trials'][name])[:220], flush=True)

    # Tool-call validity. The agent acts ONLY through tool calls: a server that writes
    # beautiful prose but malformed tool calls scores 0 on every game while /health is green.
    # The 2026-08-21 run recorded only the pass COUNT here, so when it came back 0/4 there
    # was no way to tell a broken model from a starved budget -- the check discarded its own
    # evidence. Record finish_reason, token count and a snippet per attempt so the next 0/4
    # is diagnosable from the results file alone.
    ok = 0
    attempts = []
    for i in range(4):
        a = {'i': i}
        try:
            r = chat(f'The grid changed after the last move (attempt {i}). '
                     f'Take exactly one action now.', 4096, tools=TAKE_ACTION_TOOL,
                     no_think=True)
            ch = r['choices'][0]
            m = ch.get('message') or {}
            a['finish'] = ch.get('finish_reason')
            a['completion_tokens'] = (r.get('usage') or {}).get('completion_tokens')
            a['has_tool_calls'] = bool(m.get('tool_calls'))
            a['content_head'] = (m.get('content') or '')[:160]
            a['reasoning_head'] = (m.get('reasoning_content') or '')[:160]
            tc = (m.get('tool_calls') or [])
            if tc:
                args = json.loads(tc[0]['function']['arguments'])
                act = str(args.get('action', '')).strip().upper()
                a['action'] = act[:40]
                if any(v in act for v in VALID_ACTIONS):
                    ok += 1
                    a['valid'] = True
        except Exception as exc:
            a['error'] = repr(exc)[:200]
        attempts.append(a)
        print(f'    toolcall[{i}] ' + json.dumps(a)[:240], flush=True)
    out['toolcall_valid_frac'] = round(ok / 4, 3)
    out['toolcall_attempts'] = attempts

    ds = [t['gram5_diversity'] for t in out['trials'].values() if 'gram5_diversity' in t]
    rp = [t['max_gram5_repeat'] for t in out['trials'].values() if 'max_gram5_repeat' in t]
    dr = [t['digit_retention'] for t in out['trials'].values() if 'digit_retention' in t]
    out['mean_diversity'] = round(sum(ds) / len(ds), 3) if ds else 0.0
    out['worst_repeat'] = max(rp) if rp else 0
    out['mean_digit_retention'] = round(sum(dr) / len(dr), 3) if dr else None
    out['stop_frac'] = round(sum(stops) / len(stops), 3) if stops else 0.0
    out['empty_latex_total'] = sum(t.get('empty_latex', 0) for t in out['trials'].values())
    out['json_array_ok'] = bool(out['trials'].get('structured_json', {}).get('json_array_ok'))

    fails = []
    if out['mean_diversity'] < GATE_DIVERSITY:
        fails.append(f"diversity {out['mean_diversity']} < {GATE_DIVERSITY}")
    if out['worst_repeat'] > GATE_MAX_REPEAT:
        fails.append(f"max 5-gram repeat {out['worst_repeat']} > {GATE_MAX_REPEAT}")
    if out['mean_digit_retention'] is not None and out['mean_digit_retention'] < GATE_DIGIT_RETENTION:
        fails.append(f"digit retention {out['mean_digit_retention']} < {GATE_DIGIT_RETENTION}")
    if out['stop_frac'] < GATE_STOP_FRAC:
        fails.append(f"stop_frac {out['stop_frac']} < {GATE_STOP_FRAC}")
    if out['toolcall_valid_frac'] < 1.0:
        fails.append(f"toolcall_valid_frac {out['toolcall_valid_frac']} < 1.0")
    if out['empty_latex_total']:
        fails.append(f"empty LaTeX groups: {out['empty_latex_total']}")
    out['health_fails'] = fails
    out['health_pass'] = not fails
    print(f'  HEALTH {arm_label}: pass={out["health_pass"]} diversity={out["mean_diversity"]} '
          f'repeat={out["worst_repeat"]} digits={out["mean_digit_retention"]} '
          f'stop={out["stop_frac"]} tools={out["toolcall_valid_frac"]}', flush=True)
    if fails:
        print('  FAILS: ' + '; '.join(fails), flush=True)
    return out


# 220 rows tokenised to 7,936 tokens -> the C=16 test peaked at ~12k context, well short
# of the ~32k the brief asks for. 776 rows measures 27,993 tokens on the RadixArk tokenizer;
# + LOAD_MAX_TOKENS generated = ~32k peak, i.e. the context length actually under test.
LOAD_PROMPT = ('Analyse the following puzzle at length and enumerate every hypothesis you '
               'can justify.\n\n' + ('grid row: 3 1 4 1 5 9 2 6 5 3 5 8 9 7 9 3\n' * 776))


def throughput_c16() -> dict:
    """Matched load. Reported ONLY for arms that passed the health gate -- throughput
    measured on degenerate output is not throughput."""
    res, lock = [], threading.Lock()

    def one(i):
        t = time.time()
        try:
            r = chat(LOAD_PROMPT + f'\n\nRun {i}.', LOAD_MAX_TOKENS, timeout=3600)
            u = r.get('usage') or {}
            with lock:
                res.append({'ok': True, 'seconds': time.time() - t,
                            'completion_tokens': u.get('completion_tokens') or 0,
                            'prompt_tokens': u.get('prompt_tokens') or 0})
        except Exception as exc:
            with lock:
                res.append({'ok': False, 'seconds': time.time() - t, 'error': repr(exc)[:160]})

    t0 = time.time()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(LOAD_CONCURRENCY)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0
    good = [r for r in res if r['ok']]
    gen = sum(r['completion_tokens'] for r in good)
    out = {'concurrency': LOAD_CONCURRENCY, 'wall_seconds': round(wall, 1),
           'ok': len(good), 'failed': len(res) - len(good),
           'gen_tokens': gen, 'agg_gen_tok_s': round(gen / wall, 1) if wall else 0.0,
           'per_req_decode_tok_s': round(
               sum(r['completion_tokens'] / r['seconds'] for r in good) / len(good), 2) if good else 0.0}
    # Spec accept stats, read from the engine log. NOTE: a high accept rate on a
    # degenerate arm means the loop is trivially predictable, not that MTP is working.
    txt = SERVER_LOG.read_text(errors='replace') if SERVER_LOG.exists() else ''
    # Read the acceptance LENGTH, and never fall back to acceptance RATE: vLLM logs the
    # rate in two different units in the same file ("acceptance rate: 46.7" as a percent
    # and "acceptance rate: 0.667" as a fraction). Averaging those mixed units produced a
    # nonsense accept_len_mean of 25.98 on run 5 -- impossible when num_spec_tokens=3 caps
    # the accept length at 4. SGLang says "accept len:", vLLM says "acceptance length:".
    acc = (re.findall(r'accept len: ([0-9.]+)', txt)
           or re.findall(r'acceptance length: ([0-9.]+)', txt))
    if acc:
        vals = [float(a) for a in acc]
        out['accept_len_mean'] = round(sum(vals) / len(vals), 3)
        out['accept_samples'] = len(vals)
    print(f'  LOAD: {out["agg_gen_tok_s"]} agg tok/s, {out["per_req_decode_tok_s"]} per-req, '
          f'{out["ok"]}/{LOAD_CONCURRENCY} ok, {out["wall_seconds"]}s', flush=True)
    return out


AGREE_PROMPTS = [
    'List the first 15 prime numbers, separated by commas.',
    'What is 348 times 27? Answer with the number only.',
    'Name the capitals of France, Japan, Brazil and Kenya, in that order.',
    'Explain in two sentences why the sky appears blue.',
]


def agreement_sample() -> list:
    """Greedy decode of a fixed prompt set, stored per arm. Compared across arms at the
    end. This is unsloth's top-1-agreement metric computed BETWEEN arms rather than
    against BF16 -- it says how far apart they are, not which one is right."""
    out = []
    for p in AGREE_PROMPTS:
        try:
            r = chat(p, 256, temperature=0.0)
            m = r['choices'][0]['message']
            out.append(((m.get('content') or '') or (m.get('reasoning_content') or ''))[:600])
        except Exception as exc:
            out.append(f'<error {repr(exc)[:100]}>')
    return out


# ---------------------------------------------------------------- driver

def main():
    assert_gpu()
    UNSLOTH = resolve_model_dir(resolve_dataset(*UNSLOTH_REF))
    RADIXARK = resolve_model_dir(resolve_dataset(*RADIXARK_REF))
    rec['checkpoints'] = {'unsloth': checkpoint_fingerprint(UNSLOTH),
                          'radixark': checkpoint_fingerprint(RADIXARK)}
    print(json.dumps(rec['checkpoints'], indent=1)[:1500], flush=True)
    save()

    # The brief is a 2-arm test: vLLM x unsloth against SGLang x RadixArk, interleaved so
    # hardware drift and warm-up cannot masquerade as an arm effect. C / F / K are wired
    # up but opt-in -- each costs an extra model load out of a fixed kernel runtime budget.
    # Resolved here, before the arm table, because the prerequisite checks below only
    # apply to arms actually being run.
    order = [k.strip() for k in os.getenv('VENDOR_AB_ORDER', 'A,B,A,B').split(',')]

    # Only arms in `order` need their checkpoint present, so the default 2-arm run does
    # not require the DSpark dataset to be attached.
    DSPARK = None
    if 'K' in order:
        DSPARK = resolve_model_dir(resolve_dataset(*DSPARK_REF))

    # (label, engine, model_dir, attention_backend, extras)
    ARMS = {
        'A': ('A_vllm_unsloth', 'vllm', UNSLOTH, ATTENTION, {}),
        'B': ('B_sglang_radixark', 'sglang', RADIXARK, ATTENTION, {}),
        'C': ('C_vllm_radixark', 'vllm', RADIXARK, ATTENTION, {}),
        'F': ('F_sglang_radixark_flashinfer', 'sglang', RADIXARK, 'flashinfer', {}),
        # K = the SGLang cookbook's rtx6000/nvfp4 recipe verbatim: flashinfer attention,
        # DSPARK draft, mamba tuning. Deliberately UNMATCHED -- it is the vendor-recommended
        # ceiling, the answer to "are we leaving performance on the table by matching?",
        # not a cell in the vendor 2x2.
        'K': ('K_sglang_radixark_cookbook', 'sglang', RADIXARK, 'flashinfer',
              {'mamba': True, 'dspark': DSPARK}),
    }
    for key, (_, _, path, _, extras) in ARMS.items():
        if key not in order:
            continue
        assert path.exists(), f'arm {key}: model dir missing at {path}'
        draft = extras.get('dspark')
        if draft is not None and not draft.exists():
            raise SystemExit(f'arm {key} needs the DSpark draft checkpoint at {draft}; '
                             f'attach the dataset or drop {key} from VENDOR_AB_ORDER')

    sp = prepare_sglang_tree()
    install_vllm()
    save()

    for rep, key in enumerate(order):
        key = key.strip()
        if key not in ARMS:
            continue
        label, engine, model_dir, attention, extras = ARMS[key]
        tag = f'{label}#{rep}'
        print('\n' + '=' * 90 + f'\n{tag}  ({engine} x {model_dir.name} x {attention})\n'
              + '=' * 90, flush=True)
        row = {'arm': key, 'label': label, 'engine': engine, 'rep': rep,
               'model_dir': str(model_dir), 'attention': attention,
               'mamba_tuning': bool(extras.get('mamba')),
               'spec': 'DSPARK' if extras.get('dspark') else 'EAGLE/MTP'}
        stop_arm()
        t0 = time.time()
        try:
            proc = (start_sglang(model_dir, sp, attention,
                                 mamba=bool(extras.get('mamba')),
                                 dspark=extras.get('dspark'))
                    if engine == 'sglang' else start_vllm(model_dir))
            up = wait_ready(proc)
        except Exception as exc:
            row['launch_error'] = repr(exc)[:400]
            up = False
        row['startup_seconds'] = round(time.time() - t0, 1)
        if not up:
            row['died'] = True
            row['log_tail'] = log_tail(200)
            row['root_cause'] = first_error()
            # Preserve this arm's server log before the next arm truncates the shared
            # file. Arm A died twice on 2026-08-21 and its root cause was lost this way:
            # only a 40-line tail survived, ending at 'See root cause above'.
            try:
                if SERVER_LOG.exists():
                    keep = WORKING / ('server-' + tag.replace('#', '-') + '.log')
                    keep.write_text(SERVER_LOG.read_text(errors='replace'), encoding='utf-8')
                    row['server_log'] = keep.name
            except Exception as exc:
                row['server_log_error'] = repr(exc)[:200]
            rec['arms'].append(row); save()
            print(f'  DID NOT COME UP after {row["startup_seconds"]}s', flush=True)
            continue
        time.sleep(5)
        row['served_config'] = gate2_loader_sanity(engine)
        print('  served:', json.dumps(row['served_config'])[:400], flush=True)
        if not row['served_config'].get('quant_ok'):
            # Not fatal to the run -- we still want the health numbers -- but it is a
            # loud, recorded failure of Gate 2.
            print('  GATE2 FAIL: server reports no quantization on a quantized checkpoint',
                  flush=True)
        # v2 is throughput-focused. Run 1 spent ~7.5 min per arm on the 16k health battery
        # and then threw the throughput result away because the gate failed on a probe bug.
        # So: a SHORT coherence check that is RECORDED BUT NEVER BLOCKING, then always
        # measure throughput. A degenerate arm still yields a number -- flagged, not
        # withheld -- because a withheld number is what cost us run 1.
        if FOCUS_THROUGHPUT:
            row.update(quick_coherence(tag))
        else:
            row.update(health_battery(tag))
        row['load'] = throughput_c16()
        if not row.get('health_pass'):
            row['load']['warning'] = ('coherence check did not pass; treat these tok/s as '
                                      'throughput on possibly degenerate output')
        if not FOCUS_THROUGHPUT:
            rec['agreement'].setdefault(label, agreement_sample())
        # Preserve this arm's server log before the next arm truncates the shared
        # file. Arm A died twice on 2026-08-21 and its root cause was lost this way:
        # only a 40-line tail survived, ending at 'See root cause above'.
        try:
            if SERVER_LOG.exists():
                keep = WORKING / ('server-' + tag.replace('#', '-') + '.log')
                keep.write_text(SERVER_LOG.read_text(errors='replace'), encoding='utf-8')
                row['server_log'] = keep.name
        except Exception as exc:
            row['server_log_error'] = repr(exc)[:200]
        rec['arms'].append(row); save()

    stop_arm()

    # Cross-arm agreement on the greedy samples.
    labels = sorted(rec['agreement'])
    rec['agreement_matrix'] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = rec['agreement'][labels[i]], rec['agreement'][labels[j]]
            same = sum(1 for x, y in zip(a, b) if x.strip() == y.strip())
            rec['agreement_matrix'][f'{labels[i]} vs {labels[j]}'] = {
                'exact_match': f'{same}/{len(a)}'}
    rec['done'] = True
    save()

    print('\n' + '=' * 90 + '\nSUMMARY\n' + '=' * 90, flush=True)
    for r in rec['arms']:
        if r.get('died'):
            print(f'  {r["label"]:22s} rep{r["rep"]}  DIED after {r["startup_seconds"]}s', flush=True)
            continue
        load = r.get('load', {})
        tok = load.get('agg_gen_tok_s', '-') if 'skipped' not in load else 'SKIPPED'
        print(f'  {r["label"]:22s} rep{r["rep"]}  health={r["health_pass"]} '
              f'div={r["mean_diversity"]} digits={r.get("mean_digit_retention")} '
              f'tools={r.get("toolcall_valid_frac")} agg_tok/s={tok}', flush=True)
    print('\nagreement:', json.dumps(rec.get('agreement_matrix', {})), flush=True)


try:
    main()
except BaseException as exc:
    rec['fatal'] = f'{type(exc).__name__}: {exc}'
    rec['tb'] = traceback.format_exc()[-4000:]
    save()
    print(rec['tb'], flush=True)
    raise
