"""Verify every vLLM CLI flag the probe passes still exists in 0.27.1.

An unrecognised option is SystemExit(2) with no server and no useful log -- on a billed
run that costs the whole arm. The probe already does this for SGLang by parsing --help;
vLLM has no equivalent guard, so check it offline against the wheel.
"""
import re, zipfile
from pathlib import Path

WHL = ('/home/018270239/arc-agi-3/model-staging/vllm-0271-wheelhouse/'
       'vllm-0.27.1-cp38-abi3-manylinux_2_28_x86_64.whl')
PROBE = ('/home/018270239/arc-agi-3/duck-harness/.claude/worktrees/nvfp4-vendor-ab/'
         'ARC3-Inference/runs/20260821_nvfp4_vendor_ab/vendor-ab-probe.py')

src = Path(PROBE).read_text()
blk = src[src.index('def start_vllm'):]
blk = blk[:blk.index('\ndef ')]
flags = sorted(set(re.findall(r"'(--[a-z0-9][a-z0-9-]+)'", blk)))
print(f'probe passes {len(flags)} vLLM flags:')
for f in flags:
    print('   ', f)

z = zipfile.ZipFile(WHL)
py = [n for n in z.namelist() if n.endswith('.py')]
blob = b''
for n in py:
    if any(k in n for k in ('engine/arg_utils', 'entrypoints/openai/cli_args',
                            'entrypoints/cli', 'config/', 'engine/')):
        blob += z.read(n)
text = blob.decode('utf-8', 'replace')

print('\nresolution against the 0.27.1 wheel:')
missing = []
for f in flags:
    dashed = f in text
    under = f.lstrip('-').replace('-', '_') in text
    ok = dashed or under
    print(f'  {f:34s} {"OK" if ok else "MISSING"}  ({"dashed" if dashed else ""}{" underscore" if under else ""})')
    if not ok:
        missing.append(f)

print()
if missing:
    print('MISSING FLAGS:', missing)
    raise SystemExit(1)
print('ALL vLLM FLAGS RESOLVE')
