#!/usr/bin/env python3
"""Profile audited FP32 source MatMulV3 kernels on 910B3."""
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
PACKAGE = ROOT / 'ops_nn_master_bf16_proto_package'
OUT = ROOT / 'cases/fp32_plain_v1'
audits = json.loads((OUT / 'source_audit.json').read_text())
selected = {(256, 256, 1024), (512, 512, 4096), (49152, 384, 1536),
            (2048, 2048, 4096), (128, 128, 32768)}
raw = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                      'bash', str(CANN)], capture_output=True, check=True).stdout
env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
env.update(BENCH_CUSTOM_OPP_PATH=str(PACKAGE), ASCEND_CUSTOM_OPP_PATH=str(PACKAGE),
           MMV3_SKIP_LEGACY_OP_CACHE='1', ASCEND_GLOBAL_LOG_LEVEL='3')
env['LD_LIBRARY_PATH'] = str(PACKAGE / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
records = json.loads((OUT / 'source_profiles.json').read_text()) \
    if (OUT / 'source_profiles.json').exists() else []
for audit in audits:
    shape = tuple(audit['shape'])
    if shape not in selected or any(tuple(x['shape']) == shape for x in records):
        continue
    print(shape, flush=True)
    path = OUT / ('x'.join(map(str, shape)))
    path.mkdir(parents=True, exist_ok=True)
    cmd = ['msprof', '--output=' + str(path / 'profile'), '--task-time=on',
           '--ai-core=on', 'env', str(ROOT / 'fp32_bench'), '--api', 'v3cached',
           '--device', '4', '--m', str(shape[0]), '--n', str(shape[1]),
           '--k', str(shape[2]), '--warmup', '5', '--batches', '1',
           '--repetitions', '5']
    with (path / 'msprof_stdout.log').open('w') as stdout, \
         (path / 'msprof_stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=900)
    if proc.returncode:
        raise RuntimeError(f'{shape}: {proc.returncode}: '
                           + (path / 'msprof_stderr.log').read_text()[-1500:])
    mode = audit['tiling']['tilingEnableSplitCore']
    kind = 'MIX_AIC' if mode == 3 else 'KERNEL_AICORE'
    parsed = subprocess.run(['python3', str(ROOT / 'bf16_online_vs_master/parse_profile.py'),
                             str(path / 'profile'), '--last', '5', '--kernel-type', kind],
                            capture_output=True, text=True, check=True)
    result = json.loads(parsed.stdout)
    record = {'shape': shape, 'mode': mode, 'profile': result}
    records.append(record)
    (OUT / 'source_profiles.json').write_text(json.dumps(records, indent=2) + '\n')
    print({'mode': mode, 'kernel_us': result['median_us']}, flush=True)
