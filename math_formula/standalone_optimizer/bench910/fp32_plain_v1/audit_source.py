#!/usr/bin/env python3
"""Audit clean-master FP32 MatMulV3 modes and emitted tiles on 910B3."""
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
PACKAGE = ROOT / 'ops_nn_master_bf16_proto_package'
OUT = ROOT / 'cases/fp32_plain_v1'
SHAPES = [(256, 256, 1024), (128, 128, 32768), (16, 16, 65536),
          (2048, 2048, 4096), (512, 512, 4096), (49152, 384, 1536)]

raw = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                      'bash', str(CANN)], capture_output=True, check=True).stdout
env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
env.update(BENCH_CUSTOM_OPP_PATH=str(PACKAGE), ASCEND_CUSTOM_OPP_PATH=str(PACKAGE),
           MMV3_SKIP_LEGACY_OP_CACHE='1', ASCEND_GLOBAL_LOG_LEVEL='1')
env['LD_LIBRARY_PATH'] = str(PACKAGE / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
OUT.mkdir(parents=True, exist_ok=True)
records = json.loads((OUT / 'source_audit.json').read_text()) \
    if (OUT / 'source_audit.json').exists() else []
for shape in SHAPES:
    if any(tuple(record['shape']) == shape for record in records):
        continue
    print(shape, flush=True)
    cmd = [str(ROOT / 'fp32_bench'), '--api', 'v3cached', '--device', '4',
           '--m', str(shape[0]), '--n', str(shape[1]), '--k', str(shape[2]),
           '--warmup', '1', '--batches', '1', '--repetitions', '3']
    proc = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True,
                          timeout=600)
    if proc.returncode:
        raise RuntimeError(f'{shape}: {proc.stderr[-1500:]}')
    event = json.loads(proc.stdout.splitlines()[-1])
    tiling = subprocess.run(['python3', str(ROOT / 'parse_tiling_log.py'),
                             '--pid', str(event['pid'])], env=env, capture_output=True,
                            text=True, check=True)
    record = {'shape': shape, 'event': event, 'tiling': json.loads(tiling.stdout)}
    records.append(record)
    (OUT / 'source_audit.json').write_text(json.dumps(records, indent=2) + '\n')
    print({'mode': record['tiling']['tilingEnableSplitCore'],
           'core': tuple(record['tiling'][x] for x in ('singleCoreM','singleCoreN','singleCoreK')),
           'base': tuple(record['tiling'][x] for x in ('baseM','baseN','baseK')),
           'error': event['max_relative_error']}, flush=True)
