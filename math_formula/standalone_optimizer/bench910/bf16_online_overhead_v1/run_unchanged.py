#!/usr/bin/env python3
"""Paired BF16 MatMulV3 wall time for unchanged master/online tilings."""
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = ROOT / 'bf16_online_vs_master'
OUT = ROOT / 'cases/bf16_online_overhead_v1/unchanged'
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
PACKAGES = {
    'master': ROOT / 'ops_nn_master_bf16_proto_package',
    'online': ROOT / 'ops_nn_bf16_source_base_v1_package',
}
# Each shape's full emitted tile agrees between arms in bf16_source_base_v1.
SHAPES = [(16, 16, 65408), (2048, 4096, 4096), (8192, 8192, 16384)]

base = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                       'bash', str(CANN)], check=True, capture_output=True).stdout
base_env = dict(item.decode().split('=', 1) for item in base.split(b'\0') if item)
OUT.mkdir(parents=True, exist_ok=True)
records = []
for shape in SHAPES:
    for round_idx in range(3):
        for arm in (('master', 'online') if round_idx % 2 == 0 else ('online', 'master')):
            env = base_env.copy()
            for key in list(env):
                if key.startswith('MMV3_ONLINE_'):
                    env.pop(key)
            package = PACKAGES[arm]
            env.update(BENCH_CUSTOM_OPP_PATH=str(package), ASCEND_CUSTOM_OPP_PATH=str(package),
                       MMV3_SKIP_LEGACY_OP_CACHE='1', ASCEND_GLOBAL_LOG_LEVEL='3')
            env['LD_LIBRARY_PATH'] = str(package / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
            if arm == 'online':
                env.update(MMV3_ONLINE_SELECTOR='1', MMV3_ONLINE_BF16_EXPERIMENT='1')
            cmd = [str(HERE / 'host_wall_bench'), '--api', 'v3cached', '--device', '4',
                   '--m', str(shape[0]), '--n', str(shape[1]), '--k', str(shape[2]),
                   '--prewarm', '1', '--warmup', '3', '--batches', '1', '--repetitions', '21']
            print(shape, round_idx, arm, flush=True)
            proc = subprocess.run(cmd, env=env, cwd=HERE, capture_output=True, text=True,
                                  timeout=900)
            if proc.returncode:
                raise RuntimeError(f'{shape} {arm} {round_idx}: {proc.stderr[-2000:]}')
            record = json.loads(proc.stdout.splitlines()[-1])
            record.update(arm=arm, round=round_idx, prewarm_shape=[16, 16, 128])
            records.append(record)
            (OUT / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
            print({k: round(record[k], 1) for k in ('first_host_us', 'first_wall_us',
                   'host_median_us', 'wall_median_us', 'median_us')}, flush=True)
