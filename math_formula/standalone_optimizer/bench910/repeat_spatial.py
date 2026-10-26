#!/usr/bin/env python3
"""Repeat checked source-matched BASE tile comparisons in randomized order.

Run on the prepared 910B3 checkout. Each call to run_case.sh validates the
whole FP16 output and emitted tiling before saving profiler kernel times.
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
POINTS = [
    ('spatial_256_auto', (256, 256, 1024), None, 32),
    ('spatial_256_bm16', (256, 256, 1024), 16, 16),
    ('spatial_512_auto', (512, 512, 1024), None, 112),
    ('spatial_512_bm96b', (512, 512, 1024), 96, 96),
    ('spatial_512_bm64', (512, 512, 1024), 64, 64),
    ('spatial_512_bm32', (512, 512, 1024), 32, 32),
    ('spatial_1024_auto', (1024, 1024, 4096), None, 128),
    ('spatial_1024_bm64', (1024, 1024, 4096), 64, 64),
]


def main():
    output = ROOT / 'cases/spatial_repeat_raw.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {(r['round'], r['case_id']) for r in map(json.loads,
            output.read_text().splitlines())} if output.exists() else set()
    base_env = os.environ.copy()
    base_env.update({
        'MMV3_SKIP_LEGACY_OP_CACHE': '1',
        'BENCH_API': 'v3cached',
        'BENCH_CUSTOM_OPP_PATH': str(ROOT / 'ops_nn_spatial_package'),
        'BENCH_EXPECT_SOURCE': str(ROOT / 'ops_nn_source_build/matmul/mat_mul_v3/'),
        'CANN_ROOT': '/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0',
    })
    rng = random.Random(9102026)
    for round_index in range(5):
        order = POINTS.copy()
        rng.shuffle(order)
        for case_id, shape, forced_base_m, expected_base_m in order:
            if (round_index, case_id) in done:
                continue
            env = base_env.copy()
            env.pop('MMV3_FORCE_BASE_M', None)
            if forced_base_m is not None:
                env['MMV3_FORCE_BASE_M'] = str(forced_base_m)
            directory = ROOT / 'cases/spatial_repeat' / f'r{round_index}_{case_id}'
            cmd = ['bash', str(ROOT / 'run_case.sh'),
                   *map(str, shape), str(directory), '0', '10', '8']
            if not (directory / 'case.json').exists():
                subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
            case = json.loads((directory / 'case.json').read_text())
            tiling = json.loads((directory / 'tiling.json').read_text())
            if (case['baseM'] != expected_base_m or
                    case['singleCoreM'] != expected_base_m or
                    case['shape'] != list(shape) or
                    case['max_abs_error'] != 0 or
                    case['tiling_key'] != 65536 or
                    tiling['nd2nzA'] != 0 or tiling['nd2nzB'] != 0):
                raise RuntimeError(f'unexpected checked case {case_id}: {case}')
            row = {'round': round_index, 'case_id': case_id,
                   'unix_time': time.time(), 'case': case}
            with output.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            print(f'round {round_index + 1}/5 {case_id} '
                  f"{case['kernel_median_us']:.3f} us", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'spatial repeat failed: {exc}', file=sys.stderr)
        raise
