#!/usr/bin/env python3
"""Source-verified deterministic Split-K timing across the 65535 K boundary.

Runs the original source decision twice per shape (no research tile override).
Each invocation is resumable and appends only fully checked profiler runs.
"""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
EXPERIMENT = ROOT / 'cases/k_gt_65535_v1'
MN_SHAPES = ((256, 256), (512, 512), (1024, 1024))
K_VALUES = (65408, 65536)
FIELDS = ('singleCoreM', 'singleCoreN', 'singleCoreK', 'baseM', 'baseN',
          'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb', 'depthA1', 'depthB1',
          'block_dim', 'tiling_key', 'nd2nzA', 'nd2nzB',
          'tilingEnableSplitCore')


def run_one(m, n, k, phase, round_index, repetitions):
    label = f'{m}x{n}x{k}'
    audit = json.loads((EXPERIMENT / f'audit_{label}' / 'tiling.json').read_text())
    directory = EXPERIMENT / phase / f'r{round_index}_{label}'
    env = os.environ.copy()
    env.update({
        'MMV3_SKIP_LEGACY_OP_CACHE': '1',
        'BENCH_API': 'v3cached',
        'BENCH_CUSTOM_OPP_PATH': str(ROOT / 'ops_nn_full_tile_package'),
        'BENCH_EXPECT_SOURCE':
            str(ROOT / 'ops_nn_source_build/matmul/mat_mul_v3/'),
        'BENCH_EXPECT_SPLIT_CORE_MODE': '3',
        'BENCH_EXPECT_ND2NZ_A': str(int(k > 65535)),
        'BENCH_EXPECT_ND2NZ_B': '0',
        'BENCH_PROFILE_KERNEL_TYPE': 'MIX_AIC',
        'CANN_ROOT':
            '/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0',
    })
    env.pop('MMV3_RESEARCH_BASE_TILE', None)
    env.pop('MMV3_FORCE_BASE_M', None)
    if not (directory / 'case.json').exists():
        subprocess.run(['bash', str(ROOT / 'run_case.sh'), str(m), str(n), str(k),
                        str(directory), '0', '10', str(repetitions)],
                       env=env, check=True, stdout=subprocess.DEVNULL)
    tiling = json.loads((directory / 'tiling.json').read_text())
    case = json.loads((directory / 'case.json').read_text())
    if any(tiling.get(field) != audit.get(field) for field in FIELDS):
        raise RuntimeError(f'emitted tiling differs from frozen audit: {label}')
    if (case['shape'] != [m, n, k] or case['max_relative_error'] > 0.02 or
            case['kernel_count'] != repetitions or
            case['split_core_mode'] != 3 or
            case['nd2nz_a'] != int(k > 65535) or
            case['nd2nz_b'] != 0 or
            case['estimated_nonidle_aics'] != 20 or
            not case['tiling_source'].startswith(env['BENCH_EXPECT_SOURCE'])):
        raise RuntimeError(f'bad source, output, or profile: {label}')
    return {'round': round_index, 'phase': phase, 'shape': [m, n, k],
            'case': case, 'unix_time': time.time()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('smoke', 'repeat'))
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--repetitions', type=int, default=8)
    args = parser.parse_args()
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    output = EXPERIMENT / f'{args.phase}_raw.jsonl'
    completed = {(r['round'], tuple(r['shape'])) for r in
                 map(json.loads, output.read_text().splitlines())} \
        if output.exists() else set()
    rng = random.Random(20261026)
    shapes = [(m, n, k) for m, n in MN_SHAPES for k in K_VALUES]
    for round_index in range(1 if args.phase == 'smoke' else args.rounds):
        order = shapes.copy()
        rng.shuffle(order)
        for shape in order:
            if (round_index, shape) in completed:
                continue
            row = run_one(*shape, args.phase, round_index, args.repetitions)
            with output.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            print(args.phase, f'r{round_index}', 'x'.join(map(str, shape)),
                  f'{row["case"]["kernel_median_us"]:.3f} us', flush=True)


if __name__ == '__main__':
    main()
