#!/usr/bin/env python3
"""Source-checked default versus full SolveBase tiling on the 910B3.

The optimizer proposals are frozen in the supplied audit CSV before
collecting device timings. Every arm checks all output elements and the
emitted MatMulV3 tuple before its kernel task times are admitted.
"""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
DEFAULT_TILE = (128, 256, 128, 256, 64, 20)
FIELDS = ('singleCoreM', 'singleCoreN', 'baseM', 'baseN', 'baseK',
          'block_dim')


def proposals(path):
    choices = {}
    for line in path.read_text().splitlines():
        row = line.split(',')
        if len(row) < 11 or row[3] != 'BASE':
            raise ValueError(f'invalid audit row: {line}')
        shape = tuple(map(int, row[:3]))
        if shape in choices:
            raise ValueError(f'duplicate shape: {shape}')
        choices[shape] = tuple(map(int, row[4:10]))
    return choices


def run_case(shape, arm, tile, directory, warmup, repetitions):
    env = os.environ.copy()
    env.update({
        'MMV3_SKIP_LEGACY_OP_CACHE': '1',
        'BENCH_API': 'v3cached',
        'BENCH_CUSTOM_OPP_PATH': str(ROOT / 'ops_nn_full_tile_package'),
        'BENCH_EXPECT_SOURCE':
            str(ROOT / 'ops_nn_source_build/matmul/mat_mul_v3/'),
        'CANN_ROOT':
            '/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0',
    })
    env.pop('MMV3_FORCE_BASE_M', None)
    env.pop('MMV3_RESEARCH_BASE_TILE', None)
    if arm == 'optimizer':
        env['MMV3_RESEARCH_BASE_TILE'] = ','.join(map(str, tile))
    elif arm != 'default':
        raise ValueError(arm)
    if not (directory / 'case.json').exists():
        subprocess.run(['bash', str(ROOT / 'run_case.sh'),
                        *map(str, shape), str(directory), '0',
                        str(warmup), str(repetitions)],
                       env=env, check=True, stdout=subprocess.DEVNULL)
    case = json.loads((directory / 'case.json').read_text())
    tiling = json.loads((directory / 'tiling.json').read_text())
    emitted = tuple(case[field] for field in FIELDS)
    expected = tile if arm == 'optimizer' else DEFAULT_TILE
    if (case['shape'] != list(shape) or case['max_abs_error'] != 0 or
            tiling['family'] != 'BASE' or tiling['nd2nzA'] != 0 or
            tiling['nd2nzB'] != 0 or case['tiling_key'] != 65536 or
            emitted != expected or
            not case['tiling_source'].startswith(env['BENCH_EXPECT_SOURCE']) or
            case['kernel_count'] != repetitions):
        raise RuntimeError(
            f'output/emitted-tile/source check failed: {directory}: '
            f'expected {expected}, emitted {emitted}')
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('smoke', 'repeat'))
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--shape-index', type=int,
                        help='smoke only one zero-based audit row')
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=8)
    args = parser.parse_args()
    choices = proposals(args.audit)
    shapes = list(choices)
    if args.shape_index is not None:
        shapes = [shapes[args.shape_index]]
    root = ROOT / 'cases/source_compatible_base_v1'
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == 'smoke':
        schedule = [(0, shape, arm) for shape in shapes
                    for arm in ('optimizer', 'default')]
    else:
        rng = random.Random(20261025)
        schedule = []
        for round_index in range(args.rounds):
            shuffled_shapes = shapes.copy()
            rng.shuffle(shuffled_shapes)
            for shape in shuffled_shapes:
                arms = ['optimizer', 'default']
                rng.shuffle(arms)
                schedule.extend((round_index, shape, arm) for arm in arms)
    output = root / f'{args.phase}_raw.jsonl'
    done = {(row['round'], tuple(row['shape']), row['arm']) for row in
            map(json.loads, output.read_text().splitlines())} \
        if output.exists() else set()
    for round_index, shape, arm in schedule:
        if (round_index, shape, arm) in done:
            continue
        label = 'x'.join(map(str, shape))
        directory = root / args.phase / f'r{round_index}_{label}_{arm}'
        case = run_case(shape, arm, choices[shape], directory,
                        args.warmup, args.repetitions)
        row = {'round': round_index, 'shape': shape, 'arm': arm,
               'optimizer_tile': choices[shape], 'default_tile': DEFAULT_TILE,
               'unix_time': time.time(), 'case': case}
        with output.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        print(f'{args.phase} r{round_index} {label} {arm} '
              f'{case["kernel_median_us"]:.3f} us', flush=True)


if __name__ == '__main__':
    main()
