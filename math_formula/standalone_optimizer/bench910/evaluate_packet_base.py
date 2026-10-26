#!/usr/bin/env python3
"""Source-verified SolveBase packet-objective experiment on the 910B3.

The shapes are fixed below. `defaults` records the emitted source tuple;
the local C++ solver freezes `selection.json` before smoke/repeat timings.
Only distinct, source-correct optimizer/default pairs enter repeats.
"""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
EXPERIMENT = ROOT / 'cases/packet_objective_v1'
SHAPES = (
    (512, 4096, 8192), (4096, 512, 8192),
    (1024, 4096, 4096), (4096, 1024, 4096),
    (2048, 6144, 4096), (6144, 2048, 4096),
    (1536, 3072, 8192), (3072, 1536, 8192),
    (256, 8192, 8192), (8192, 256, 8192),
    (384, 6144, 8192), (6144, 384, 8192),
)
FIELDS = ('singleCoreM', 'singleCoreN', 'baseM', 'baseN', 'baseK',
          'block_dim')


def environment(tile):
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
    if tile is not None:
        env['MMV3_RESEARCH_BASE_TILE'] = ','.join(map(str, tile))
    return env


def run_case(shape, tile, directory, repetitions):
    env = environment(tile)
    if not (directory / 'case.json').exists():
        subprocess.run(['bash', str(ROOT / 'run_case.sh'),
                        *map(str, shape), str(directory), '0',
                        '10', str(repetitions)], env=env, check=True,
                       stdout=subprocess.DEVNULL)
    case = json.loads((directory / 'case.json').read_text())
    tiling = json.loads((directory / 'tiling.json').read_text())
    emitted = tuple(case[field] for field in FIELDS)
    if (case['shape'] != list(shape) or case['max_abs_error'] != 0 or
            tiling['family'] != 'BASE' or tiling['nd2nzA'] != 0 or
            tiling['nd2nzB'] != 0 or case['tiling_key'] != 65536 or
            not case['tiling_source'].startswith(env['BENCH_EXPECT_SOURCE']) or
            case['kernel_count'] != repetitions or
            (tile is not None and emitted != tuple(tile))):
        raise RuntimeError(f'bad source/correctness/tiling: {directory}: '
                           f'expected={tile} emitted={emitted}')
    return case, emitted


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')


def defaults():
    path = EXPERIMENT / 'defaults.jsonl'
    done = {tuple(row['shape']) for row in
            map(json.loads, path.read_text().splitlines())} \
        if path.exists() else set()
    for shape in SHAPES:
        if shape in done:
            continue
        label = 'x'.join(map(str, shape))
        directory = EXPERIMENT / 'defaults' / label
        try:
            case, tile = run_case(shape, None, directory, 4)
            row = {'shape': shape, 'status': 'eligible', 'tile': tile,
                   'case': case}
            print(f'{label} default={tile}', flush=True)
        except (subprocess.CalledProcessError, RuntimeError) as error:
            row = {'shape': shape, 'status': 'ineligible',
                   'reason': str(error)}
            print(f'{label} ineligible: {error}', flush=True)
        append(path, row)


def selection(path):
    result = json.loads(path.read_text())
    if [tuple(shape) for shape in result['declared_shapes']] != list(SHAPES):
        raise ValueError('selection shape list differs from frozen suite')
    return [row for row in result['rows']
            if row['status'] == 'selected' and
            row['optimizer_tile'] != row['default_tile']]


def timed(phase, selection_path, rounds, repetitions):
    choices = selection(selection_path)
    output = EXPERIMENT / f'{phase}_raw.jsonl'
    done = {(row['round'], tuple(row['shape']), row['arm']) for row in
            map(json.loads, output.read_text().splitlines())} \
        if output.exists() else set()
    schedule = []
    rng = random.Random(20261026)
    for round_index in range(1 if phase == 'smoke' else rounds):
        shuffled = choices.copy()
        rng.shuffle(shuffled)
        for row in shuffled:
            arms = ['optimizer', 'default']
            rng.shuffle(arms)
            schedule.extend((round_index, row, arm) for arm in arms)
    for round_index, row, arm in schedule:
        shape = tuple(row['shape'])
        if (round_index, shape, arm) in done:
            continue
        label = 'x'.join(map(str, shape))
        tile = row['optimizer_tile'] if arm == 'optimizer' else None
        expected = row['optimizer_tile'] if arm == 'optimizer' \
            else row['default_tile']
        directory = EXPERIMENT / phase / f'r{round_index}_{label}_{arm}'
        case, emitted = run_case(shape, tile, directory, repetitions)
        if list(emitted) != expected:
            raise RuntimeError(f'default changed: {shape} expected={expected} '
                               f'emitted={emitted}')
        append(output, {'round': round_index, 'shape': shape, 'arm': arm,
                        'optimizer_tile': row['optimizer_tile'],
                        'default_tile': row['default_tile'],
                        'case': case, 'unix_time': time.time()})
        print(f'{phase} r{round_index} {label} {arm} '
              f'{case["kernel_median_us"]:.3f} us', flush=True)


def main():
    global EXPERIMENT, SHAPES
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('defaults', 'smoke', 'repeat'))
    parser.add_argument('--shapes-file', type=Path,
                        help='JSON list of [M,N,K] shapes for a new suite')
    parser.add_argument('--experiment', type=Path,
                        help='separate output directory for a new suite')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--repetitions', type=int, default=8)
    args = parser.parse_args()
    if args.shapes_file is not None:
        SHAPES = tuple(tuple(map(int, shape)) for shape in
                       json.loads(args.shapes_file.read_text()))
        if not SHAPES or len(set(SHAPES)) != len(SHAPES) or any(
                len(shape) != 3 or min(shape) <= 0 for shape in SHAPES):
            parser.error('--shapes-file must contain unique positive M,N,K triples')
    if args.experiment is not None:
        EXPERIMENT = args.experiment
    if args.shapes_file is not None and args.experiment is None:
        parser.error('--experiment is required with --shapes-file')
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'defaults':
        defaults()
    else:
        if args.selection is None:
            parser.error('--selection is required for timed phases')
        timed(args.phase, args.selection, args.rounds, args.repetitions)


if __name__ == '__main__':
    main()
