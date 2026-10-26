#!/usr/bin/env python3
"""Source-checked, restricted BASE M-tile experiment on the prepared 910B3.

The declared workload list is fixed before looking at any new timings.
Selection uses the current reduced SolveBase objective and primitive probe
slopes, with source default N/K/core choices held fixed. This tests the
source-executable M-tile slice of the model, not full SolveReducedMip.
"""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SHAPES = [
    (128, 256, 1024), (128, 256, 4096),
    (256, 256, 1024), (256, 256, 4096),
    (256, 512, 1024), (256, 512, 4096),
    (512, 256, 1024), (512, 256, 4096),
    (512, 512, 1024), (512, 512, 4096),
    (1024, 512, 4096), (1024, 1024, 4096),
]
LARGE_SHAPES = [
    (2048, 2048, 4096),
    (2048, 4096, 8192),
    (4096, 2048, 8192),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (8192, 4096, 4096),
    (8192, 8192, 4096),
]
PROFILE = {
    'cube_flop_per_us_per_aic': 14719432.0,
    'read_bytes_per_us_20aic': 3710212.0,
    'write_bytes_per_us_20aic': 5229502.0,
    'read_packet_us': 0.109620,
    'write_packet_us': 0.162586,
}


def environment(base_m):
    env = os.environ.copy()
    env.update({
        'MMV3_SKIP_LEGACY_OP_CACHE': '1',
        'BENCH_API': 'v3cached',
        'BENCH_CUSTOM_OPP_PATH': str(ROOT / 'ops_nn_spatial_package'),
        'BENCH_EXPECT_SOURCE': str(ROOT / 'ops_nn_source_build/matmul/mat_mul_v3/'),
        'CANN_ROOT': '/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0',
    })
    env.pop('MMV3_FORCE_BASE_M', None)
    if base_m is not None:
        env['MMV3_FORCE_BASE_M'] = str(base_m)
    return env


def run_case(shape, directory, base_m=None, warmup=10, repetitions=8):
    if not (directory / 'case.json').exists():
        subprocess.run(['bash', str(ROOT / 'run_case.sh'),
                        *map(str, shape), str(directory), '0',
                        str(warmup), str(repetitions)],
                       env=environment(base_m), check=True,
                       stdout=subprocess.DEVNULL)
    case = json.loads((directory / 'case.json').read_text())
    tiling = json.loads((directory / 'tiling.json').read_text())
    if (case['shape'] != list(shape) or case['max_abs_error'] != 0 or
            tiling['family'] != 'BASE' or tiling['nd2nzA'] != 0 or
            tiling['nd2nzB'] != 0 or
            (base_m is not None and
             (case['baseM'] != base_m or case['singleCoreM'] != base_m))):
        raise RuntimeError(f'source/correctness/tiling check failed: {directory}')
    return case


def suite_root(suite):
    return ROOT / ('cases/optimizer_vs_default' if suite == 'small'
                   else 'cases/optimizer_vs_default_large')


def defaults(suite):
    root = suite_root(suite)
    shapes = SHAPES if suite == 'small' else LARGE_SHAPES
    output = root / 'default_cases.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {tuple(r['shape']) for r in map(json.loads,
            output.read_text().splitlines())} if output.exists() else set()
    for shape in shapes:
        if shape in done:
            continue
        label = 'x'.join(map(str, shape))
        directory = root / 'defaults' / label
        try:
            case = run_case(shape, directory)
            row = {'shape': shape, 'status': 'eligible', 'case': case}
            print(f'{label} baseM={case["baseM"]} baseN={case["baseN"]} '
                  f'baseK={case["baseK"]} {case["kernel_median_us"]:.3f} us',
                  flush=True)
        except subprocess.CalledProcessError as exc:
            row = {'shape': shape, 'status': 'source_rejected', 'reason': str(exc)}
            print(f'{label} source rejected', flush=True)
        with output.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')


def reduced_base_us(shape, base_m, base_n, base_k, available_cores):
    m, n, k = shape
    nm = math.ceil(m / base_m)
    nn = math.ceil(n / base_n)
    output_tiles = nm * nn
    occupied = min(available_cores, output_tiles)
    read_bytes = 2 * k * (m * nn + n * nm)
    read_packets = 2 * output_tiles * math.ceil(k / base_k)
    cube = 2 * m * n * k / (occupied * PROFILE['cube_flop_per_us_per_aic'])
    memory = (read_bytes / PROFILE['read_bytes_per_us_20aic'] +
              2 * m * n / PROFILE['write_bytes_per_us_20aic'] +
              read_packets * PROFILE['read_packet_us'] +
              output_tiles * PROFILE['write_packet_us'])
    return max(cube, memory)


def select(default_file, output, suite):
    rows = [json.loads(line) for line in default_file.read_text().splitlines()]
    selected = []
    for row in rows:
        if row['status'] != 'eligible':
            selected.append(row)
            continue
        case = row['case']
        shape = tuple(row['shape'])
        if (case['singleCoreM'] != case['baseM'] or
                case['singleCoreN'] != case['baseN'] or
                case['baseM'] % 16 != 0 or case['baseN'] % 16 != 0):
            selected.append({'shape': shape, 'status': 'outside_restricted_domain',
                             'default': case})
            continue
        # Source override is downward-only and changes baseM and singleCoreM
        # together. All candidate values are fixed before comparison timing.
        values = list(range(16, case['baseM'] + 1, 16))
        scores = {str(v): reduced_base_us(shape, v, case['baseN'],
                                          case['baseK'], case['block_dim'])
                  for v in values}
        chosen = min(values, key=lambda v: (scores[str(v)], -v))
        selected.append({'shape': shape, 'status': 'selected',
                         'default_base_m': case['baseM'],
                         'selected_base_m': chosen,
                         'default_case': case,
                         'surrogate_us_by_base_m': scores})
    output.write_text(json.dumps({'declared_shapes':
                                  SHAPES if suite == 'small' else LARGE_SHAPES,
                                  'profile': PROFILE,
                                  'rows': selected}, indent=2, sort_keys=True))
    for row in selected:
        if row['status'] == 'selected':
            print(row['shape'], row['default_base_m'], '->',
                  row['selected_base_m'], flush=True)
        else:
            print(row['shape'], row['status'], flush=True)


def compare(selection_file, suite):
    selected = json.loads(selection_file.read_text())['rows']
    pairs = [r for r in selected if r['status'] == 'selected' and
             r['selected_base_m'] != r['default_base_m']]
    root = suite_root(suite)
    output = root / 'comparison_raw.jsonl'
    done = {(r['round'], tuple(r['shape']), r['arm']) for r in
            map(json.loads, output.read_text().splitlines())} if output.exists() else set()
    rng = random.Random(20261025)
    for round_index in range(10):
        schedule = [(r, arm) for r in pairs for arm in ('default', 'optimizer')]
        rng.shuffle(schedule)
        for row, arm in schedule:
            shape = tuple(row['shape'])
            if (round_index, shape, arm) in done:
                continue
            base_m = row['selected_base_m'] if arm == 'optimizer' else None
            label = 'x'.join(map(str, shape))
            directory = (root / 'comparison' /
                         f'r{round_index}_{label}_{arm}')
            case = run_case(shape, directory, base_m)
            if arm == 'default' and case['baseM'] != row['default_base_m']:
                raise RuntimeError(f'default tiling changed: {shape}')
            record = {'round': round_index, 'shape': shape, 'arm': arm,
                      'case': case}
            with output.open('a') as stream:
                stream.write(json.dumps(record, sort_keys=True) + '\n')
            print(f'round {round_index+1}/10 {label} {arm} '
                  f'{case["kernel_median_us"]:.3f} us', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('defaults', 'select', 'compare'))
    parser.add_argument('--defaults', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--suite', choices=('small', 'large'), default='small')
    args = parser.parse_args()
    if args.phase == 'defaults':
        defaults(args.suite)
    elif args.phase == 'select':
        select(args.defaults, args.output, args.suite)
    else:
        compare(args.selection, args.suite)


if __name__ == '__main__':
    main()
