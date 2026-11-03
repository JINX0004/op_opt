#!/usr/bin/env python3
"""Compare old and revised online mode-5 selector packages on device 5."""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
sys.path.insert(0, str(ROOT))
import splitk_family_experiment as bench

OUT = ROOT / 'cases/online_mode5_model_compare_v1'
SHAPES = ((49152, 384, 1536), (65536, 384, 1536),
          (81920, 384, 1536))
PACKAGES = {
    'old': ROOT / 'ops_nn_mip_grouped_package',
    'revised': ROOT / 'ops_nn_mip_mode5_package',
}
SOURCE = '/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/'
original_environment = bench.environment
profile_modes = {}


def environment(arm, shape):
    env = original_environment('default', shape)
    for name in ('MMV3_RESEARCH_SPLITK_FAMILY', 'MMV3_MIP_DECISIONS_FILE',
                 'MMV3_ONLINE_TILE', 'MMV3_MODE5_TILE_EXPERIMENT'):
        env.pop(name, None)
    env['MMV3_ONLINE_SELECTOR'] = '1'
    env['MMV3_ONLINE_ROW_READ_PROFILE'] = '1'
    mode = profile_modes.get((arm, shape))
    if mode is not None:
        env['BENCH_PROFILE_KERNEL_TYPE'] = (
            'KERNEL_AICORE' if mode == 0 else 'MIX_AIC')
    return env


bench.environment = environment


def configure(arm):
    bench.PACKAGE = PACKAGES[arm]
    bench.SOURCE = SOURCE


def audit(arm, shape):
    configure(arm)
    idle = OUT / 'idle_checks' / bench.label(shape) / arm
    idle.mkdir(parents=True, exist_ok=True)
    bench.require_idle(idle, 'before_audit')
    result = bench.audit_one(arm, shape)
    bench.require_idle(idle, 'after_audit')
    error = result.get('event', {}).get('max_relative_error', 1)
    if result['returncode'] != 0 or error > .02:
        raise RuntimeError(f'failed {shape} {arm} audit: {result}')
    profile_modes[(arm, shape)] = result['tiling']['tilingEnableSplitCore']
    return result


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('audit', 'timing'))
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--m', type=int, choices=(49152, 65536, 81920))
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if args.m is not None:
        OUT = OUT / str(args.m)
    shapes = tuple(s for s in SHAPES if args.m is None or s[0] == args.m)
    bench.OUT = OUT
    bench.DEVICE = args.device
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for shape in shapes:
        row = {'shape': shape, 'arms': {}}
        for arm in PACKAGES:
            result = audit(arm, shape)
            row['arms'][arm] = {
                'tiling': result['tiling'],
                'max_relative_error': result['event']['max_relative_error'],
                'times_us': [],
            }
        rows.append(row)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    if args.phase == 'audit':
        for row in rows:
            print(row['shape'], [(arm,
                row['arms'][arm]['tiling']['tilingEnableSplitCore'],
                row['arms'][arm]['tiling']['singleCoreM'],
                row['arms'][arm]['tiling']['singleCoreN'])
                for arm in PACKAGES], flush=True)
        return
    for round_index in range(args.rounds):
        for row in rows:
            shape = tuple(row['shape'])
            order = ('old', 'revised') if round_index % 2 == 0 else \
                    ('revised', 'old')
            for arm in order:
                configure(arm)
                case = bench.time_one(arm, shape, round_index, 8)
                row['arms'][arm]['times_us'].append(case['kernel_median_us'])
                (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        for arm in PACKAGES:
            row['arms'][arm]['median_us'] = statistics.median(
                row['arms'][arm]['times_us'])
        print(row['shape'], row['arms']['old']['median_us'],
              row['arms']['revised']['median_us'], flush=True)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
