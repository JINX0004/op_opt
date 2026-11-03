#!/usr/bin/env python3
"""Run the frozen two-tile mode-5 comparison on another eligible M."""

import argparse
import json
import statistics
from pathlib import Path

import run_mode5_tile_experiment as experiment

ROOT = Path('/home/xianfei/solvebase_bench_20261025')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('audit', 'timing'))
    parser.add_argument('--m', type=int, required=True)
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if args.m not in (65536, 81920):
        parser.error('this held-out runner covers M=65536 or 81920')
    experiment.SHAPE = (args.m, 384, 1536)
    experiment.OUT = ROOT / 'cases/mode5_tile_heldout_v1' / str(args.m)
    experiment.PACKAGE = ROOT / 'ops_nn_mode5_tile_validation_package'
    experiment.TILES = {
        'large': (16 * ((args.m + 20 * 16 - 1) // (20 * 16)), 384),
        'small': (128, 128),
    }
    experiment.bench.DEVICE = args.device
    experiment.bench.OUT = experiment.OUT
    experiment.bench.PACKAGE = experiment.PACKAGE
    experiment.bench.SOURCE = experiment.SOURCE
    experiment.OUT.mkdir(parents=True, exist_ok=True)
    audits = {arm: experiment.audit(arm) for arm in experiment.TILES}
    if args.phase == 'audit':
        print(json.dumps({arm: {'tiling': row['tiling'],
                                'max_relative_error': row['event']['max_relative_error']}
                          for arm, row in audits.items()}, indent=2))
        return
    rows = []
    for round_index in range(args.rounds):
        for arm in (('large', 'small') if round_index % 2 == 0 else
                    ('small', 'large')):
            case = experiment.bench.time_one(arm, experiment.SHAPE,
                                             round_index, 8)
            rows.append({'round': round_index, 'arm': arm,
                         'kernel_median_us': case['kernel_median_us'],
                         'counters_us': experiment.counters(case),
                         'profile_csv': case['profile_csv']})
            (experiment.OUT / 'summary.json').write_text(
                json.dumps(rows, indent=2) + '\n')
    medians = {arm: statistics.median(r['kernel_median_us'] for r in rows
                                     if r['arm'] == arm)
               for arm in experiment.TILES}
    print('median_us', medians, 'small_over_large',
          medians['small'] / medians['large'], flush=True)


if __name__ == '__main__':
    main()
