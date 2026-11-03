#!/usr/bin/env python3
"""Paired source-versus-small mode-5 tile experiment on one idle 910B3."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
sys.path.insert(0, str(ROOT))
import splitk_family_experiment as bench

SHAPE = (49152, 384, 1536)
OUT = ROOT / 'cases/mode5_tile_experiment_v1'
PACKAGE = ROOT / 'ops_nn_mode5_tile_experiment_package'
SOURCE = '/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/'
TILES = {'large': (2464, 384), 'small': (128, 128)}
COUNTERS = ('Task Duration(us)', 'aicore_time(us)', 'aic_mac_time(us)',
            'aic_mte1_time(us)', 'aic_mte2_time(us)',
            'aic_fixpipe_time(us)', 'aic_scalar_time(us)')

original_environment = bench.environment


def environment(arm, shape):
    env = original_environment('default', shape)
    for name in ('MMV3_RESEARCH_SPLITK_FAMILY', 'MMV3_MIP_DECISIONS_FILE',
                 'MMV3_ONLINE_SELECTOR', 'MMV3_ONLINE_ROW_READ_PROFILE',
                 'MMV3_ONLINE_TILE'):
        env.pop(name, None)
    env['MMV3_MODE5_TILE_EXPERIMENT'] = 's' if arm == 'small' else 'l'
    env['BENCH_PROFILE_KERNEL_TYPE'] = 'MIX_AIC'
    return env


bench.environment = environment


def audit(arm):
    idle = OUT / 'idle_checks' / arm
    idle.mkdir(parents=True, exist_ok=True)
    bench.require_idle(idle, 'before_audit')
    result = bench.audit_one(arm, SHAPE)
    bench.require_idle(idle, 'after_audit')
    tiling = result.get('tiling', {})
    expected = TILES[arm]
    observed = (tiling.get('singleCoreM'), tiling.get('singleCoreN'))
    error = result.get('event', {}).get('max_relative_error', 1.0)
    if result['returncode'] != 0 or error > 0.02 or observed != expected or \
            tiling.get('tilingEnableSplitCore') != 5 or \
            tiling.get('singleCoreK') != 384 or \
            tiling.get('block_dim') != 20:
        raise RuntimeError(f'failed {arm} audit: geometry={observed}, '
                           f'mode={tiling.get("tilingEnableSplitCore")}, '
                           f'error={error}, rc={result["returncode"]}')
    return result


def counters(case):
    source_dir = Path(case['profile_csv']).parent
    summary_files = list(source_dir.glob('op_summary_*.csv'))
    if len(summary_files) != 1:
        raise RuntimeError(f'expected one op_summary CSV in {source_dir}')
    with summary_files[0].open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if row['OP Type'] == 'MatMulV3'][-8:]
    if len(rows) != 8:
        raise RuntimeError(f'expected eight MatMulV3 tasks in {summary_files[0]}')
    values = {key: statistics.median(float(row[key]) for row in rows)
              for key in COUNTERS}
    if abs(values['Task Duration(us)'] - case['kernel_median_us']) > 0.01:
        raise RuntimeError('op summary and task time disagree')
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('audit', 'timing'))
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    bench.DEVICE = args.device
    bench.OUT = OUT
    bench.PACKAGE = PACKAGE
    bench.SOURCE = SOURCE
    OUT.mkdir(parents=True, exist_ok=True)

    audits = {arm: audit(arm) for arm in TILES}
    if args.phase == 'audit':
        print(json.dumps({arm: {'tiling': row['tiling'],
                                'max_relative_error': row['event']['max_relative_error']}
                          for arm, row in audits.items()}, indent=2))
        return

    rows = []
    for round_index in range(args.rounds):
        for arm in (('large', 'small') if round_index % 2 == 0 else
                    ('small', 'large')):
            case = bench.time_one(arm, SHAPE, round_index, 8)
            rows.append({'round': round_index, 'arm': arm,
                         'kernel_median_us': case['kernel_median_us'],
                         'counters_us': counters(case),
                         'profile_csv': case['profile_csv']})
            (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    medians = {arm: statistics.median(r['kernel_median_us'] for r in rows
                                     if r['arm'] == arm) for arm in TILES}
    print('median_us', medians, 'large_over_small',
          medians['large'] / medians['small'], flush=True)


if __name__ == '__main__':
    main()
