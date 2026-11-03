#!/usr/bin/env python3
"""Paired source-backed FP16 MatMulV3 audit and kernel timing on 910B3."""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
sys.path.insert(0, str(ROOT))
import splitk_family_experiment as bench

OUT = ROOT / 'cases/online_vs_master_fp16_v1'
PACKAGES = {
    'master': ROOT / 'ops_nn_master_fp16_package',
    'online': ROOT / 'ops_nn_mip_branch_package',
}
SOURCES = {
    'master': '/home/xianfei/dev/ops-nn-sribd-master-bf16/matmul/mat_mul_v3/',
    'online': '/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/',
}
SHAPES = [
    (16, 16, 102400),
    (64, 64, 65536),
    (128, 128, 32768),
    (1792, 768, 27392),
    (2048, 768, 32768),
    (2560, 1024, 36864),
    (1024, 2560, 36864),
    (2048, 2048, 4096),
    (4096, 4096, 8192),
    (4096, 4096, 32768),
    (49152, 384, 1536),
    (2560, 1024, 73728),
    (8192, 4096, 16384),
]

profile_mode = [None]
original_environment = bench.environment


def environment(arm, shape):
    env = original_environment('default', shape)
    env.pop('MMV3_MIP_DECISIONS_FILE', None)
    env.pop('MMV3_RESEARCH_SPLITK_FAMILY', None)
    if arm == 'online':
        env['MMV3_ONLINE_SELECTOR'] = '1'
    else:
        env.pop('MMV3_ONLINE_SELECTOR', None)
    if profile_mode[0] is not None:
        env['BENCH_PROFILE_KERNEL_TYPE'] = (
            'KERNEL_AICORE' if profile_mode[0] == 0 and shape[2] <= 65535
            else 'MIX_AIC')
    return env


bench.environment = environment


def configure(arm, mode=None):
    bench.PACKAGE = PACKAGES[arm]
    bench.SOURCE = SOURCES[arm]
    profile_mode[0] = mode


def audit(arm, shape):
    configure(arm)
    check_dir = OUT / 'idle_checks' / bench.label(shape) / arm
    check_dir.mkdir(parents=True, exist_ok=True)
    bench.require_idle(check_dir, 'before_audit')
    result = bench.audit_one(arm, shape)
    bench.require_idle(check_dir, 'after_audit')
    return result


def timing(arm, shape, round_index, repetitions):
    audit_result = audit(arm, shape)
    if audit_result['returncode'] != 0 or 'tiling' not in audit_result:
        raise RuntimeError(f'audit failed: {arm} {shape}')
    mode = audit_result['tiling']['tilingEnableSplitCore']
    configure(arm, mode)
    return bench.time_one(arm, shape, round_index, repetitions)


def summarize(shapes, rounds):
    rows = []
    for shape in shapes:
        label = bench.label(shape)
        row = {'shape': list(shape)}
        for arm in ('master', 'online'):
            audit_path = OUT / 'audit' / label / arm / 'result.json'
            if audit_path.exists():
                result = json.loads(audit_path.read_text())
                row[f'{arm}_returncode'] = result['returncode']
                row[f'{arm}_mode'] = result.get('tiling', {}).get('tilingEnableSplitCore')
                row[f'{arm}_max_relative_error'] = result.get('event', {}).get('max_relative_error')
            samples = []
            for r in range(rounds):
                p = OUT / 'timing' / f'r{r}' / label / arm / 'case.json'
                if p.exists():
                    samples.append(json.loads(p.read_text())['kernel_median_us'])
            if samples:
                row[f'{arm}_round_medians_us'] = samples
                row[f'{arm}_median_us'] = statistics.median(samples)
        if row.get('master_median_us') and row.get('online_median_us'):
            row['speedup_master_over_online'] = (
                row['master_median_us'] / row['online_median_us'])
        rows.append(row)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print('summary', bench.label(row['shape']),
              'modes', row.get('master_mode'), row.get('online_mode'),
              'us', row.get('master_median_us'), row.get('online_median_us'),
              'speedup', row.get('speedup_master_over_online'), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('audit', 'timing', 'all'), default='all')
    parser.add_argument('--indices', default='all',
                        help='comma-separated zero-based SHAPES indices or all')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=12)
    parser.add_argument('--device', type=int, default=4)
    args = parser.parse_args()
    if args.rounds < 1 or args.repetitions < 1:
        parser.error('rounds and repetitions must be positive')
    indices = range(len(SHAPES)) if args.indices == 'all' else [
        int(x) for x in args.indices.split(',')]
    shapes = [SHAPES[i] for i in indices]
    OUT.mkdir(parents=True, exist_ok=True)
    bench.OUT = OUT
    bench.DEVICE = args.device
    (OUT / 'dataset.json').write_text(json.dumps(SHAPES, indent=2) + '\n')
    failures = []
    for shape in shapes:
        try:
            for arm in ('master', 'online'):
                if args.phase in ('audit', 'all'):
                    audit(arm, shape)
            if args.phase in ('timing', 'all'):
                for round_index in range(args.rounds):
                    arms = ('master', 'online') if round_index % 2 == 0 else ('online', 'master')
                    for arm in arms:
                        timing(arm, shape, round_index, args.repetitions)
        except Exception as error:
            failures.append({'shape': list(shape), 'error': str(error)})
            print('failed', bench.label(shape), repr(error), flush=True)
        summarize(SHAPES, args.rounds)
    (OUT / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n')


if __name__ == '__main__':
    main()
