#!/usr/bin/env python3
"""Paired device-4 online-family audit: generic vs row-aware BASE reads."""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
sys.path.insert(0, str(ROOT))
import splitk_family_experiment as bench

OUT = ROOT / 'cases/online_narrow_b_fp16_v1'
PACKAGE = ROOT / 'ops_nn_mip_narrow_package'
SOURCE = '/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/'
SHAPES = [
    (32, 16, 102400),
    (64, 16, 102400),
    (128, 16, 65536),
    (256, 16, 32768),
    (16, 32, 102400),
    (32, 32, 102400),
    (64, 32, 65536),
]
profile_mode = [None]
original_environment = bench.environment


def environment(arm, shape):
    env = original_environment('default', shape)
    env.pop('MMV3_RESEARCH_SPLITK_FAMILY', None)
    env.pop('MMV3_ONLINE_TILE', None)
    env['MMV3_ONLINE_SELECTOR'] = '1'
    if arm == 'narrow':
        env['MMV3_ONLINE_ROW_READ_PROFILE'] = '1'
    else:
        env.pop('MMV3_ONLINE_ROW_READ_PROFILE', None)
    if profile_mode[0] is not None:
        env['BENCH_PROFILE_KERNEL_TYPE'] = (
            'KERNEL_AICORE' if profile_mode[0] == 0 and shape[2] <= 65535
            else 'MIX_AIC')
    return env


bench.environment = environment
bench.PACKAGE = PACKAGE
bench.SOURCE = SOURCE
bench.OUT = OUT
bench.DEVICE = 4
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for shape in SHAPES:
    row = {'shape': list(shape), 'arms': {}}
    for arm in ('generic', 'narrow'):
        profile_mode[0] = None
        idle = OUT / 'idle_checks' / bench.label(shape) / arm
        idle.mkdir(parents=True, exist_ok=True)
        bench.require_idle(idle, 'before_audit')
        audit = bench.audit_one(arm, shape)
        bench.require_idle(idle, 'after_audit')
        if audit['returncode'] != 0 or audit.get('event', {}).get('max_relative_error', 1) > .02:
            raise RuntimeError(f'audit failed: {shape} {arm}: {audit}')
        row['arms'][arm] = {'mode': audit['tiling']['tilingEnableSplitCore'],
                            'tiling': audit['tiling'], 'times_us': []}
    for round_index in range(2):
        order = ('generic', 'narrow') if round_index == 0 else ('narrow', 'generic')
        for arm in order:
            profile_mode[0] = row['arms'][arm]['mode']
            case = bench.time_one(arm, shape, round_index, 8)
            row['arms'][arm]['times_us'].append(case['kernel_median_us'])
    for arm in ('generic', 'narrow'):
        row['arms'][arm]['median_us'] = statistics.median(row['arms'][arm]['times_us'])
    row['speedup_generic_over_narrow'] = (
        row['arms']['generic']['median_us'] / row['arms']['narrow']['median_us'])
    rows.append(row)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('summary', bench.label(shape),
          'modes', row['arms']['generic']['mode'], row['arms']['narrow']['mode'],
          'us', row['arms']['generic']['median_us'], row['arms']['narrow']['median_us'],
          'speedup', row['speedup_generic_over_narrow'], flush=True)
