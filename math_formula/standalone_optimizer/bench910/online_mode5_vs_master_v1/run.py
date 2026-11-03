#!/usr/bin/env python3
"""Fresh FP16 comparison: clean master versus revised mode-5 online selector."""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
sys.path.insert(0, str(ROOT))
import splitk_family_experiment as bench

OUT_ROOT = ROOT / 'cases/online_mode5_vs_master_fp16_v1'
PACKAGES = {
    'default': ROOT / 'ops_nn_master_fp16_package',
    'online': ROOT / 'ops_nn_mip_mode5_package',
}
SOURCES = {
    'default': '/home/xianfei/dev/ops-nn-sribd-master-bf16/matmul/mat_mul_v3/',
    'online': '/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/',
}
SHAPES = [
    (16, 16, 102400), (32, 16, 102400), (64, 16, 102400),
    (128, 16, 65536), (256, 16, 32768), (16, 32, 102400),
    (32, 32, 102400), (64, 32, 65536),
    (64, 64, 65536), (128, 128, 32768), (1792, 768, 27392),
    (2048, 768, 32768), (2560, 1024, 36864),
    (1024, 2560, 36864), (2048, 2048, 4096),
    (4096, 4096, 8192), (4096, 4096, 32768),
    (49152, 384, 1536), (2560, 1024, 73728),
    (8192, 4096, 16384),
    (65536, 384, 1536), (81920, 384, 1536),
]

profile_mode = [None]
original_environment = bench.environment


def environment(arm, shape):
    env = original_environment('default', shape)
    for name in ('MMV3_RESEARCH_SPLITK_FAMILY', 'MMV3_MIP_DECISIONS_FILE',
                 'MMV3_ONLINE_SELECTOR', 'MMV3_ONLINE_ROW_READ_PROFILE',
                 'MMV3_ONLINE_TILE'):
        env.pop(name, None)
    if arm == 'online':
        env['MMV3_ONLINE_SELECTOR'] = '1'
        env['MMV3_ONLINE_ROW_READ_PROFILE'] = '1'
    if profile_mode[0] is not None:
        env['BENCH_PROFILE_KERNEL_TYPE'] = (
            'KERNEL_AICORE' if profile_mode[0] == 0 and shape[2] <= 65535
            else 'MIX_AIC')
    return env


bench.environment = environment

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, required=True)
parser.add_argument('--indices', required=True,
                    help='comma-separated zero-based indices in SHAPES')
parser.add_argument('--tag', help='unique output folder label (default: device ID)')
args = parser.parse_args()
indices = [int(x) for x in args.indices.split(',')]
if not indices or len(set(indices)) != len(indices) or any(
        i < 0 or i >= len(SHAPES) for i in indices):
    parser.error('indices must be distinct valid SHAPES positions')
OUT = OUT_ROOT / (args.tag or f'device{args.device}')
bench.OUT = OUT
bench.DEVICE = args.device
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'indices.json').write_text(json.dumps(indices) + '\n')


def configure(arm, mode=None):
    bench.PACKAGE = PACKAGES[arm]
    bench.SOURCE = SOURCES[arm]
    profile_mode[0] = mode


rows = []
for shape_index in indices:
    shape = SHAPES[shape_index]
    row = {'shape_index': shape_index, 'shape': list(shape), 'arms': {}}
    try:
        for arm in ('default', 'online'):
            configure(arm)
            idle = OUT / 'idle_checks' / bench.label(shape) / arm
            idle.mkdir(parents=True, exist_ok=True)
            bench.require_idle(idle, 'before_audit')
            audit = bench.audit_one(arm, shape)
            bench.require_idle(idle, 'after_audit')
            error = audit.get('event', {}).get('max_relative_error', 1)
            if audit['returncode'] != 0 or error > .02:
                raise RuntimeError(f'failed audit: {shape} {arm}: {audit}')
            row['arms'][arm] = {
                'mode': audit['tiling']['tilingEnableSplitCore'],
                'tiling': audit['tiling'],
                'max_relative_error': error,
                'times_us': [],
            }
        changed = row['arms']['default']['mode'] != row['arms']['online']['mode']
        # Repeat every changed-family decision and all three mode-5
        # shapes; the other equal-family cases retain the earlier suite's
        # one-round protocol.
        rounds = 2 if changed or shape_index in (17, 20, 21) else 1
        for round_index in range(rounds):
            order = ('default', 'online') if round_index == 0 else ('online', 'default')
            for arm in order:
                configure(arm, row['arms'][arm]['mode'])
                result = bench.time_one(arm, shape, round_index, 8)
                row['arms'][arm]['times_us'].append(result['kernel_median_us'])
        for arm in ('default', 'online'):
            row['arms'][arm]['median_us'] = statistics.median(row['arms'][arm]['times_us'])
        row['speedup_default_over_online'] = (
            row['arms']['default']['median_us'] / row['arms']['online']['median_us'])
        print('summary', bench.label(shape),
              'modes', row['arms']['default']['mode'], row['arms']['online']['mode'],
              'us', row['arms']['default']['median_us'], row['arms']['online']['median_us'],
              'speedup', row['speedup_default_over_online'], flush=True)
    except Exception as exc:
        row['error'] = str(exc)
        print('failed', bench.label(shape), repr(exc), flush=True)
    rows.append(row)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
