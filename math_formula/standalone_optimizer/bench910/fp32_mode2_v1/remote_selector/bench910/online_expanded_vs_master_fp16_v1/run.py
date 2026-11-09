#!/usr/bin/env python3
"""Expanded held-out FP16 comparison: clean master versus online selector."""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
sys.path.insert(0, str(ROOT))
import splitk_family_experiment as bench

OUT_ROOT = ROOT / 'cases/online_expanded_vs_master_fp16_v1'
PACKAGES = {
    'default': ROOT / 'ops_nn_master_fp16_package',
    'online': ROOT / 'ops_nn_mip_mode5_package',
}
SOURCES = {
    'default': '/home/xianfei/dev/ops-nn-sribd-master-bf16/matmul/mat_mul_v3/',
    'online': '/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/',
}
SHAPES = [
    # 0-15: narrow output and the K=65535 A-conversion boundary.
    (16, 16, 65408), (16, 16, 65536), (16, 16, 131072),
    (32, 64, 102400), (64, 16, 65536), (64, 64, 102400),
    (128, 32, 65536), (128, 64, 65536), (256, 64, 65536),
    (256, 128, 32768), (384, 128, 65536), (512, 128, 73728),
    (128, 256, 32768), (128, 512, 16384), (512, 256, 32768),
    (256, 512, 16384),
    # 16-35: source family/path transition, including M/N swaps.
    (1536, 768, 24576), (1792, 768, 32768), (2048, 768, 27392),
    (2304, 768, 32768), (2560, 768, 32768), (2048, 1024, 32768),
    (2304, 1024, 36864), (2816, 1024, 36864),
    (3072, 1024, 36864), (2560, 1024, 32768),
    (2560, 1024, 49152), (2560, 1024, 65536),
    (3072, 1024, 73728), (1024, 2304, 36864),
    (1024, 2816, 36864), (768, 2048, 32768),
    (2048, 1536, 32768), (2048, 2048, 27392),
    (2048, 3072, 32768), (3072, 2048, 32768),
    # 36-47: mode-5 channel and large-M geometry.
    (16384, 256, 1024), (32768, 256, 1536),
    (49152, 256, 1536), (65536, 256, 1536),
    (98304, 256, 1536), (32768, 384, 1024),
    (32768, 384, 1536), (32768, 384, 2048),
    (98304, 384, 1536), (49152, 512, 1536),
    (65536, 512, 2048), (81920, 512, 1536),
    # 48-63: large M/N, both orientations, and larger K.
    (2048, 4096, 4096), (4096, 2048, 4096),
    (4096, 8192, 4096), (8192, 4096, 4096),
    (4096, 4096, 16384), (8192, 8192, 8192),
    (8192, 4096, 32768), (4096, 8192, 16384),
    (10240, 3072, 8192), (3072, 10240, 8192),
    (16384, 2048, 4096), (2048, 16384, 4096),
    (12288, 4096, 8192), (4096, 12288, 8192),
    (8192, 8192, 16384), (16384, 4096, 8192),
]

profile_mode = [None]
profile_conversion = [None]
disable_online_selector = [False]
original_environment = bench.environment


def environment(arm, shape):
    env = original_environment('default', shape)
    for name in ('MMV3_RESEARCH_SPLITK_FAMILY', 'MMV3_MIP_DECISIONS_FILE',
                 'MMV3_ONLINE_SELECTOR', 'MMV3_ONLINE_ROW_READ_PROFILE',
                 'MMV3_ONLINE_TILE'):
        env.pop(name, None)
    if arm == 'online' and not disable_online_selector[0]:
        env['MMV3_ONLINE_SELECTOR'] = '1'
        env['MMV3_ONLINE_ROW_READ_PROFILE'] = '1'
    if profile_mode[0] is not None:
        env['BENCH_PROFILE_KERNEL_TYPE'] = (
            'KERNEL_AICORE' if profile_mode[0] == 0 and
            not any(profile_conversion[0] or ())
            else 'MIX_AIC')
    if profile_conversion[0] is not None:
        env['BENCH_EXPECT_ND2NZ_A'] = str(profile_conversion[0][0])
        env['BENCH_EXPECT_ND2NZ_B'] = str(profile_conversion[0][1])
    return env


bench.environment = environment

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, required=True)
parser.add_argument('--indices', required=True,
                    help='comma-separated zero-based indices in SHAPES')
parser.add_argument('--tag', help='unique output folder label (default: device ID)')
parser.add_argument('--rounds', type=int,
                    help='override timing rounds per shape (for diagnostic repeats)')
parser.add_argument('--disable-online-selector', action='store_true',
                    help='compare the online package with its selector disabled')
parser.add_argument('--shapes-json', type=Path,
                    help='optional supplemental shape list, in M,N,K order')
parser.add_argument('--out-root', type=Path,
                    help='optional root for a supplemental comparison')
parser.add_argument('--online-package', type=Path,
                    help='optional patched online package for a controlled comparison')
parser.add_argument('--online-source',
                    help='source path expected for --online-package')
parser.add_argument('--default-package', type=Path,
                    help='optional package for the selector-disabled arm')
parser.add_argument('--default-source',
                    help='source path expected for --default-package')
args = parser.parse_args()
disable_online_selector[0] = args.disable_online_selector
if args.online_package:
    PACKAGES['online'] = args.online_package
if args.online_source:
    SOURCES['online'] = args.online_source
if args.default_package:
    PACKAGES['default'] = args.default_package
if args.default_source:
    SOURCES['default'] = args.default_source
if args.shapes_json:
    SHAPES = [tuple(map(int, shape)) for shape in
              json.loads(args.shapes_json.read_text())]
    if any(len(shape) != 3 or min(shape) <= 0 for shape in SHAPES):
        parser.error('--shapes-json must contain positive M,N,K triples')
if args.out_root:
    OUT_ROOT = args.out_root
indices = [int(x) for x in args.indices.split(',')]
if not indices or len(set(indices)) != len(indices) or any(
        i < 0 or i >= len(SHAPES) for i in indices):
    parser.error('indices must be distinct valid SHAPES positions')
OUT = OUT_ROOT / (args.tag or f'device{args.device}')
bench.OUT = OUT
bench.DEVICE = args.device
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'indices.json').write_text(json.dumps(indices) + '\n')


def configure(arm, mode=None, conversion=None):
    bench.PACKAGE = PACKAGES[arm]
    bench.SOURCE = SOURCES[arm]
    profile_mode[0] = mode
    profile_conversion[0] = conversion


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
        changed = any(
            row['arms']['default']['tiling'].get(field) !=
            row['arms']['online']['tiling'].get(field)
            for field in bench.GEOMETRY_FIELDS)
        # Repeat changed decisions and mode-5 controls in reversed order.
        # Identical family+tiling cases need one timing round for drift checks.
        rounds = args.rounds or (2 if changed or row['arms']['default']['mode'] == 5 else 1)
        for round_index in range(rounds):
            order = ('default', 'online') if round_index == 0 else ('online', 'default')
            for arm in order:
                tiling = row['arms'][arm]['tiling']
                configure(arm, row['arms'][arm]['mode'],
                          (tiling['nd2nzA'], tiling['nd2nzB']))
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
