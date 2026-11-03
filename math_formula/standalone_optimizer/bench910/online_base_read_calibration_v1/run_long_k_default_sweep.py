#!/usr/bin/env python3
"""Measure source default selection across the K=65535 A-conversion boundary."""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
sys.path.insert(0, str(ROOT))
import splitk_family_experiment as bench

OUT = ROOT / 'cases/online_long_k_default_sweep_v1'
PACKAGE = ROOT / 'ops_nn_master_fp16_package'
SOURCE = '/home/xianfei/dev/ops-nn-sribd-master-bf16/matmul/mat_mul_v3/'
SHAPES = [(2560, 1024, k) for k in (65408, 65536, 73728)]
original_environment = bench.environment


def environment(arm, shape):
    env = original_environment('default', shape)
    for name in ('MMV3_RESEARCH_SPLITK_FAMILY', 'MMV3_MIP_DECISIONS_FILE',
                 'MMV3_ONLINE_TILE'):
        env.pop(name, None)
    env.pop('MMV3_ONLINE_SELECTOR', None)
    env.pop('MMV3_ONLINE_ROW_READ_PROFILE', None)
    env['BENCH_PROFILE_KERNEL_TYPE'] = 'MIX_AIC'
    return env


bench.environment = environment
bench.OUT = OUT
bench.PACKAGE = PACKAGE
bench.SOURCE = SOURCE
bench.DEVICE = 4
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for shape in SHAPES:
    idle = OUT / 'idle_checks' / bench.label(shape)
    idle.mkdir(parents=True, exist_ok=True)
    bench.require_idle(idle, 'before_audit')
    audit = bench.audit_one('default', shape)
    bench.require_idle(idle, 'after_audit')
    if audit['returncode'] != 0 or audit.get('event', {}).get('max_relative_error', 1) > .02:
        raise RuntimeError(f'audit failed: {shape}: {audit}')
    if audit['tiling']['tilingEnableSplitCore'] != 3:
        raise RuntimeError(f'not deterministic Split-K: {shape}: {audit["tiling"]}')
    times = [bench.time_one('default', shape, round_index, 8)['kernel_median_us']
             for round_index in range(2)]
    row = {'shape': list(shape), 'tiling': audit['tiling'],
           'max_relative_error': audit['event']['max_relative_error'],
           'round_medians_us': times, 'median_us': statistics.median(times)}
    rows.append(row)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('summary', bench.label(shape), row['median_us'], flush=True)
