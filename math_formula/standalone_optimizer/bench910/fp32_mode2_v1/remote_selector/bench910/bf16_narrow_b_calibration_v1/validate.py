#!/usr/bin/env python3
"""Paired source-backed BF16 boundary check: master, generic, narrow-row."""

import hashlib
import importlib.util
import json
from pathlib import Path

BENCH = Path('/home/xianfei/solvebase_bench_20261025')
BASELINE = BENCH / 'ops_nn_bf16_mode5_revised_package'
DIAGNOSTIC = BENCH / 'ops_nn_bf16_narrow_rows_diagnostic_package'
SOURCE_ONLINE = '/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/'
OUT = BENCH / 'cases/bf16_narrow_b_calibration_v1/boundary'
RUNNER = Path(__file__).resolve().parent.parent / 'bf16_100_vs_master_v1/run.py'
spec = importlib.util.spec_from_file_location('bf16_100_runner', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

runner.OUT = OUT
runner.PACKAGES['generic'] = BASELINE
runner.PACKAGES['online'] = DIAGNOSTIC
runner.SOURCE['generic'] = SOURCE_ONLINE
runner.HOST_SHA['generic'] = runner.HOST_SHA['online']
LIB = 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
runner.HOST_SHA['online'] = hashlib.sha256((DIAGNOSTIC / LIB).read_bytes()).hexdigest()


def env_for(arm):
    package = runner.PACKAGES[arm]
    env = runner.CANN_ENV.copy()
    for key in ('MMV3_ONLINE_SELECTOR', 'MMV3_ONLINE_BF16_EXPERIMENT',
                'MMV3_ONLINE_ROW_READ_PROFILE', 'MMV3_ONLINE_TILE',
                'MMV3_BF16_MODE5_TILE_PROBE'):
        env.pop(key, None)
    env.update(CANN_ROOT=runner.CANN, BENCH_API='v3cached',
               BENCH_CUSTOM_OPP_PATH=str(package),
               ASCEND_CUSTOM_OPP_PATH=str(package),
               MMV3_SKIP_LEGACY_OP_CACHE='1')
    env['LD_LIBRARY_PATH'] = str(package / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    if arm in ('generic', 'online'):
        env.update(MMV3_ONLINE_SELECTOR='1', MMV3_ONLINE_BF16_EXPERIMENT='1')
    return env


runner.env_for = env_for
SHAPES = [(16, 16, 65408), (32, 16, 65408),
          (16, 16, 65536), (16, 16, 102400),
          (256, 512, 16384)]


def main():
    runner.check_packages()
    for index, shape in enumerate(SHAPES):
        base = OUT / f'device4/{index:03d}_{runner.label(shape)}'
        record = base / 'record.json'
        if record.exists():
            print(f'skip {shape}', flush=True)
            continue
        print(f'start {shape}', flush=True)
        audits = {arm: runner.audit(index, shape, 4, arm)
                  for arm in ('master', 'generic', 'online')}
        rounds = 2 if index < 2 else 1
        measured = []
        for round_index in range(rounds):
            order = ('master', 'generic', 'online') if round_index == 0 \
                else ('online', 'generic', 'master')
            for arm in order:
                row = runner.timing(index, shape, 4, arm, round_index, audits[arm])
                measured.append(row)
                print(round_index, arm,
                      row['tiling']['tilingEnableSplitCore'],
                      row['kernel_median_us'], flush=True)
        runner.write(record, {'index': index, 'shape': shape, 'audits': audits,
                              'measurements': measured, 'device': 4})
        print(f'done {shape}', flush=True)


if __name__ == '__main__':
    main()
