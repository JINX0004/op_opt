#!/usr/bin/env python3
"""Paired device-4 probe of the selector's mode-2 tile for one FP32 shape."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
RUNNER = ROOT / 'fp32_source_base_100_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_source_base_100_runner', RUNNER)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
runner = experiment.runner
package = ROOT / 'ops_nn_fp32_mode2_model_tile_package'
host = package / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
runner.PACKAGES['online'] = package
runner.HASHES['online'] = hashlib.sha256(host.read_bytes()).hexdigest()
runner.OUT = ROOT / 'cases/fp32_mode2_model_tile_2560x768x32768'
old_environment = runner.environment

def environment(arm, model_mode, log_level):
    env = old_environment(arm, model_mode, log_level)
    if arm == 'online':
        env['MMV3_FORCE_MODE2_MODEL_TILE'] = '1'
    return env

runner.environment = environment
experiment.experiment.check_package()
shape = [2560, 768, 32768]
index = 36
device = 4
base = runner.OUT / f'device{device}' / f'{index:03d}_{runner.label(shape)}'
record_path = base / 'record.json'
if record_path.exists():
    print(record_path.read_text())
    raise SystemExit(0)
audits = {arm: runner.audit(index, shape, device, arm, 0)
          for arm in ('master', 'online')}
master = audits['master']['tiling']
online = audits['online']['tiling']
expected = {'tilingEnableSplitCore': 2, 'singleCoreM': 128,
            'singleCoreN': 768, 'singleCoreK': 256,
            'baseM': 128, 'baseN': 128, 'baseK': 64,
            'stepM': 2, 'stepN': 1}
for field, value in expected.items():
    if online[field] != value:
        raise RuntimeError(f'forced emitted tile mismatch: {field}: {online[field]} != {value}')
print('AUDIT', {k: online[k] for k in expected},
      'max_relative_error', audits['online']['event']['max_relative_error'], flush=True)
measurements = []
for round_index, order in enumerate((('master', 'online'), ('online', 'master'))):
    for arm in order:
        item = runner.profile(index, shape, device, arm, 0, round_index, audits[arm])
        measurements.append(item)
        print('PROFILE', round_index, arm, item['profile']['median_us'], flush=True)
record = {'index': index, 'shape': shape, 'device': device,
          'host_sha256': runner.HASHES['online'],
          'audits': audits, 'measurements': measurements}
runner.write(record_path, record)
print('RECORD', record_path, flush=True)
