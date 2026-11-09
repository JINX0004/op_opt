#!/usr/bin/env python3
"""Smoke-test the actual FP32 host gate with source and online selection."""

import importlib.util
import json
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_100_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

runner.OUT = ROOT / 'cases/fp32_online_new_profile_smoke_v1'
package = ROOT / 'ops_nn_fp32_online_profile_package'
branch_source = '/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/'
for arm in ('research_default', 'proposed'):
    runner.PACKAGES[arm] = package
    runner.SOURCES[arm] = branch_source
original_environment = runner.environment


def environment(arm, mode, log_level):
    env = original_environment('research_default', mode, log_level)
    env.pop('MMV3_RESEARCH_SPLITK_FAMILY', None)
    if arm == 'proposed':
        env['MMV3_ONLINE_SELECTOR'] = '1'
    return env


runner.environment = environment


def main():
    # Source uses BASE at index 97. The new profile selects deterministic.
    index, shape, device = 97, [4096, 64, 27328], 4
    audits = {arm: runner.audit(index, shape, device, arm, mode)
              for arm, mode in (('research_default', 0), ('proposed', 3))}
    if audits['research_default']['tiling']['tilingEnableSplitCore'] != 0:
        raise RuntimeError('source did not select BASE')
    if audits['proposed']['tiling']['tilingEnableSplitCore'] != 3:
        raise RuntimeError('online did not select deterministic Split-K')
    measurements = [runner.profile(index, shape, device, arm, mode, 0,
                                   audits[arm])
                    for arm, mode in (('research_default', 0), ('proposed', 3))]
    for item in measurements:
        print(item['arm'], item['profile']['median_us'], flush=True)
    runner.write(runner.OUT / 'record.json',
                 {'shape': shape, 'device': device, 'audits': audits,
                  'measurements': measurements})


if __name__ == '__main__':
    main()
