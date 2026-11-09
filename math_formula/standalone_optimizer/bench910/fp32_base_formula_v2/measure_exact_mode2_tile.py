#!/usr/bin/env python3
"""Compare clean master, source mode 2, and the exact model mode-2 tile."""

import importlib.util
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_source_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = ROOT / 'cases/fp32_base_formula_v2/exact_mode2_tile'
runner.PACKAGES['proposed'] = ROOT / 'ops_nn_fp32_mode2_exact_tile_package'
runner.HASHES['proposed'] = runner.sha(
    runner.PACKAGES['proposed'] /
    'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so')

original_environment = runner.environment


def environment(arm, mode, log_level):
    if arm == 'master':
        return original_environment(arm, mode, log_level)
    env = original_environment('research_default', mode, log_level)
    env['MMV3_RESEARCH_SPLITK_FAMILY'] = 'single'
    if arm == 'proposed':
        package = runner.PACKAGES['proposed']
        env['BENCH_CUSTOM_OPP_PATH'] = str(package)
        env['ASCEND_CUSTOM_OPP_PATH'] = str(package)
        env['LD_LIBRARY_PATH'] = str(package / 'op_api/lib') + ':' + runner.CANN_ENV.get('LD_LIBRARY_PATH', '')
        env['MMV3_RESEARCH_MODE2_EXACT_TILE'] = '1'
    return env


runner.environment = environment

CASES = ((10, [2560, 768, 32768], 6, (128, 768, 256)),
         (15, [2560, 1024, 32768], 7, (128, 1024, 256)))


def check_package_hashes():
    for arm, package in runner.PACKAGES.items():
        library = package / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
        if runner.sha(library) != runner.HASHES[arm]:
            raise RuntimeError(f'{arm} package host library changed')
    build_hashes = (ROOT / 'fp32_base_formula_v2/exact_mode2_tile/build_hashes.txt').read_text().splitlines()
    if runner.HASHES['proposed'] != build_hashes[-1].split()[0]:
        raise RuntimeError('exact tile package does not match build output')


def one_case(case, timing):
    index, shape, device, expected = case
    audits = {arm: runner.audit(index, shape, device, arm, 2)
              for arm in ('master', 'research_default', 'proposed')}
    master = audits['master']['tiling']
    baseline = audits['research_default']['tiling']
    exact = audits['proposed']['tiling']
    if master['tilingEnableSplitCore'] != 3 or baseline['tilingEnableSplitCore'] != 2:
        raise RuntimeError(f'wrong control family at {index}')
    fields = ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN', 'singleCoreK',
              'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb',
              'depthA1', 'depthB1', 'block_dim')
    observed = tuple(exact[key] for key in fields)
    wanted = (2, *expected, 128, 128, 64, 2, 1, 4, 4, 8, 8, 20)
    if observed != wanted:
        raise RuntimeError(f'exact tile mismatch at {index}: {observed} != {wanted}')
    if any(audits[arm]['event']['max_relative_error'] > .02 for arm in audits):
        raise RuntimeError(f'output mismatch at {index}')
    measurements = []
    if timing:
        for round_index, order in enumerate((('master', 'research_default', 'proposed'),
                                             ('proposed', 'research_default', 'master'))):
            for arm in order:
                row = runner.profile(index, shape, device, arm, 2, round_index, audits[arm])
                measurements.append(row)
                print(index, arm, round_index, row['profile']['median_us'], flush=True)
    return {'index': index, 'shape': shape, 'device': device,
            'expected_model_tile': expected, 'audits': audits,
            'measurements': measurements}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('audit', 'timing'):
        raise SystemExit('usage: measure_exact_mode2_tile.py audit|timing')
    check_package_hashes()
    timing = sys.argv[1] == 'timing'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda case: one_case(case, timing), CASES))
    runner.write(runner.OUT / 'records.json', results)
    summary = []
    for row in results:
        timings = {arm: statistics.median(item['profile']['median_us']
                    for item in row['measurements'] if item['arm'] == arm)
                   for arm in ('master', 'research_default', 'proposed')} if timing else {}
        summary.append({'index': row['index'], 'shape': row['shape'],
                        'source_tiles': {arm: row['audits'][arm]['tiling']
                                         for arm in row['audits']},
                        'median_kernel_us': timings})
    runner.write(runner.OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
