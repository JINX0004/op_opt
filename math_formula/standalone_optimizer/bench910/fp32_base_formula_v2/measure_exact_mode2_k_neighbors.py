#!/usr/bin/env python3
"""Hold the exact FP32 mode-2 tile fixed across nearby K row pitches."""

import csv
import importlib.util
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_source_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = ROOT / 'cases/fp32_base_formula_v2/exact_mode2_k_neighbors'
runner.PACKAGES['proposed'] = ROOT / 'ops_nn_fp32_mode2_k_neighbors_package'

original_environment = runner.environment


def environment(arm, mode, log_level):
    env = original_environment('research_default', mode, log_level)
    package = runner.PACKAGES['proposed']
    env['BENCH_CUSTOM_OPP_PATH'] = str(package)
    env['ASCEND_CUSTOM_OPP_PATH'] = str(package)
    env['LD_LIBRARY_PATH'] = str(package / 'op_api/lib') + ':' + runner.CANN_ENV.get('LD_LIBRARY_PATH', '')
    env['MMV3_RESEARCH_SPLITK_FAMILY'] = 'single'
    env['MMV3_RESEARCH_MODE2_EXACT_TILE'] = '1'
    return env


runner.environment = environment

METRICS = ('aicore_time(us)', 'aic_mac_time(us)', 'aic_mte1_time(us)',
           'aic_mte2_time(us)', 'aic_fixpipe_time(us)', 'aic_scalar_time(us)')
KS = (32512, 32768, 33024)


def counters(profile):
    directory = Path(profile['profile']['profile_csv']).parent
    matches = list(directory.glob('op_summary_*.csv'))
    if len(matches) != 1:
        raise RuntimeError((directory, matches))
    rows = [row for row in csv.DictReader(matches[0].open())
            if row['OP Type'] == 'MatMulV3'][-5:]
    if len(rows) != 5:
        raise RuntimeError((directory, len(rows)))
    return {metric: statistics.median(float(row[metric]) for row in rows)
            for metric in METRICS}


def one_geometry(n, device):
    audits = {}
    for k in KS:
        shape = [2560, n, k]
        audit = runner.audit(k, shape, device, 'proposed', 2)
        t = audit['tiling']
        expected = (2, 128, n, 256, 128, 128, 64, 2, 1, 4, 4, 8, 8, 20)
        actual = tuple(t[key] for key in
            ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN', 'singleCoreK',
             'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb',
             'depthA1', 'depthB1', 'block_dim'))
        if actual != expected or audit['event']['max_relative_error'] > .02:
            raise RuntimeError((shape, actual, expected, audit['event']['max_relative_error']))
        audits[k] = audit
    measurements = {k: [] for k in KS}
    for round_index, order in enumerate((KS, tuple(reversed(KS)))):
        for k in order:
            shape = [2560, n, k]
            profile = runner.profile(k, shape, device, 'proposed', 2,
                                     round_index, audits[k])
            row = {'round': round_index, 'profile': profile,
                   'counters': counters(profile)}
            measurements[k].append(row)
            print(n, k, round_index, profile['profile']['median_us'], flush=True)
    return [{'shape': [2560, n, k], 'device': device,
             'audit': audits[k], 'measurements': measurements[k],
             'median_us': statistics.median(row['profile']['profile']['median_us']
                                            for row in measurements[k]),
             'median_counters': {metric: statistics.median(row['counters'][metric]
                                                   for row in measurements[k])
                                 for metric in METRICS}}
            for k in KS]


def main():
    library = runner.PACKAGES['proposed'] / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
    build_hash = (ROOT / 'fp32_base_formula_v2/exact_mode2_k_neighbors/build_hashes.txt').read_text().splitlines()[-1].split()[0]
    if runner.sha(library) != build_hash:
        raise RuntimeError('neighbor package host library changed')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = sorted((row for group in pool.map(lambda item: one_geometry(*item),
                         ((768, 6), (1024, 7))) for row in group),
                         key=lambda row: (row['shape'][1], row['shape'][2]))
    runner.write(runner.OUT / 'records.json', results)
    print(json.dumps([{'shape': row['shape'], 'median_us': row['median_us'],
                       'median_counters': row['median_counters']}
                      for row in results], indent=2), flush=True)


if __name__ == '__main__':
    main()
