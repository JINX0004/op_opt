#!/usr/bin/env python3
"""Reverse-order, same-device FP32 BASE K-boundary verification."""

import importlib.util
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = ROOT / 'cases/fp32_base_formula_v2/same_device'

SERIES = ((4, 2560, 768, (32512, 32768, 33024)),
          (5, 2560, 1024, (48896, 49152, 49408)))


def run_series(series):
    device, m, n, ks = series
    audits = {}
    for k in ks:
        shape = [m, n, k]
        audits[k] = runner.audit(k, shape, device, 'proposed', 0)
    results = {}
    for round_index in range(2):
        for k in (ks if round_index == 0 else tuple(reversed(ks))):
            shape = [m, n, k]
            profile = runner.profile(k, shape, device, 'proposed', 0,
                                     round_index, audits[k])
            results.setdefault(k, []).append(profile['profile']['median_us'])
            print(device, shape, round_index, profile['profile']['median_us'], flush=True)
    records = []
    for k in ks:
        tile = audits[k]['tiling']
        records.append({'shape': [m, n, k], 'device': device,
                        'tile': {key: tile.get(key) for key in runner.TILE_FIELDS},
                        'rounds_us': results[k],
                        'median_us': statistics.median(results[k]),
                        'relative_error': audits[k]['event']['max_relative_error']})
    return records


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_series, series) for series in SERIES]
        records = [record for future in futures for record in future.result()]
    runner.write(runner.OUT / 'records.json', records)
    print('completed', len(records), flush=True)


if __name__ == '__main__':
    main()
