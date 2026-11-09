#!/usr/bin/env python3
"""Controlled source-emitted FP32 mode-2 K and geometry sweep on 910B3."""

import argparse
import csv
import importlib.util
import json
import statistics
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
OUT = ROOT / 'cases/fp32_mode2_sweep_v1'
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_family_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = OUT
runner.PACKAGES['sweep'] = runner.RESEARCH
runner.SOURCES['sweep'] = runner.SOURCES['proposed']
original_environment = runner.environment


def environment(arm, mode, log_level):
    env = original_environment('research_default', mode, log_level)
    env['MMV3_RESEARCH_SPLITK_FAMILY'] = 'single'
    return env


runner.environment = environment

# The first group isolates K on one device. Other groups test whether a
# per-slice residual transfers to different source-emitted M/N tiles.
CASES = {
    4: [(2816, 1024, k) for k in
        (27392, 32768, 36864, 40960, 45056, 47104, 48128, 48896,
         49152, 49408, 57344, 65408)],
    5: [(3072, 1024, k) for k in
        (32768, 36864, 45056, 48128, 49152, 49408)],
    6: [(2048, 2048, k) for k in
        (27392, 32768, 36864, 49152, 48896, 49408)],
    7: [(2816, 1536, k) for k in
        (30720, 32256, 32768, 33792, 36864, 49152, 48896, 49408)],
}
EXPECTED_FP32_TILE = {'tilingEnableSplitCore': 2, 'baseM': 128,
                      'baseN': 128, 'baseK': 64, 'singleCoreK': 256}
COUNTERS = ('Task Duration(us)', 'aicore_time(us)', 'aic_mac_time(us)',
            'aic_scalar_time(us)', 'aic_mte1_time(us)',
            'aic_mte2_time(us)', 'aic_fixpipe_time(us)')


def op_counters(msprof_dir):
    paths = list(Path(msprof_dir).glob('PROF_*/mindstudio_profiler_output/op_summary_*.csv'))
    rows = [row for path in paths for row in csv.DictReader(path.open())
            if row['OP Type'] == 'MatMulV3']
    if len(rows) < 5:
        raise RuntimeError(f'fewer than five MatMulV3 rows in {msprof_dir}')
    rows = rows[-5:]
    return {key: statistics.median(float(row[key]) for row in rows)
            for key in COUNTERS}


def run_one(device, index, shape, phase):
    directory = OUT / f'device{device}' / f'{index:03d}_{runner.label(shape)}'
    try:
        audit = runner.audit(index, shape, device, 'sweep', 2)
        tile = audit['tiling']
        if any(tile.get(key) != value for key, value in EXPECTED_FP32_TILE.items()):
            raise RuntimeError(f'unexpected mode/base tile: {tile}')
        if phase == 'audit':
            return {'device': device, 'index': index, 'shape': shape,
                    'tile': {key: tile.get(key) for key in runner.TILE_FIELDS}}
        rounds = 2 if shape in ((2816, 1024, 36864),
                                (2816, 1024, 49152)) else 1
        profiles = [runner.profile(index, shape, device, 'sweep', 2, r, audit)
                    for r in range(rounds)]
        counters = [op_counters(directory / f'round{r}' / 'sweep' / 'msprof')
                    for r in range(rounds)]
        result = {
            'device': device, 'index': index, 'shape': shape,
            'tile': {key: tile.get(key) for key in runner.TILE_FIELDS},
            'median_us': statistics.median(p['profile']['median_us'] for p in profiles),
            'counters': {key: statistics.median(row[key] for row in counters)
                         for key in COUNTERS},
            'profile_rounds_us': [p['profile']['median_us'] for p in profiles],
        }
        runner.write(directory / 'record.json', result)
        print(device, shape, result['median_us'], 'us', flush=True)
        return result
    except Exception as exc:
        runner.write(directory / 'failure.json',
                     {'error': str(exc), 'traceback': traceback.format_exc()})
        print('FAILED', device, shape, str(exc), flush=True)
        return {'device': device, 'index': index, 'shape': shape,
                'error': str(exc)}


def on_device(device, phase):
    return [run_one(device, index, shape, phase)
            for index, shape in enumerate(CASES[device])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('audit', 'profile'))
    args = parser.parse_args()
    library = (runner.RESEARCH /
               'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so')
    if runner.sha(library) != runner.HASHES['proposed']:
        raise RuntimeError('forced-family package changed')
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(lambda d: on_device(d, args.phase), CASES))
    records = [row for group in groups for row in group]
    runner.write(OUT / f'{args.phase}_summary.json', records)
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    main()
