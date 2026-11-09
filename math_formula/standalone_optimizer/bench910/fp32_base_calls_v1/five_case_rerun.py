#!/usr/bin/env python3
"""Paired 910B3 rerun of the five FP32 source-score family changes."""

import csv
import importlib.util
import json
import statistics
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'cases/fp32_base_five_family_rerun_v1'
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_family_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = OUT
runner.PACKAGES.update(old_base=runner.RESEARCH, new_selected=runner.RESEARCH)
runner.SOURCES.update(old_base=runner.SOURCES['proposed'],
                      new_selected=runner.SOURCES['proposed'])
original_environment = runner.environment


def environment(arm, mode, log_level):
    if arm == 'old_base':
        return original_environment('proposed', 0, log_level)
    if arm == 'new_selected':
        if mode == 2:
            env = original_environment('research_default', 2, log_level)
            env['MMV3_RESEARCH_SPLITK_FAMILY'] = 'single'
            return env
        if mode == 3:
            return original_environment('proposed', 3, log_level)
        raise RuntimeError(f'unsupported newly selected mode {mode}')
    return original_environment(arm, mode, log_level)


runner.environment = environment


def cases():
    wanted = {0, 8, 13, 19, 21}
    rows = list(csv.DictReader((HERE / 'source_reselection.csv').open()))
    selected = []
    for row in rows:
        index = int(row['index'])
        if index not in wanted:
            continue
        shape = tuple(int(x) for x in row['shape'].split('x'))
        selected.append((index, shape, int(row['new_source_mode'])))
    if {index for index, _, _ in selected} != wanted:
        raise RuntimeError('missing one of the five changed cases')
    return selected


def one(index, shape, selected_mode, device):
    base = OUT / f'device{device}' / f'{index:03d}_{"x".join(map(str, shape))}'
    try:
        arms = {'master': selected_mode, 'old_base': 0,
                'new_selected': selected_mode}
        audits = {arm: runner.audit(index, shape, device, arm, mode)
                  for arm, mode in arms.items()}
        for arm, mode in arms.items():
            emitted = audits[arm]['tiling']['tilingEnableSplitCore']
            if emitted != mode:
                raise RuntimeError(f'{arm} emitted mode {emitted}, expected {mode}')
        measurements = []
        for round_index in range(2):
            order = ('master', 'old_base', 'new_selected') if round_index == 0 \
                else ('new_selected', 'old_base', 'master')
            for arm in order:
                item = runner.profile(index, shape, device, arm,
                                      arms[arm], round_index, audits[arm])
                measurements.append(item)
                print(index, shape, device, round_index, arm,
                      item['profile']['median_us'], flush=True)
        medians = {arm: statistics.median(
            item['profile']['median_us'] for item in measurements
            if item['arm'] == arm) for arm in arms}
        tile_fields = runner.TILE_FIELDS
        tilings = {arm: {field: audits[arm]['tiling'].get(field)
                         for field in tile_fields} for arm in arms}
        result = {
            'index': index, 'shape': shape, 'device': device,
            'selected_mode': selected_mode, 'tilings': tilings,
            'medians_us': medians,
            'selected_vs_master_pct': 100 * (medians['new_selected'] /
                                             medians['master'] - 1),
            'selected_vs_old_base_pct': 100 * (medians['new_selected'] /
                                               medians['old_base'] - 1),
        }
        runner.write(base / 'record.json', result)
        return result
    except Exception as error:
        runner.write(base / 'failure.json',
                     {'error': str(error), 'traceback': traceback.format_exc()})
        raise


def main():
    selected = cases()
    device_by_index = {0: 4, 8: 5, 13: 6, 19: 7, 21: 4}

    def on_device(device):
        return [one(i, shape, mode, device) for i, shape, mode in selected
                if device_by_index[i] == device]

    with ThreadPoolExecutor(max_workers=4) as pool:
        grouped = list(pool.map(on_device, (4, 5, 6, 7)))
    results = sorted((record for group in grouped for record in group),
                     key=lambda record: record['index'])
    runner.write(OUT / 'summary.json', results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
