#!/usr/bin/env python3
"""Hold FP32 BASE M/N fixed and probe nearby K row strides on 910B3."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = ROOT / 'cases/fp32_base_formula_v2/k_sweep'

CASES = (
    (2560, 768, 32512), (2560, 768, 32768),
    (2560, 768, 33024), (2560, 768, 36864),
    (2560, 1024, 48896), (2560, 1024, 49152),
    (2560, 1024, 49408),
)


def measure(index, shape):
    device = 4 + index % 4
    directory = runner.OUT / f'device{device}' / f'{index:03d}_{runner.label(shape)}'
    result_path = directory / 'record.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    audit = runner.audit(index, list(shape), device, 'proposed', 0)
    profile = runner.profile(index, list(shape), device, 'proposed', 0, 0, audit)
    result = {'index': index, 'shape': shape, 'device': device,
              'tile': {key: audit['tiling'].get(key) for key in runner.TILE_FIELDS},
              'median_us': profile['profile']['median_us'],
              'relative_error': profile['event']['max_relative_error']}
    runner.write(result_path, result)
    print(shape, result['median_us'], result['tile']['baseK'], flush=True)
    return result


def on_device(device):
    return [measure(i, shape) for i, shape in enumerate(CASES)
            if 4 + i % 4 == device]


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(on_device, device) for device in (4, 5, 6, 7)]
        records = [record for future in futures for record in future.result()]
    runner.write(runner.OUT / 'records.json', sorted(records, key=lambda row: row['index']))
    print('completed', len(records), flush=True)


if __name__ == '__main__':
    main()
