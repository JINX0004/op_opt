#!/usr/bin/env python3
"""Measure forced source BASE near FP32 internal K-call boundaries on 910B3."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'cases/fp32_base_calls_v1'
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_family_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = OUT

# Boundary pairs change the number of internal Iterate/GetTensorC calls
# while barely changing arithmetic work. The M/N pairs expose how the
# additional output service scales with the amount of output data.
SHAPES = [
    (16, 16, 8064), (16, 16, 8192), (16, 16, 8256),
    (16, 16, 16384), (16, 16, 16448),
    (16, 16, 32768), (16, 16, 32832), (16, 16, 65408),
    (64, 64, 8192), (64, 64, 8256),
    (128, 128, 8192), (128, 128, 8256),
]


def measure_device(device):
    records = []
    for index, shape in enumerate(SHAPES):
        if index % 4 != device - 4:
            continue
        audited = runner.audit(index, shape, device, 'proposed', 0)
        if audited['tiling']['tilingEnableSplitCore'] != 0:
            raise RuntimeError(f'not BASE: {shape}')
        measured = runner.profile(index, shape, device, 'proposed', 0, 0, audited)
        tile = audited['tiling']
        records.append({
            'shape': shape, 'device': device,
            'median_us': measured['profile']['median_us'],
            'tiling': {name: tile[name] for name in runner.TILE_FIELDS if name in tile},
            'profile_csv': measured['profile']['profile_csv'],
        })
        print(device, shape, records[-1]['median_us'], flush=True)
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(measure_device, (4, 5, 6, 7)))
    records = sorted((item for group in groups for item in group),
                     key=lambda item: SHAPES.index(tuple(item['shape'])))
    (OUT / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
