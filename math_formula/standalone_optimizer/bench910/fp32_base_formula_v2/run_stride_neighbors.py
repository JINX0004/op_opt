#!/usr/bin/env python3
"""Use msprof task times around three new fixed-tile BASE K outliers."""

import importlib.util
import json
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
OUT = ROOT / 'cases/fp32_base_formula_v2/stride_neighbors'
GRID = ROOT / 'cases/fp32_base_formula_v2/stride_grid'
spec = importlib.util.spec_from_file_location('fp32_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = OUT

KS = (24320, 24576, 24832, 40704, 40960, 41216,
      57088, 57344, 57600)


def audit(k):
    center_path = GRID / 'device4' / f'{k}_2560x768x{k}' / 'audit/proposed/result.json'
    if center_path.exists():
        return json.loads(center_path.read_text())
    return runner.audit(k, [2560, 768, k], 4, 'proposed', 0)


def main():
    rows = []
    for k in KS:
        audited = audit(k)
        tile = audited['tiling']
        if (tile['baseM'], tile['baseN'], tile['baseK'], tile['stepKa'],
                tile['stepKb'], tile['block_dim']) != (128, 256, 32, 8, 4, 20):
            raise RuntimeError(f'tile changed at K={k}: {tile}')
        result = runner.profile(k, [2560, 768, k], 4, 'proposed', 0,
                                0, audited)
        row = {'k': k, 'measured_us': result['profile']['median_us'],
               'tile': {field: tile.get(field) for field in runner.TILE_FIELDS}}
        rows.append(row)
        runner.write(OUT / 'records.json', rows)
        print(k, row['measured_us'], flush=True)


if __name__ == '__main__':
    main()
