#!/usr/bin/env python3
"""Audit forced-family FP32 mode-2 tiles around 192-KiB A row pitch."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
OUT = ROOT / 'cases/fp32_mode2_transfer_holdout_v1'
spec = importlib.util.spec_from_file_location(
    'sweep', ROOT / 'fp32_mode2_sweep_v1/run.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)
runner = sweep.runner
runner.OUT = OUT
GEOMETRIES = ((2816, 1024, 4), (3072, 1024, 5),
              (2048, 2048, 6), (2816, 1536, 7),
              (8192, 4096, 5))
KS = (48896, 49152, 49408)


def one_geometry(item):
    m, n, device = item
    rows = []
    for k in KS:
        shape = [m, n, k]
        audit = runner.audit(k, shape, device, 'sweep', 2)
        tile = audit['tiling']
        if (tile['tilingEnableSplitCore'] != 2 or
                tile['baseM'] != 128 or tile['baseN'] != 128 or
                audit['event']['max_relative_error'] > .02):
            raise RuntimeError((shape, tile))
        row = {'shape': shape, 'device': device,
               'tile': {key: tile.get(key) for key in runner.TILE_FIELDS},
               'max_relative_error': audit['event']['max_relative_error']}
        rows.append(row)
        print(shape, row['tile']['singleCoreM'],
              row['tile']['singleCoreN'], row['tile']['singleCoreK'],
              row['tile']['stepM'], flush=True)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(one_geometry, GEOMETRIES[:4]))
    groups.append(one_geometry(GEOMETRIES[4]))
    (ROOT / 'fp32_mode2_transfer_holdout_v1' / 'tile_audits.json').write_text(
        json.dumps([row for group in groups for row in group], indent=2) + '\n')


if __name__ == '__main__':
    main()
