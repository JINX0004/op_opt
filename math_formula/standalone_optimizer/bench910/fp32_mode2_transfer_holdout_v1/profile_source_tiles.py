#!/usr/bin/env python3
"""Profile audited source-emitted mode-2 tiles at 191/192/193-KiB pitch."""

import csv
import importlib.util
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = ROOT / 'fp32_mode2_transfer_holdout_v1'
spec = importlib.util.spec_from_file_location('audit_tiles', HERE / 'audit_source_tiles.py')
audit_tiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_tiles)
runner = audit_tiles.runner
KS = audit_tiles.KS
METRICS = ('aicore_time(us)', 'aic_mac_time(us)', 'aic_mte1_time(us)',
           'aic_mte2_time(us)', 'aic_fixpipe_time(us)',
           'aic_scalar_time(us)')


def counters(profile):
    directory = Path(profile['profile']['profile_csv']).parent
    matches = list(directory.glob('op_summary_*.csv'))
    if len(matches) != 1:
        raise RuntimeError((directory, matches))
    rows = [row for row in csv.DictReader(matches[0].open())
            if row['OP Type'] == 'MatMulV3'][-5:]
    if len(rows) != 5:
        raise RuntimeError((directory, len(rows)))
    return {key: statistics.median(float(row[key]) for row in rows)
            for key in METRICS}


def one_geometry(item):
    m, n, device = item
    audits = {k: runner.audit(k, [m, n, k], device, 'sweep', 2)
              for k in KS}
    records = {k: [] for k in KS}
    for round_index, order in enumerate((KS, tuple(reversed(KS)))):
        for k in order:
            profile = runner.profile(k, [m, n, k], device, 'sweep', 2,
                                     round_index, audits[k])
            row = {'round': round_index,
                   'kernel_us': profile['profile']['median_us'],
                   'counters': counters(profile)}
            records[k].append(row)
            print('source', m, n, k, round_index, row['kernel_us'],
                  row['counters']['aic_mte2_time(us)'], flush=True)
    return [{'shape': [m, n, k], 'device': device,
             'tile': {key: audits[k]['tiling'].get(key)
                      for key in runner.TILE_FIELDS},
             'measurements': records[k],
             'kernel_us': statistics.median(x['kernel_us'] for x in records[k]),
             'mte2_us': statistics.median(x['counters']['aic_mte2_time(us)']
                                          for x in records[k])}
            for k in KS]


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(one_geometry, audit_tiles.GEOMETRIES[:4]))
    groups.append(one_geometry(audit_tiles.GEOMETRIES[4]))
    (HERE / 'source_profiles.json').write_text(
        json.dumps([row for group in groups for row in group], indent=2) + '\n')


if __name__ == '__main__':
    main()
