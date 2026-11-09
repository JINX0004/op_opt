#!/usr/bin/env python3
"""Extract paired AIC stage counters from the exact-tile mode-2 profiles."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
RECORDS = ROOT / 'cases/fp32_base_formula_v2/exact_mode2_tile/records.json'
OUT = ROOT / 'cases/fp32_base_formula_v2/exact_mode2_tile/counters.json'
METRICS = ('Task Duration(us)', 'aicore_time(us)', 'aic_mac_time(us)',
           'aic_mte1_time(us)', 'aic_mte2_time(us)',
           'aic_fixpipe_time(us)', 'aic_scalar_time(us)')


def one_profile(profile):
    folder = Path(profile['profile']['profile_csv']).parent
    matches = list(folder.glob('op_summary_*.csv'))
    if len(matches) != 1:
        raise RuntimeError((folder, matches))
    rows = [row for row in csv.DictReader(matches[0].open())
            if row['OP Type'] == 'MatMulV3'][-5:]
    if len(rows) != 5:
        raise RuntimeError((folder, len(rows)))
    return {name: statistics.median(float(row[name]) for row in rows)
            for name in METRICS}


def main():
    result = []
    for record in json.loads(RECORDS.read_text()):
        arms = {}
        for arm in ('master', 'research_default', 'proposed'):
            profiles = [one_profile(row) for row in record['measurements']
                        if row['arm'] == arm]
            if len(profiles) != 2:
                raise RuntimeError((record['index'], arm, len(profiles)))
            arms[arm] = {name: statistics.median(row[name] for row in profiles)
                         for name in METRICS}
        result.append({'index': record['index'], 'shape': record['shape'],
                       'arms': arms})
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
