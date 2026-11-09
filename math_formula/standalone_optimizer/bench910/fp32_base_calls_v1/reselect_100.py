#!/usr/bin/env python3
"""Rescore recorded source BASE tiles; retain paired 910 timing provenance."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDS = HERE.parent / 'fp32_100_rerun_20261106/records.json'
if not RECORDS.exists():
    RECORDS = Path('/home/xianfei/solvebase_bench_20261025/fp32_100_rerun_20261106/records.json')


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: reselect_100.py /path/to/mmopt_selector')
    selector = Path(sys.argv[1])
    records = json.loads(RECORDS.read_text())
    source_rows = []
    chosen = []
    for record in records:
        source = next((record['audits'][arm]['tiling'] for arm in
                       ('proposed', 'master')
                       if record['audits'][arm]['tiling']['tilingEnableSplitCore'] == 0), None)
        if source is None:
            continue
        m, n, k = record['shape']
        core_m, core_n = source['singleCoreM'], source['singleCoreN']
        used_cores = min(20, ((m + core_m - 1) // core_m) *
                         ((n + core_n - 1) // core_n))
        source_rows.append(f'{m} {n} {k} fp32 {used_cores} {core_m} {core_n} '
                           f'{source["baseM"]} {source["baseN"]} {source["baseK"]}')
        chosen.append(record)
    result = subprocess.run([str(selector), '--source-base'],
                            input='\n'.join(source_rows) + '\n', text=True,
                            capture_output=True, check=True)
    scores = list(csv.DictReader(io.StringIO(result.stdout)))
    rows = []
    for record, score in zip(chosen, scores):
        measurements = record['measurements']
        actual = {}
        for arm in ('master', 'proposed'):
            values = sorted(item['profile']['median_us'] for item in measurements
                            if item['arm'] == arm)
            actual[arm] = values[len(values) // 2] if values else None
        rows.append({
            'index': record['index'], 'shape': 'x'.join(map(str, record['shape'])),
            'master_mode': record['audits']['master']['tiling']['tilingEnableSplitCore'],
            'old_model_mode': record['model']['mode'],
            'new_source_mode': score['mode'],
            'optimized_base_us': score['optimized_base_us'],
            'source_base_us': score['source_base_us'],
            'new_winner_us': score['objective_us'],
            'master_measured_us': actual['master'],
            'proposed_measured_us': actual['proposed'],
        })
    out = HERE / 'source_reselection.csv'
    with out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'scored {len(rows)}/{len(records)} source BASE tiles')
    changed = [r for r in rows if int(r['new_source_mode']) != int(r['old_model_mode'])]
    print(f'changed model family on {len(changed)} scored shapes')
    for row in changed:
        print(row['index'], row['shape'], row['old_model_mode'],
              '->', row['new_source_mode'], row['source_base_us'],
              row['new_winner_us'])


if __name__ == '__main__':
    main()
