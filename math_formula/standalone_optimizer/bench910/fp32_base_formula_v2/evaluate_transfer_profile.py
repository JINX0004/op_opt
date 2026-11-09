#!/usr/bin/env python3
"""Score audited source BASE tiles under the opt-in MTE2 service profile."""

import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent


def read(name):
    return json.loads((HERE / name).read_text())


def candidate(dataset, index, m, n, k, tile, measured):
    return {'dataset': dataset, 'index': index, 'm': m, 'n': n, 'k': k,
            'measured_us': measured, 'tile': tile}


def main(scorer):
    candidates = []
    baseline_rows = list(csv.DictReader((HERE / 'source_base_81.csv').open()))
    prior = json.loads((BENCH / 'fp32_100_rerun_20261106/records.json').read_text())
    by_index = {row['index']: row for row in prior}
    for row in baseline_rows:
        index = int(row['index'])
        record = by_index[index]
        tile = next(record['audits'][arm]['tiling']
                    for arm in ('proposed', 'master')
                    if record['audits'][arm]['tiling']['tilingEnableSplitCore'] == 0)
        candidates.append(candidate('source_base_81', index,
            *(int(row[name]) for name in ('m', 'n', 'k')),
            tile, float(row['measured_us'])))
    for index, record in enumerate(read('same_device_records.json')):
        m, n, k = record['shape']
        candidates.append(candidate('paired_K', index, m, n, k,
                                    record['tile'], record['median_us']))
    for index, record in enumerate(read('stride_neighbors_records.json')):
        candidates.append(candidate('new_K_neighbors', index, 2560, 768,
                                    record['k'], record['tile'],
                                    record['measured_us']))
    request = ''
    for row in candidates:
        tile = row['tile']
        values = [row['m'], row['n'], row['k'],
                  min(20, math.ceil(row['m'] / tile['singleCoreM']) *
                          math.ceil(row['n'] / tile['singleCoreN'])),
                  tile['singleCoreM'], tile['singleCoreN'], tile['singleCoreK'],
                  tile['baseM'], tile['baseN'], tile['baseK']]
        request += ' '.join(map(str, values)) + '\n'
    result = subprocess.run([scorer], input=request, text=True,
                            capture_output=True, check=True)
    scores = list(csv.DictReader(result.stdout.splitlines()))
    if len(scores) != len(candidates):
        raise RuntimeError((len(scores), len(candidates)))
    output = []
    for row, score in zip(candidates, scores):
        before = float(score['baseline_us'])
        after = float(score['transfer_us'])
        measured = row['measured_us']
        output.append({'dataset': row['dataset'], 'index': row['index'],
                       'm': row['m'], 'n': row['n'], 'k': row['k'],
                       'baseline_us': before, 'transfer_us': after,
                       'measured_us': measured,
                       'baseline_error_pct': 100 * (before / measured - 1),
                       'transfer_error_pct': 100 * (after / measured - 1)})
    with (HERE / 'transfer_profile_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(output)
    for name in ('source_base_81', 'paired_K', 'new_K_neighbors'):
        group = [row for row in output if row['dataset'] == name]
        before = statistics.mean(abs(row['baseline_error_pct']) for row in group)
        after = statistics.mean(abs(row['transfer_error_pct']) for row in group)
        changed = sum(row['transfer_us'] > row['baseline_us'] + 0.01 for row in group)
        print(f'{name}: n={len(group)}, changed={changed}, MAPE {before:.2f}% -> {after:.2f}%')
    for row in output:
        if row['transfer_us'] > row['baseline_us'] + 0.01:
            print(row)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: evaluate_transfer_profile.py /path/to/score_base_mte2')
    main(sys.argv[1])
