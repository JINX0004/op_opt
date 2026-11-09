#!/usr/bin/env python3
"""Compare source BASE scores with the controlled 910B3 K/stride probes."""

import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: analyze.py /path/to/score_source_tiles')
    records = json.loads((HERE / 'same_device_records.json').read_text())
    request = []
    for record in records:
        m, n, k = record['shape']
        tile = record['tile']
        if (tile['baseM'], tile['baseN'], tile['baseK'], tile['stepKa'],
                tile['stepKb']) != (128, 256, 32, 8, 4):
            raise RuntimeError(f'unexpected emitted BASE tile: {record}')
        cores = min(20, math.ceil(m / tile['singleCoreM']) *
                    math.ceil(n / tile['singleCoreN']))
        request.append(' '.join(map(str, (m, n, k, 0, cores,
            tile['singleCoreM'], tile['singleCoreN'], tile['singleCoreK'],
            tile['baseM'], tile['baseN'], tile['baseK']))))
    process = subprocess.run([sys.argv[1]], input='\n'.join(request) + '\n',
                             text=True, capture_output=True, check=True)
    scores = list(csv.DictReader(process.stdout.splitlines()))
    if len(scores) != len(records):
        raise RuntimeError('missing source scores')
    comparisons = []
    for record, score in zip(records, scores):
        m, n, k = record['shape']
        predicted = float(score['source_score_us'])
        measured = record['median_us']
        comparisons.append({'m': m, 'n': n, 'k': k, 'device': record['device'],
                            'source_score_us': predicted, 'measured_us': measured,
                            'error_pct': 100 * (predicted / measured - 1),
                            'round0_us': record['rounds_us'][0],
                            'round1_us': record['rounds_us'][1]})
    write_csv(HERE / 'comparison.csv', comparisons)

    probe = json.loads((HERE / 'stride_probe.json').read_text())
    derived = []
    for m, n, k in ((2560, 768, 32512), (2560, 768, 32768),
                    (2560, 768, 33024), (2560, 1024, 48896),
                    (2560, 1024, 49152), (2560, 1024, 49408)):
        for operand in ('A', 'B'):
            one = next(r for r in probe if r['m'] == m and r['n'] == n and
                       r['k'] == k and r['operand'] == operand and r['copies'] == 1)
            many = next(r for r in probe if r['m'] == m and r['n'] == n and
                        r['k'] == k and r['operand'] == operand and r['copies'] == 128)
            derived.append({'m': m, 'n': n, 'k': k, 'operand': operand,
                            'tile_axis': many['tile_axis'], 'k_span': many['k_span'],
                            'one_copy_us': one['device_elapsed_us'],
                            'many_copy_us': many['device_elapsed_us'],
                            'incremental_copy_us':
                                (many['device_elapsed_us'] - one['device_elapsed_us']) / 127})
    write_csv(HERE / 'probe_summary.csv', derived)

    prior = HERE.parent / 'fp32_100_rerun_20261106'
    source_scores = {int(row['index']): float(row['source_base_us'])
                     for row in csv.DictReader(
                         (HERE.parent / 'fp32_base_calls_v1/source_reselection.csv').open())}
    prior_records = json.loads((prior / 'records.json').read_text())
    source_base_rows = []
    for record in prior_records:
        index = record['index']
        if index not in source_scores:
            continue
        arm = next(arm for arm in ('proposed', 'master') if
                   record['audits'][arm]['tiling']['tilingEnableSplitCore'] == 0)
        actual = statistics.median(item['profile']['median_us'] for item in
                                   record['measurements'] if item['arm'] == arm)
        m, n, k = record['shape']
        source_base_rows.append({'index': index, 'm': m, 'n': n, 'k': k,
                                 'score_us': source_scores[index],
                                 'measured_us': actual,
                                 'error_pct': 100 * (source_scores[index] / actual - 1)})
    if len(source_base_rows) != 81:
        raise RuntimeError('expected 81 audited source BASE cases')
    write_csv(HERE / 'source_base_81.csv', source_base_rows)
    for name, subset in (
        ('wide non-spike', [r for r in source_base_rows
                            if r['n'] > 128 and r['k'] not in (32768, 49152)]),
        ('wide spike', [r for r in source_base_rows
                        if r['n'] > 128 and r['k'] in (32768, 49152)]),
        ('narrow N', [r for r in source_base_rows if r['n'] <= 128]),
    ):
        print(f'{name}: n={len(subset)}, MAPE='
              f'{statistics.mean(abs(r["error_pct"]) for r in subset):.1f}%, '
              f'bias={statistics.mean(r["error_pct"] for r in subset):+.1f}%')
    for n, k in ((768, 32768), (1024, 49152)):
        middle = next(r for r in comparisons if r['n'] == n and r['k'] == k)
        edges = [r for r in comparisons if r['n'] == n and abs(r['k'] - k) == 256]
        baseline = sum(r['measured_us'] for r in edges) / 2
        print(f'N={n}, K={k}: measured {middle["measured_us"]:.1f} us, '
              f'neighbor baseline {baseline:.1f} us, '
              f'excess {(middle["measured_us"]/baseline-1)*100:.1f}%')


if __name__ == '__main__':
    main()
