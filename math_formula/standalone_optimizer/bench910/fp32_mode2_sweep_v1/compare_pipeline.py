#!/usr/bin/env python3
"""Score audited mode-2 tiles with the opt-in C++ pipeline profile."""

import argparse
import csv
import io
import json
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
audits = json.loads((HERE / 'audit_summary.json').read_text())
measured = list(csv.DictReader((HERE / 'results.csv').open()))
by_shape = {(int(row['m']), int(row['n']), int(row['k'])): row
            for row in measured}

parser = argparse.ArgumentParser()
parser.add_argument('scorer', type=Path)
args = parser.parse_args()

request = []
for item in audits:
    m, n, k = item['shape']
    t = item['tile']
    request.append(' '.join(map(str, (m, n, k,
        t['singleCoreM'], t['singleCoreN'], t['singleCoreK'],
        t['baseM'], t['baseN'], t['baseK'],
        t['stepM'], t['stepN'], t['stepKa'], t['block_dim']))))

proc = subprocess.run([str(args.scorer)], input='\n'.join(request) + '\n',
                      text=True, capture_output=True, check=True)
scored = list(csv.DictReader(io.StringIO(proc.stdout)))
if len(scored) != len(audits):
    raise RuntimeError('score count differs from audited tile count')

rows = []
for score in scored:
    key = tuple(int(score[k]) for k in ('m', 'n', 'k'))
    source = by_shape[key]
    if abs(float(score['baseline_us']) - float(source['model_us'])) > .002:
        raise RuntimeError(f'C++ baseline disagrees with prior score: {key}')
    actual = float(source['measured_us'])
    row = dict(source)
    row['pipeline_us'] = round(float(score['pipeline_us']), 3)
    row['pipeline_error_us'] = round(row['pipeline_us'] - actual, 3)
    row['pipeline_abs_pct'] = round(100 * abs(row['pipeline_error_us']) / actual, 3)
    row['baseline_abs_pct'] = round(
        100 * abs(float(source['model_us']) - actual) / actual, 3)
    rows.append(row)

with (HERE / 'pipeline_results.csv').open('w', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

def metrics(group):
    return {'count': len(group),
            'baseline_mape_pct': statistics.mean(x['baseline_abs_pct'] for x in group),
            'pipeline_mape_pct': statistics.mean(x['pipeline_abs_pct'] for x in group)}

ordinary = [x for x in rows if int(x['k']) != 49152]
summary = {'all': metrics(rows), 'excluding_k49152': metrics(ordinary),
           'by_geometry_excluding_k49152': {}}
for m, n in sorted({(int(x['m']), int(x['n'])) for x in rows}):
    group = [x for x in ordinary if (int(x['m']), int(x['n'])) == (m, n)]
    summary['by_geometry_excluding_k49152'][f'{m}x{n}'] = metrics(group)

(HERE / 'pipeline_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
for x in rows:
    if (int(x['m']), int(x['n']), int(x['k'])) == (2816, 1024, 36864):
        print('anchor', {k: x[k] for k in
            ('model_us', 'pipeline_us', 'measured_us', 'pipeline_error_us')})
