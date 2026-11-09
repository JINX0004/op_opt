#!/usr/bin/env python3
"""Check opt-in pipeline scores on the 11 source-emitted FP32 mode-2 tiles."""

import argparse
import csv
import io
import json
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODE2 = HERE.parent / 'fp32_mode2_v1'
FAMILY = HERE.parent / 'fp32_100_source_family_v1'
SWEEP_SHAPES = {(int(r['m']), int(r['n']), int(r['k'])) for r in
                csv.DictReader((HERE / 'results.csv').open())}

parser = argparse.ArgumentParser()
parser.add_argument('scorer', type=Path)
args = parser.parse_args()

inputs = []
for line in (MODE2 / 'source_mode2_tiles.txt').read_text().splitlines():
    (index, m, n, k, cores, cm, cn, ck, bm, bn, bk, sm, sn, sk) = (
        map(int, line.split()))
    inputs.append((index, m, n, k, cm, cn, ck, bm, bn, bk, sm, sn, sk,
                   cores))

request = '\n'.join(' '.join(map(str, row[1:])) for row in inputs) + '\n'
proc = subprocess.run([str(args.scorer)], input=request, text=True,
                      capture_output=True, check=True)
scores = list(csv.DictReader(io.StringIO(proc.stdout)))
measured = {int(r['index']): r for r in
            csv.DictReader((FAMILY / 'comparison.csv').open())}

rows = []
for source, score in zip(inputs, scores):
    index, m, n, k = source[:4]
    actual = float(measured[index]['master_us'])
    old, new = float(score['baseline_us']), float(score['pipeline_us'])
    rows.append({'index': index, 'm': m, 'n': n, 'k': k,
                 'in_fixed_tile_sweep': (m, n, k) in SWEEP_SHAPES,
                 'baseline_us': round(old, 3),
                 'pipeline_us': round(new, 3),
                 'measured_us': round(actual, 3),
                 'baseline_abs_pct': round(100 * abs(old - actual) / actual, 3),
                 'pipeline_abs_pct': round(100 * abs(new - actual) / actual, 3)})

with (HERE / 'source_tile_pipeline_comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

heldout = [r for r in rows if not r['in_fixed_tile_sweep']]
summary = {'source_tiles': len(rows), 'heldout_tiles': len(heldout),
           'heldout_baseline_mape_pct': statistics.mean(
               r['baseline_abs_pct'] for r in heldout),
           'heldout_pipeline_mape_pct': statistics.mean(
               r['pipeline_abs_pct'] for r in heldout)}
(HERE / 'source_tile_pipeline_summary.json').write_text(
    json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
