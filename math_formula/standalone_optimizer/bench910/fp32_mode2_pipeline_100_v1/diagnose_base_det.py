#!/usr/bin/env python3
"""Score the emitted BASE and deterministic tiles in the 21 family switches."""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'fp32_100_rerun_20261106'


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: diagnose_base_det.py /path/to/score_source_tiles')
    changed = [row for row in csv.DictReader((HERE / 'family_changed.csv').open())
               if row['master_mode'] == '3' and row['pipeline_mode'] == '0']
    records = json.loads((PRIOR / 'records.json').read_text())
    source_input = []
    for row in changed:
        record = records[int(row['index'])]
        m, n, k = record['shape']
        for arm in ('master', 'proposed'):
            tile = record['audits'][arm]['tiling']
            mode = tile['tilingEnableSplitCore']
            cores = (tile['block_dim'] if mode == 3 else
                     min(20, math.ceil(m / tile['singleCoreM']) *
                         math.ceil(n / tile['singleCoreN'])))
            source_input.append(' '.join(map(str, (m, n, k, mode, cores,
                tile['singleCoreM'], tile['singleCoreN'], tile['singleCoreK'],
                tile['baseM'], tile['baseN'], tile['baseK']))))
    result = subprocess.run([sys.argv[1]], input='\n'.join(source_input) + '\n',
                            text=True, capture_output=True, check=True)
    scores = list(csv.DictReader(result.stdout.splitlines()))
    if len(changed) != 21 or len(scores) != 42:
        raise RuntimeError('expected 21 audited deterministic-to-BASE pairs')
    output = []
    for row, deterministic, base in zip(changed, scores[::2], scores[1::2]):
        det_score = float(deterministic['source_score_us'])
        base_score = float(base['source_score_us'])
        det_actual = float(row['master_us'])
        base_actual = float(row['pipeline_family_us'])
        output.append({
            'index': int(row['index']), 'shape': row['shape'],
            'det_score': det_score, 'det_measured': det_actual,
            'det_error_pct': 100 * (det_score / det_actual - 1),
            'base_score': base_score, 'base_measured': base_actual,
            'base_error_pct': 100 * (base_score / base_actual - 1),
            'score_winner': 'BASE' if base_score < det_score else 'det',
            'measured_winner': 'BASE' if base_actual < det_actual else 'det'})
    with (HERE / 'source_base_det_diagnosis.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Source-tile rank correct: {sum(r['score_winner'] == r['measured_winner'] for r in output)}/21")
    for family in ('det', 'base'):
        errors = [r[f'{family}_error_pct'] for r in output]
        print(f'{family}: mean bias {sum(errors)/len(errors):+.1f}%, '
              f'MAPE {sum(map(abs, errors))/len(errors):.1f}%')


if __name__ == '__main__':
    main()
