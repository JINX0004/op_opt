#!/usr/bin/env python3
"""Compare FP32 clean-master family/tiles with the reduced model."""

import csv
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/xianfei/solvebase_bench_20261025')
OUT = ROOT / 'cases/fp32_family_boundary_v1'
SCORE = ROOT / 'fp32_family_boundary_v1/score_source'


def mode_name(mode):
    return {0: 'BASE', 2: 'SINGLE_2', 3: 'DETERMINISTIC_SPLIT_K',
            5: 'SINGLE_5', 6: 'SINGLE_6'}.get(mode, f'MODE_{mode}')


def main():
    dataset = json.loads((HERE / 'dataset.json').read_text())
    model = {}
    with (OUT / 'model_decisions.csv').open() as f:
        for row in csv.DictReader(f):
            model[(int(row['m']), int(row['n']), int(row['k']))] = row
    family_scores = {}
    with (OUT / 'family_scores.csv').open() as f:
        for row in csv.DictReader(f):
            family_scores[(int(row['m']), int(row['n']), int(row['k']))] = row
    records = []
    for index, case in enumerate(dataset):
        shape = tuple(case['shape'])
        path = OUT / f"{index:02d}_{'x'.join(map(str, shape))}"
        if not (path / 'result.json').exists():
            records.append({'index': index, 'group': case['group'], 'shape': shape,
                            'failure': (path / 'failure.json').read_text()
                            if (path / 'failure.json').exists() else 'not audited'})
            continue
        result = json.loads((path / 'result.json').read_text())
        tiling = result['tiling']
        mode = tiling['tilingEnableSplitCore']
        prediction = model[shape]
        family = family_scores[shape]
        predicted_mode = int(prediction['mode'])
        record = {'index': index, 'group': case['group'], 'shape': shape,
                  'source_mode': mode, 'source_family': mode_name(mode),
                  'model_mode': predicted_mode, 'model_family': mode_name(predicted_mode),
                  'family_agrees': mode == predicted_mode,
                  'model_winner_us': float(prediction['objective_us']),
                  'model_base_us': float(family['base_us']) if family['base_us'] else None,
                  'model_split_us': float(family['split_us']) if family['split_us'] else None,
                  'source_core': [tiling[x] for x in ('singleCoreM', 'singleCoreN', 'singleCoreK')],
                  'source_base': [tiling[x] for x in ('baseM', 'baseN', 'baseK')],
                  'model_core': [int(prediction[x]) for x in
                                 ('singleCoreM', 'singleCoreN', 'singleCoreK')],
                  'model_base': [int(prediction[x]) for x in ('baseM', 'baseN', 'baseK')],
                  'correctness_error': result['event']['max_relative_error']}
        if mode in (0, 3):
            used_cores = (min(20, tiling['output_tiles']) if mode == 0
                          else tiling['block_dim'])
            score_input = [*shape, mode, used_cores,
                           *(tiling[x] for x in ('singleCoreM', 'singleCoreN',
                                                 'singleCoreK', 'baseM', 'baseN',
                                                 'baseK')),
                           1 if mode == 0 else tiling['stepKa'], 3600000]
            proc = subprocess.run([str(SCORE)], input=' '.join(map(str, score_input)) + '\n',
                                  capture_output=True, text=True)
            if proc.returncode == 0:
                record['source_tile_model_us'] = float(proc.stdout.strip().split(',')[-1])
            else:
                record['source_score_error'] = proc.stderr.strip()
        if (path / 'profile.json').exists():
            profile = json.loads((path / 'profile.json').read_text())
            record['measured_kernel_us'] = profile['median_us']
            if record.get('source_tile_model_us'):
                record['source_tile_error_pct'] = 100 * (
                    record['source_tile_model_us'] / profile['median_us'] - 1)
        records.append(record)
    (OUT / 'comparison.json').write_text(json.dumps(records, indent=2) + '\n')
    fields = ['index', 'group', 'm', 'n', 'k', 'source_mode', 'model_mode',
              'family_agrees', 'measured_kernel_us', 'source_tile_model_us',
              'source_tile_error_pct', 'model_winner_us', 'model_base_us',
              'model_split_us', 'source_core',
              'source_base', 'model_core', 'model_base', 'failure']
    with (OUT / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, '') for key in fields}
            row.update(zip(('m', 'n', 'k'), record['shape']))
            writer.writerow(row)
    audited = [r for r in records if 'source_mode' in r]
    profiled = [r for r in audited if 'measured_kernel_us' in r]
    print(json.dumps({'audited': len(audited),
                      'family_agreements': sum(r['family_agrees'] for r in audited),
                      'source_modes': {str(m): sum(r['source_mode'] == m for r in audited)
                                       for m in sorted({r['source_mode'] for r in audited})},
                      'profiled': len(profiled),
                      'source_tile_scored': sum('source_tile_model_us' in r for r in audited)},
                     indent=2))


if __name__ == '__main__':
    main()
