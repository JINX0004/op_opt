#!/usr/bin/env python3
"""Summarize paired FP32 source-family times and fixed-tile model scores."""

import csv
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'cases/fp32_forced_family_v1'
MODEL = ROOT / 'cases/fp32_family_boundary_v1/model_decisions.csv'
SCORER = ROOT / 'fp32_family_boundary_v1/score_source'
KSTEP_SCORER = Path('/tmp/score_fp32_kstep_diagnostic')


def label(shape):
    return 'x'.join(map(str, shape))


def score(shape, tiling, scorer):
    mode = tiling['tilingEnableSplitCore']
    cores = min(20, tiling['output_tiles']) if mode == 0 else tiling['block_dim']
    args = [*shape, mode, cores,
            *(tiling[x] for x in ('singleCoreM', 'singleCoreN', 'singleCoreK',
                                  'baseM', 'baseN', 'baseK')),
            1 if mode == 0 else tiling['stepKa'], 3600000]
    proc = subprocess.run([str(scorer)], input=' '.join(map(str, args)) + '\n',
                          capture_output=True, text=True)
    if proc.returncode:
        return None, proc.stderr.strip()
    return float(proc.stdout.strip().split(',')[-1]), None


def main():
    decisions = {}
    with MODEL.open() as f:
        for row in csv.DictReader(f):
            decisions[tuple(int(row[x]) for x in ('m', 'n', 'k'))] = row
    records = []
    for shape in json.loads((HERE / 'dataset.json').read_text()):
        path = OUT / label(shape)
        arms = {}
        for arm in ('default', 'base', 'split'):
            audit_path = path / 'audit' / arm / 'result.json'
            if not audit_path.exists():
                arms[arm] = {'audit_failure': (path / 'audit' / arm / 'failure.json').read_text()
                             if (path / 'audit' / arm / 'failure.json').exists() else 'not run'}
                continue
            data = json.loads(audit_path.read_text())
            result = {'mode': data['tiling']['tilingEnableSplitCore'],
                      'core': [data['tiling'][x] for x in
                               ('singleCoreM', 'singleCoreN', 'singleCoreK')],
                      'base': [data['tiling'][x] for x in ('baseM', 'baseN', 'baseK')],
                      'stepKa': data['tiling']['stepKa'],
                      'stepKb': data['tiling']['stepKb'],
                      'block_dim': data['tiling']['block_dim'],
                      'max_relative_error': data['event']['max_relative_error']}
            if arm != 'default':
                result['source_tile_model_us'], result['model_error'] = score(
                    shape, data['tiling'], SCORER)
                if arm == 'base' and KSTEP_SCORER.exists():
                    result['l1_step_diagnostic_us'], result['l1_step_error'] = score(
                        shape, data['tiling'], KSTEP_SCORER)
                rounds = []
                durations = []
                for index in range(2):
                    profile_path = path / f'round{index}' / arm / 'result.json'
                    if profile_path.exists():
                        p = json.loads(profile_path.read_text())['profile']
                        rounds.append(p['median_us'])
                        durations.extend(p['durations_us'])
                if rounds:
                    result['round_medians_us'] = rounds
                if len(rounds) == 2:
                    result['measured_us'] = statistics.median(durations)
                    result['kernel_times_us'] = durations
            arms[arm] = result
        prediction = decisions[tuple(shape)]
        record = {'shape': shape, 'model_mode': int(prediction['mode']),
                  'model_winner_us': float(prediction['objective_us']),
                  'arms': arms}
        if all('measured_us' in arms[x] for x in ('base', 'split')):
            base = arms['base']['measured_us']
            split = arms['split']['measured_us']
            record['faster_mode'] = 0 if base <= split else 3
            record['model_regret_pct'] = 100 * (
                (base if record['model_mode'] == 0 else split) / min(base, split) - 1)
            default_mode = arms['default']['mode']
            record['master_regret_pct'] = 100 * (
                (base if default_mode == 0 else split) / min(base, split) - 1)
        records.append(record)
    (OUT / 'analysis.json').write_text(json.dumps(records, indent=2) + '\n')
    fields = ['m', 'n', 'k', 'master_mode', 'model_mode', 'base_us', 'split_us',
              'faster_mode', 'model_regret_pct', 'master_regret_pct',
              'base_source_score_us', 'split_source_score_us',
              'base_l1_step_diagnostic_us']
    with (OUT / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            a = record['arms']
            row = dict(zip(('m', 'n', 'k'), record['shape']))
            row.update(master_mode=a['default'].get('mode', ''), model_mode=record['model_mode'],
                       base_us=a['base'].get('measured_us', ''),
                       split_us=a['split'].get('measured_us', ''),
                       faster_mode=record.get('faster_mode', ''),
                       model_regret_pct=record.get('model_regret_pct', ''),
                       master_regret_pct=record.get('master_regret_pct', ''),
                       base_source_score_us=a['base'].get('source_tile_model_us', ''),
                       split_source_score_us=a['split'].get('source_tile_model_us', ''),
                       base_l1_step_diagnostic_us=a['base'].get('l1_step_diagnostic_us', ''))
            writer.writerow(row)
    complete = [r for r in records if 'faster_mode' in r]
    print(json.dumps({'shapes': len(records), 'paired': len(complete),
                      'model_faster_family': sum(r['model_mode'] == r['faster_mode']
                                                 for r in complete),
                      'master_faster_family': sum(r['arms']['default']['mode'] == r['faster_mode']
                                                  for r in complete)}, indent=2))


if __name__ == '__main__':
    main()
