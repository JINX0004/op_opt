#!/usr/bin/env python3
"""Summarize the held-out, paired FP32 online-selector experiment."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = Path('/home/xianfei/solvebase_bench_20261025/cases/fp32_online_30_new_v1')
MODES = {0: 'BASE', 2: 'SC split-K mode 2', 3: 'deterministic split-K',
         5: 'SC split-K mode 5', 6: 'SC split-K mode 6'}


def gmean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def summarize(rows):
    if not rows:
        return None
    ratios = [row['master_over_online'] for row in rows]
    return {'count': len(rows), 'geomean_master_over_online': gmean(ratios),
            'median_master_over_online': statistics.median(ratios),
            'online_faster': sum(x > 1 for x in ratios),
            'master_faster': sum(x < 1 for x in ratios)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--records', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    shapes = json.loads((HERE / 'dataset.json').read_text())
    if args.records:
        records = json.loads(args.records.read_text())
        failures = []
    else:
        records = [json.loads(p.read_text()) for p in args.raw.glob('device*/*/record.json')]
        failures = [str(p) for p in args.raw.glob('device*/*/failure.json')
                    if not (p.parent / 'record.json').exists()]
    by_index = {r['index']: r for r in records}
    if len(by_index) != len(records):
        raise RuntimeError('duplicate case index')
    if (len(records) != 30 or failures) and not args.allow_incomplete:
        raise RuntimeError(f'incomplete run: {len(records)}/30, failures={failures}')
    cases = []
    max_error = 0.0
    for index, record in sorted(by_index.items()):
        shape = shapes[index]
        if record['shape'] != shape or record['device'] != 4 + index % 4:
            raise RuntimeError(f'shape/device mismatch at case {index}')
        audits = record['audits']
        master, control, online = (audits[x]['tiling'] for x in
                                   ('master', 'research_default', 'online'))
        fields = ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN', 'singleCoreK',
                  'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb',
                  'depthA1', 'depthB1', 'block_dim', 'workspace_bytes',
                  'nd2nzA', 'nd2nzB', 'tiling_key')
        if any(master.get(k) != control.get(k) for k in fields):
            raise RuntimeError(f'unforced branch differs from master at case {index}')
        measurements = defaultdict(list)
        for item in record['measurements']:
            if item['profile']['kernel_count'] != 5 or item['event']['max_relative_error'] > .02:
                raise RuntimeError(f'profile/correctness failure at case {index}')
            max_error = max(max_error, item['event']['max_relative_error'])
            measurements[item['arm']].append(item)
        if set(measurements) != {'master', 'online'} or \
                len(measurements['master']) != len(measurements['online']):
            raise RuntimeError(f'unpaired profiles at case {index}')
        rounds = len(measurements['master'])
        if (record['changed_tiling'] and rounds != 2) or \
                (not record['changed_tiling'] and rounds not in (1, 3)):
            raise RuntimeError(f'incorrect paired round count at case {index}')
        master_us = statistics.median(x['profile']['median_us'] for x in measurements['master'])
        online_us = statistics.median(x['profile']['median_us'] for x in measurements['online'])
        model_us = float(record['model']['objective_us'])
        cases.append({'index': index, 'shape': 'x'.join(map(str, shape)),
                      'device': record['device'],
                      'master_family': MODES.get(master['tilingEnableSplitCore'], str(master['tilingEnableSplitCore'])),
                      'online_family': MODES.get(online['tilingEnableSplitCore'], str(online['tilingEnableSplitCore'])),
                      'model_family': MODES.get(int(record['model']['mode']), record['model']['mode']),
                      'model_online_family_agree': int(record['model']['mode']) == online['tilingEnableSplitCore'],
                      'family_changed': master['tilingEnableSplitCore'] != online['tilingEnableSplitCore'],
                      'tiling_changed': record['changed_tiling'],
                      'master_core': 'x'.join(str(master[k]) for k in ('singleCoreM', 'singleCoreN', 'singleCoreK')),
                      'online_core': 'x'.join(str(online[k]) for k in ('singleCoreM', 'singleCoreN', 'singleCoreK')),
                      'master_us': master_us, 'online_us': online_us,
                      'master_over_online': master_us / online_us,
                      'model_objective_us': model_us, 'model_over_online': model_us / online_us,
                      'rounds': rounds})
    changed = [r for r in cases if r['family_changed']]
    same = [r for r in cases if not r['family_changed']]
    rng = random.Random(20261109)
    boot = sorted(gmean([rng.choice(cases)['master_over_online'] for _ in cases])
                  for _ in range(20000)) if cases else []
    result = {'complete': len(cases) == 30 and not failures,
              'record_count': len(cases), 'failures': failures,
              'all': summarize(cases), 'family_changed': summarize(changed),
              'family_unchanged': summarize(same),
              'offline_online_family_mismatches': sum(not r['model_online_family_agree'] for r in cases),
              'shape_bootstrap_95pct': [boot[500], boot[19500]] if boot else None,
              'max_relative_error': max_error,
              'worst_online': sorted(cases, key=lambda r: r['master_over_online'])[:10],
              'best_online': sorted(cases, key=lambda r: r['master_over_online'], reverse=True)[:10]}
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, rows in [('comparison.csv', cases), ('family_changed.csv', changed),
                       ('family_unchanged.csv', same)]:
        with (HERE / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cases[0]) if cases else [],
                                    lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('worst_online', 'best_online')}, indent=2))


if __name__ == '__main__':
    main()
