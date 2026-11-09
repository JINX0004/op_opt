#!/usr/bin/env python3
"""Analyze paired 100-shape results after source-emitted mode-2 scoring."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'fp32_source_base_100_v1/comparison.csv'
MODES = {0: 'BASE', 2: 'SC split-K mode 2', 3: 'deterministic split-K',
         5: 'SC split-K mode 5', 6: 'SC split-K mode 6'}
TILE_FIELDS = ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN', 'singleCoreK',
               'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb',
               'depthA1', 'depthB1', 'block_dim', 'workspace_bytes',
               'nd2nzA', 'nd2nzB', 'tiling_key')


def gmean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def summarize(rows):
    if not rows:
        return None
    values = [r['master_over_online'] for r in rows]
    return {'count': len(rows), 'geomean_master_over_online': gmean(values),
            'median_master_over_online': statistics.median(values),
            'online_faster': sum(x > 1 for x in values),
            'master_faster': sum(x < 1 for x in values),
            'online_over_5pct_faster': sum(x > 1.05 for x in values),
            'master_over_5pct_faster': sum(x < 1 / 1.05 for x in values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_text())
    previous = {int(r['index']): r for r in csv.DictReader(PREVIOUS.open())}
    shapes = json.loads((HERE.parent / 'fp32_source_base_100_v1/dataset.json').read_text())
    if len(records) != len(shapes) or len(previous) != len(shapes) or len(shapes) != 100:
        raise RuntimeError('expected exactly 100 old and new cases')
    by_index = {r['index']: r for r in records}
    if sorted(by_index) != list(range(100)):
        raise RuntimeError('missing or duplicate case index')
    cases = []
    max_error = 0.0
    for index in range(100):
        record = by_index[index]
        shape = shapes[index]
        if record['shape'] != shape or record['device'] != 4 + index % 4:
            raise RuntimeError(f'shape/device mismatch at {index}')
        audits = record['audits']
        master, control, online = (audits[x]['tiling'] for x in
                                   ('master', 'research_default', 'online'))
        if any(master.get(k) != control.get(k) for k in TILE_FIELDS):
            raise RuntimeError(f'unforced host differs from master at {index}')
        measurements = defaultdict(list)
        for item in record['measurements']:
            if item['event']['max_relative_error'] > .02 or item['profile']['kernel_count'] != 5:
                raise RuntimeError(f'correctness/profile failure at {index}')
            max_error = max(max_error, item['event']['max_relative_error'])
            measurements[item['arm']].append(item)
        if set(measurements) != {'master', 'online'} or \
                len(measurements['master']) != len(measurements['online']):
            raise RuntimeError(f'unpaired profiles at {index}')
        rounds = len(measurements['master'])
        if (record['changed_tiling'] and rounds != 2) or \
                (not record['changed_tiling'] and rounds != 1):
            raise RuntimeError(f'incorrect round count at {index}')
        master_us = statistics.median(x['profile']['median_us'] for x in measurements['master'])
        online_us = statistics.median(x['profile']['median_us'] for x in measurements['online'])
        previous_row = previous[index]
        old_family = previous_row['online_family']
        new_family = MODES.get(online['tilingEnableSplitCore'], str(online['tilingEnableSplitCore']))
        cases.append({'index': index, 'shape': 'x'.join(map(str, shape)),
                      'master_family': MODES.get(master['tilingEnableSplitCore'],
                                                 str(master['tilingEnableSplitCore'])),
                      'old_online_family': old_family, 'new_online_family': new_family,
                      'family_changed_vs_master': master['tilingEnableSplitCore'] != online['tilingEnableSplitCore'],
                      'family_changed_vs_old': old_family != new_family,
                      'tiling_changed': record['changed_tiling'],
                      'master_core': 'x'.join(str(master[k]) for k in
                                              ('singleCoreM', 'singleCoreN', 'singleCoreK')),
                      'new_online_core': 'x'.join(str(online[k]) for k in
                                                  ('singleCoreM', 'singleCoreN', 'singleCoreK')),
                      'master_us': master_us, 'old_online_us': float(previous_row['online_us']),
                      'online_us': online_us, 'master_over_online': master_us / online_us,
                      'old_online_over_new_online': float(previous_row['online_us']) / online_us,
                      'rounds': rounds})
    changed_vs_master = [r for r in cases if r['family_changed_vs_master']]
    changed_vs_old = [r for r in cases if r['family_changed_vs_old']]
    unchanged_vs_master = [r for r in cases if not r['family_changed_vs_master']]
    rng = random.Random(20261109)
    bootstrap = sorted(gmean([rng.choice(cases)['master_over_online'] for _ in cases])
                       for _ in range(20000))
    result = {'complete': True, 'record_count': 100, 'all': summarize(cases),
              'family_changed_vs_master': summarize(changed_vs_master),
              'family_unchanged_vs_master': summarize(unchanged_vs_master),
              'changed_vs_old_count': len(changed_vs_old),
              'shape_bootstrap_95pct': [bootstrap[500], bootstrap[19500]],
              'max_relative_error': max_error,
              'worst_online': sorted(cases, key=lambda r: r['master_over_online'])[:15],
              'best_online': sorted(cases, key=lambda r: r['master_over_online'], reverse=True)[:15]}
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, rows in [('comparison.csv', cases),
                       ('family_changed.csv', changed_vs_master),
                       ('selection_changes.csv', changed_vs_old)]:
        with (HERE / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cases[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('worst_online', 'best_online')}, indent=2))


if __name__ == '__main__':
    main()
