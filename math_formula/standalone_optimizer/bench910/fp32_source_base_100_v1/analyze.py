#!/usr/bin/env python3
"""Combine the verified 30-case run with 70 fresh paired FP32 measurements."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIRST_30 = HERE.parent / 'fp32_source_base_fix_v1/records.json'
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
    ratios = [r['master_over_online'] for r in rows]
    return {'count': len(rows), 'geomean_master_over_online': gmean(ratios),
            'median_master_over_online': statistics.median(ratios),
            'online_faster': sum(x > 1 for x in ratios),
            'master_faster': sum(x < 1 for x in ratios),
            'online_over_5pct_faster': sum(x > 1.05 for x in ratios),
            'master_over_5pct_faster': sum(x < 1 / 1.05 for x in ratios)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--added-records', type=Path, required=True)
    args = parser.parse_args()
    shapes = json.loads((HERE / 'dataset.json').read_text())
    first = json.loads(FIRST_30.read_text())
    added = json.loads(args.added_records.read_text())
    records = first + added
    if len(shapes) != 100 or len(first) != 30 or len(added) != 70 or len(records) != 100:
        raise RuntimeError('expected 30 reused and 70 new cases')
    by_index = {r['index']: r for r in records}
    if sorted(by_index) != list(range(100)):
        raise RuntimeError('missing or duplicate case index')
    cases = []
    max_error = 0.0
    for index in range(100):
        record = by_index[index]
        shape = shapes[index]
        if record['shape'] != shape or record['device'] != 4 + index % 4:
            raise RuntimeError(f'shape/device mismatch at case {index}')
        audits = record['audits']
        master, control, online = (audits[x]['tiling'] for x in
                                   ('master', 'research_default', 'online'))
        if any(master.get(k) != control.get(k) for k in TILE_FIELDS):
            raise RuntimeError(f'unforced branch differs from master at {index}')
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
                (not record['changed_tiling'] and rounds not in (1, 3)):
            raise RuntimeError(f'incorrect round count at {index}')
        master_us = statistics.median(x['profile']['median_us'] for x in measurements['master'])
        online_us = statistics.median(x['profile']['median_us'] for x in measurements['online'])
        master_mode = master['tilingEnableSplitCore']
        online_mode = online['tilingEnableSplitCore']
        cases.append({'index': index, 'origin': 'reused_30' if index < 30 else 'added_70',
                      'shape': 'x'.join(map(str, shape)), 'device': record['device'],
                      'master_family': MODES.get(master_mode, str(master_mode)),
                      'online_family': MODES.get(online_mode, str(online_mode)),
                      'family_changed': master_mode != online_mode,
                      'tiling_changed': record['changed_tiling'],
                      'master_core': 'x'.join(str(master[k]) for k in ('singleCoreM', 'singleCoreN', 'singleCoreK')),
                      'online_core': 'x'.join(str(online[k]) for k in ('singleCoreM', 'singleCoreN', 'singleCoreK')),
                      'master_base': 'x'.join(str(master[k]) for k in ('baseM', 'baseN', 'baseK')),
                      'online_base': 'x'.join(str(online[k]) for k in ('baseM', 'baseN', 'baseK')),
                      'master_workspace_mib': master['workspace_bytes'] / 1048576,
                      'online_workspace_mib': online['workspace_bytes'] / 1048576,
                      'master_us': master_us, 'online_us': online_us,
                      'master_over_online': master_us / online_us, 'rounds': rounds})
    changed = [r for r in cases if r['family_changed']]
    unchanged = [r for r in cases if not r['family_changed']]
    rng = random.Random(20261110)
    bootstrap = sorted(gmean([rng.choice(cases)['master_over_online'] for _ in cases])
                       for _ in range(20000))
    result = {'complete': True, 'record_count': 100,
              'all': summarize(cases), 'reused_30': summarize(cases[:30]),
              'added_70': summarize(cases[30:]),
              'family_changed': summarize(changed), 'family_unchanged': summarize(unchanged),
              'shape_bootstrap_95pct': [bootstrap[500], bootstrap[19500]],
              'max_relative_error': max_error,
              'family_pairs': dict(sorted((f'{a} -> {b}', n) for (a, b), n in
                  ((pair, sum((r['master_family'], r['online_family']) == pair for r in changed))
                   for pair in {(r['master_family'], r['online_family']) for r in changed}))),
              'worst_online': sorted(cases, key=lambda r: r['master_over_online'])[:15],
              'best_online': sorted(cases, key=lambda r: r['master_over_online'], reverse=True)[:15]}
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, rows in [('comparison.csv', cases), ('family_changed.csv', changed),
                       ('family_unchanged.csv', unchanged)]:
        with (HERE / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cases[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('worst_online', 'best_online')}, indent=2))


if __name__ == '__main__':
    main()
