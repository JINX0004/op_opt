#!/usr/bin/env python3
"""Combine the 96 unchanged paired runs with four fresh 910B3 pairs."""

import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'fp32_100_rerun_20261106'
MODE = {'BASE': 0, 'deterministic': 3, 'mode2': 2, 'mode5': 5}


def gmean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def summary(rows):
    return {'count': len(rows),
            'geomean_master_over_pipeline': gmean([r['speedup'] for r in rows]),
            'pipeline_faster': sum(r['speedup'] > 1 for r in rows),
            'master_faster': sum(r['speedup'] < 1 for r in rows)}


def main():
    shapes = json.loads((PRIOR / 'dataset.json').read_text())
    previous = list(csv.DictReader((PRIOR / 'comparison.csv').open()))
    decisions = (HERE / 'pipeline_decisions.txt').read_text().splitlines()
    fresh = {r['index']: r for r in json.loads((HERE / 'changed_records.json').read_text())}
    if len(shapes) != 100 or len(previous) != 100 or len(decisions) != 100:
        raise RuntimeError('expected 100 cases')
    if set(fresh) != {17, 18, 20, 42}:
        raise RuntimeError(f'expected four changed cases, got {sorted(fresh)}')
    rows = []
    for i, (shape, old, decision) in enumerate(zip(shapes, previous, decisions)):
        if i != int(old['index']) or decision.split()[0] != 'x'.join(map(str, shape)):
            raise RuntimeError(f'shape/index mismatch at {i}')
        name = decision.split('winner=')[1].split()[0]
        new_mode = MODE[name]
        row = {'index': i, 'shape': old['shape'], 'device': int(old['device']),
               'master_mode': int(old['master_mode']), 'prior_mode': int(old['model_mode']),
               'pipeline_mode': new_mode,
               'master_us': float(old['master_us']),
               'pipeline_family_us': float(old['model_family_us']),
               'measurement_source': 'prior paired run',
               'master_core': old['master_core'],
               'pipeline_family_core': old['model_family_core']}
        if i in fresh:
            record = fresh[i]
            if (record['shape'] != shape or record['device'] != 4 + i % 4 or
                    record['new_model_mode'] != new_mode):
                raise RuntimeError(f'fresh record mismatch at {i}')
            audits = record['audits']
            a, b, c = (audits[arm]['tiling'] for arm in
                       ('master', 'proposed', 'research_default'))
            if any(a.get(field) != c.get(field) for field in
                   ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN',
                    'singleCoreK', 'baseM', 'baseN', 'baseK', 'stepKa', 'stepKb')):
                raise RuntimeError(f'unforced research/master tile mismatch at {i}')
            if b['tilingEnableSplitCore'] != new_mode:
                raise RuntimeError(f'forced family mismatch at {i}')
            times = {arm: [] for arm in ('master', 'proposed')}
            for item in record['measurements']:
                if item['event']['max_relative_error'] > .02 or item['profile']['kernel_count'] != 5:
                    raise RuntimeError(f'profile failure at {i}')
                times[item['arm']].append(item['profile']['median_us'])
            if len(times['master']) != 2 or len(times['proposed']) != 2:
                raise RuntimeError(f'expected two paired rounds at {i}')
            row.update(master_us=statistics.median(times['master']),
                       pipeline_family_us=statistics.median(times['proposed']),
                       measurement_source='fresh paired run',
                       master_core='x'.join(str(a[x]) for x in
                                              ('singleCoreM', 'singleCoreN', 'singleCoreK')),
                       pipeline_family_core='x'.join(str(b[x]) for x in
                                                      ('singleCoreM', 'singleCoreN', 'singleCoreK')))
        elif new_mode != row['prior_mode']:
            raise RuntimeError(f'unmeasured new selection at {i}')
        row['family_changed'] = row['master_mode'] != new_mode
        row['speedup'] = row['master_us'] / row['pipeline_family_us']
        rows.append(row)
    changed = [r for r in rows if r['family_changed']]
    unchanged = [r for r in rows if not r['family_changed']]
    rng = random.Random(91020261107)
    bootstrap = sorted(gmean([rng.choice(rows)['speedup'] for _ in rows])
                       for _ in range(20000))
    result = {'all': summary(rows), 'family_changed': summary(changed),
              'family_unchanged': summary(unchanged),
              'transitions': {f'{a}->{b}': count for (a, b), count in
                              sorted(Counter((r['master_mode'], r['pipeline_mode'])
                                             for r in rows).items())},
              'bootstrap_95pct': [bootstrap[500], bootstrap[19500]],
              'fresh_cases': [r for r in rows if r['measurement_source'] == 'fresh paired run'],
              'worst_changed': sorted(changed, key=lambda r: r['speedup'])[:10],
              'best_changed': sorted(changed, key=lambda r: r['speedup'], reverse=True)[:10]}
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, subset in (('comparison.csv', rows), ('family_changed.csv', changed),
                         ('family_unchanged.csv', unchanged)):
        with (HERE / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('worst_changed', 'best_changed')}, indent=2))


if __name__ == '__main__':
    main()
