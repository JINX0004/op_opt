#!/usr/bin/env python3
"""Summarize source-backed FP32 master/model-family kernel comparisons."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = Path('/home/xianfei/solvebase_bench_20261025/cases/fp32_100_source_family_v1')


def gmean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def summarize(rows):
    if not rows:
        return None
    ratios = [row['speedup'] for row in rows]
    return {'count': len(rows), 'geomean_speedup': gmean(ratios),
            'median_speedup': statistics.median(ratios),
            'model_faster': sum(x > 1 for x in ratios),
            'master_faster': sum(x < 1 for x in ratios),
            'same_timing': sum(x == 1 for x in ratios)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--records', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    shapes = json.loads((HERE / 'dataset.json').read_text())
    groups = json.loads((HERE / 'groups.json').read_text())
    if args.records:
        records = json.loads(args.records.read_text())
        failures = []
    else:
        records = [json.loads(path.read_text())
                   for path in args.raw.glob('device*/*/record.json')]
        failures = [str(path) for path in args.raw.glob('device*/*/failure.json')
                    if not (path.parent / 'record.json').exists()]
    by_index = {row['index']: row for row in records}
    if len(by_index) != len(records):
        raise RuntimeError('duplicate case index')
    if (len(records) != 100 or failures) and not args.allow_incomplete:
        raise RuntimeError(f'incomplete experiment: {len(records)}/100, failures={len(failures)}')
    cases = []
    max_error = 0.0
    for index, record in sorted(by_index.items()):
        shape = shapes[index]
        if record['shape'] != shape or record['device'] != 4 + index % 4:
            raise RuntimeError(f'index/shape/device mismatch at {index}')
        audits = record['audits']
        a, b, control = (audits[x]['tiling'] for x in
                         ('master', 'proposed', 'research_default'))
        if any(a.get(field) != control.get(field) for field in
               ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN',
                'singleCoreK', 'baseM', 'baseN', 'baseK', 'stepKa', 'stepKb')):
            raise RuntimeError(f'research default/master mismatch at {index}')
        if b['tilingEnableSplitCore'] != record['model']['mode']:
            raise RuntimeError(f'forced family mismatch at {index}')
        measurements = defaultdict(list)
        for item in record['measurements']:
            if item['event']['max_relative_error'] > .02 or item['profile']['kernel_count'] != 5:
                raise RuntimeError(f'profile failure at {index}')
            max_error = max(max_error, item['event']['max_relative_error'])
            measurements[item['arm']].append(item)
        if set(measurements) != {'master', 'proposed'} or \
                len(measurements['master']) != len(measurements['proposed']):
            raise RuntimeError(f'unpaired profiles at {index}')
        rounds = len(measurements['master'])
        if rounds != (2 if record['changed_tiling'] else 1):
            raise RuntimeError(f'incorrect repeat count at {index}')
        master_rounds = [item['profile']['median_us'] for item in measurements['master']]
        proposed_rounds = [item['profile']['median_us'] for item in measurements['proposed']]
        master_us = statistics.median(master_rounds)
        proposed_us = statistics.median(proposed_rounds)
        speedup = master_us / proposed_us
        cases.append({'index': index, 'group': 'fp16_supported' if index < 80 else 'fp32_boundary',
                      'shape': 'x'.join(map(str, shape)), 'device': record['device'],
                      'master_mode': a['tilingEnableSplitCore'],
                      'model_mode': record['model']['mode'],
                      'changed_family': a['tilingEnableSplitCore'] != b['tilingEnableSplitCore'],
                      'changed_tiling': record['changed_tiling'],
                      'master_us': master_us, 'model_family_us': proposed_us,
                      'speedup': speedup, 'rounds': rounds,
                      'master_core': f"{a['singleCoreM']}x{a['singleCoreN']}x{a['singleCoreK']}",
                      'model_family_core': f"{b['singleCoreM']}x{b['singleCoreN']}x{b['singleCoreK']}",
                      'model_objective_us': record['model']['objective_us'],
                      'master_rounds_us': master_rounds,
                      'model_family_rounds_us': proposed_rounds})
    changed = [row for row in cases if row['changed_family']]
    unchanged = [row for row in cases if not row['changed_family']]
    other = [row for row in cases if row['master_mode'] not in (0, 3)]
    supported = [row for row in cases if row['master_mode'] in (0, 3)]
    rng = random.Random(91020261106)
    bootstrap = sorted(gmean([rng.choice(cases)['speedup'] for _ in cases])
                       for _ in range(20000)) if cases else []
    result = {'complete': len(cases) == 100 and not failures,
              'record_count': len(cases), 'failures': failures,
              'all': summarize(cases), 'family_changed': summarize(changed),
              'family_unchanged': summarize(unchanged),
              'master_base_or_deterministic': summarize(supported),
              'changed_within_supported': summarize([row for row in changed
                                                      if row['master_mode'] in (0, 3)]),
              'master_other_family': summarize(other),
              'groups': {name: summarize([row for row in cases if start <= row['index'] < end])
                         for name, (start, end) in groups.items()},
              'shape_bootstrap_95pct': [bootstrap[500], bootstrap[19500]] if bootstrap else None,
              'max_relative_error': max_error,
              'worst_model_family': sorted(cases, key=lambda row: row['speedup'])[:20],
              'best_model_family': sorted(cases, key=lambda row: row['speedup'], reverse=True)[:20]}
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    fieldnames = [key for key in cases[0] if not key.endswith('_rounds_us')] if cases else []
    for name, rows in [('comparison.csv', cases), ('family_changed.csv', changed),
                       ('family_unchanged.csv', unchanged), ('master_other_family.csv', other)]:
        with (HERE / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('worst_model_family', 'best_model_family')}, indent=2))


if __name__ == '__main__':
    main()
