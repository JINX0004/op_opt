#!/usr/bin/env python3
"""Validate and summarize 100 source-backed BF16 master/online pairs."""

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path('/home/xianfei/solvebase_bench_20261025/cases/bf16_100_vs_master_v1')
GROUPS = ((0, 64, 'previous_general'), (64, 72, 'previous_mode5_boundary'),
          (72, 79, 'new_narrow_long_k'), (79, 86, 'new_family_transition'),
          (86, 93, 'new_mode5_boundary'), (93, 100, 'new_large_mn'))


def gmean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def summarize(rows):
    ratios = [row['speedup'] for row in rows]
    return {'cases': len(rows), 'geomean_master_over_online': gmean(ratios),
            'median_shape_speedup': statistics.median(ratios),
            'online_faster': sum(x > 1 for x in ratios),
            'online_slower': sum(x < 1 for x in ratios),
            'family_changes': sum(row['master_mode'] != row['online_mode'] for row in rows),
            'tiling_changes': sum(row['changed_tiling'] for row in rows)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--records', type=Path,
                        help='portable JSON array of the 100 per-shape records')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    shapes = json.loads((HERE / 'dataset.json').read_text())
    if args.records:
        source_rows = json.loads(args.records.read_text())
    else:
        source_rows = [json.loads(path.read_text())
                       for path in args.raw.glob('device*/*/record.json')]
    records = {}
    for row in source_rows:
        index = row['index']
        if index in records:
            raise RuntimeError(f'duplicate index {index}')
        if row['shape'] != shapes[index]:
            raise RuntimeError(f'shape mismatch at {index}')
        records[index] = row
    failures = ([] if args.records else
                [str(path) for path in args.raw.glob('device*/*/failure.json')
                 if not (path.parent / 'record.json').exists()])
    if (len(records) != 100 or failures) and not args.allow_incomplete:
        raise RuntimeError(f'incomplete: {len(records)}/100 records, {len(failures)} failures')
    if not records:
        raise RuntimeError('no records')

    cases = []
    for index in sorted(records):
        record = records[index]
        measurements = record['measurements']
        by_arm = defaultdict(list)
        for row in measurements:
            if row['max_relative_error'] > .02:
                raise RuntimeError(f'output error at {index}')
            by_arm[row['arm']].append(row)
        if set(by_arm) != {'master', 'online'}:
            raise RuntimeError(f'missing arm at {index}')
        rounds = {arm: len(rows) for arm, rows in by_arm.items()}
        if rounds['master'] != rounds['online']:
            raise RuntimeError(f'unpaired rounds at {index}')
        for arm, rows in by_arm.items():
            audited = record['audits'][arm]['tiling']
            if any(row['tiling']['tilingEnableSplitCore'] !=
                   audited['tilingEnableSplitCore'] for row in rows):
                raise RuntimeError(f'mode changed during timing at {index}')
        master = statistics.median(row['kernel_median_us'] for row in by_arm['master'])
        online = statistics.median(row['kernel_median_us'] for row in by_arm['online'])
        a = record['audits']['master']['tiling']
        b = record['audits']['online']['tiling']
        if record['changed_tiling'] or record['mode5_control']:
            if rounds['master'] != 2:
                raise RuntimeError(f'missing repeat at {index}')
        elif rounds['master'] != 1:
            raise RuntimeError(f'unexpected repeat at {index}')
        cases.append({'index': index, 'shape': 'x'.join(map(str, shapes[index])),
                      'device': record['device'], 'master_mode': a['tilingEnableSplitCore'],
                      'online_mode': b['tilingEnableSplitCore'],
                      'changed_tiling': record['changed_tiling'],
                      'master_us': master, 'online_us': online,
                      'speedup': master / online,
                      'rounds': rounds['master'],
                      'master_tile': f"{a['singleCoreM']}x{a['singleCoreN']}x{a['singleCoreK']}",
                      'online_tile': f"{b['singleCoreM']}x{b['singleCoreN']}x{b['singleCoreK']}"})

    rng = random.Random(91020261103)
    bootstrap = sorted(gmean([rng.choice(cases)['speedup'] for _ in cases])
                       for _ in range(20000))
    result = {'complete': len(cases) == 100 and not failures,
              'record_count': len(cases), 'failure_paths': failures,
              'all': summarize(cases),
              'family_changed': summarize([row for row in cases if row['master_mode'] != row['online_mode']])
              if any(row['master_mode'] != row['online_mode'] for row in cases) else None,
              'family_unchanged': summarize([row for row in cases if row['master_mode'] == row['online_mode']]),
              'groups': {name: summarize([row for row in cases if start <= row['index'] < end])
                         for start, end, name in GROUPS
                         if any(start <= row['index'] < end for row in cases)},
              'shape_bootstrap_95pct': [bootstrap[500], bootstrap[19500]],
              'max_relative_error': max(row['max_relative_error']
                                        for record in records.values()
                                        for row in record['measurements']),
              'changed_family_rows': [row for row in cases
                                      if row['master_mode'] != row['online_mode']],
              'worst_online_rows': sorted(cases, key=lambda row: row['speedup'])[:15]}
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    with (HERE / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=cases[0].keys(),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(cases)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('changed_family_rows', 'worst_online_rows')}, indent=2))


if __name__ == '__main__':
    main()
