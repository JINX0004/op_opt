#!/usr/bin/env python3
"""Validate BF16 narrow-B packet and source-backed boundary evidence."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def packet_slopes(rows, time_key):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row.get('round', 0), row['dtype'], row['operand'],
               row['tile_axis'], row['k_span'], row['cores'])
        grouped[key][row['copies']] = row[time_key]
    slopes = defaultdict(list)
    for key, by_count in grouped.items():
        if set(by_count) != {1, 25}:
            raise RuntimeError(f'missing paired packet count: {key}')
        _, dtype, operand, axis, span, cores = key
        slopes[(dtype, operand, axis, span, cores)].append(
            (by_count[25] - by_count[1]) / 24)
    return {f'{dtype}_{operand}{axis}_K{span}_c{cores}':
            {'per_packet_us': values, 'median_us': statistics.median(values)}
            for (dtype, operand, axis, span, cores), values in sorted(slopes.items())}


def main():
    events = [json.loads(line) for line in (HERE / 'event_sweep.jsonl').read_text().splitlines()]
    tasks = [json.loads(line) for line in (HERE / 'task_sweep.jsonl').read_text().splitlines()]
    records = json.loads((HERE / 'boundary_records.json').read_text())
    decisions = list(csv.DictReader((HERE / 'decision_sweep.csv').open(newline='')))
    if len(events) != 72 or len(tasks) != 8 or len(records) != 5 or len(decisions) != 100:
        raise RuntimeError('incomplete packet, boundary, or 100-shape decision evidence')
    event_slopes = packet_slopes(events, 'device_elapsed_us')
    task_slopes = packet_slopes(tasks, 'median_us')
    if len(event_slopes) != 12 or len(task_slopes) != 4:
        raise RuntimeError('incomplete packet slope groups')

    expected = [(16, 16, 65408), (32, 16, 65408), (16, 16, 65536),
                (16, 16, 102400), (256, 512, 16384)]
    boundary = []
    for index, record in enumerate(records):
        if tuple(record['shape']) != expected[index]:
            raise RuntimeError(f'wrong boundary shape at {index}')
        arms = {}
        for arm in ('master', 'generic', 'online'):
            audit = record['audits'][arm]
            timed = [row for row in record['measurements'] if row['arm'] == arm]
            if len(timed) != (2 if index < 2 else 1):
                raise RuntimeError(f'missing timing round at {index}, {arm}')
            if audit['event']['max_relative_error'] > .02 or any(
                    row['max_relative_error'] > .02 for row in timed):
                raise RuntimeError(f'output error at {index}, {arm}')
            mode = audit['tiling']['tilingEnableSplitCore']
            if any(row['tiling']['tilingEnableSplitCore'] != mode for row in timed):
                raise RuntimeError(f'family changed between audit and timing at {index}')
            arms[arm] = {
                'mode': mode,
                'tile': {key: audit['tiling'][key] for key in
                         ('singleCoreM', 'singleCoreN', 'singleCoreK',
                          'baseM', 'baseN', 'baseK', 'block_dim')},
                'round_us': [row['kernel_median_us'] for row in timed],
                'median_us': statistics.median(row['kernel_median_us'] for row in timed),
            }
        boundary.append({'shape': record['shape'], 'arms': arms,
                         'generic_over_online': arms['generic']['median_us'] /
                                                arms['online']['median_us']})
    changed = [row for row in decisions if row['generic_mode'] != row['narrow_mode']]
    if [tuple(map(int, (row['m'], row['n'], row['k']))) for row in changed] != [(16, 16, 65408)]:
        raise RuntimeError(f'unexpected 100-shape decision changes: {changed}')
    result = {'event_slopes': event_slopes, 'task_slopes': task_slopes,
              'boundary': boundary, 'decision_changes': changed,
              'preflight_fallbacks': sum(row['generic_mode'] == '-1' for row in decisions),
              'max_relative_error': max(
                  [record['audits'][arm]['event']['max_relative_error']
                   for record in records for arm in ('master', 'generic', 'online')] +
                  [row['max_relative_error'] for record in records
                   for row in record['measurements']])}
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print('BF16 B16 one-core event/task slopes',
          event_slopes['bf16_B16_K4096_c1']['median_us'],
          task_slopes['bf16_B16_K4096_c1']['median_us'])
    print('100-shape changed decisions', len(changed),
          'preflight fallbacks', result['preflight_fallbacks'])
    for row in boundary:
        print(row['shape'], {arm: (value['mode'], value['median_us'])
                              for arm, value in row['arms'].items()})


if __name__ == '__main__':
    main()
