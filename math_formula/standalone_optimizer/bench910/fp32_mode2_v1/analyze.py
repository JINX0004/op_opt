#!/usr/bin/env python3
"""Combine the original FP32 100-case pairs with newly paired mode-2 cases."""

import json
import csv
import math
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'fp32_100_source_family_v1'


def median_us(record, arm):
    values = [item['profile']['median_us'] for item in record['measurements']
              if item['arm'] == arm]
    if not values:
        raise ValueError(f'missing {arm} timing')
    return statistics.median(values)


def summary(rows):
    ratios = [row['speedup'] for row in rows]
    return {'count': len(rows),
            'geomean_speedup': math.exp(statistics.mean(map(math.log, ratios))),
            'proposed_faster': sum(value > 1 for value in ratios),
            'master_faster': sum(value < 1 for value in ratios)}


def main():
    shapes = json.loads((PARENT / 'dataset.json').read_text())
    previous = json.loads((PARENT / 'records.json').read_text())
    decisions = list(csv.DictReader((HERE / 'mode2_decisions.csv').open()))
    validation = {row['index']: row for row in
                  json.loads((HERE / 'validation_records.json').read_text())}
    if len(shapes) != 100 or len(previous) != 100 or len(decisions) != 100:
        raise ValueError('incomplete 100-case input')
    rows = []
    validated = set()
    for index, (shape, record, decision) in enumerate(zip(shapes, previous, decisions)):
        new_mode = int(decision['mode'])
        old_mode = record['model']['mode']
        master_mode = record['audits']['master']['tiling']['tilingEnableSplitCore']
        if record['index'] != index or record['shape'] != shape or \
                [int(decision[key]) for key in ('m', 'n', 'k')] != shape:
            raise ValueError(f'shape/order mismatch at {index}')
        if new_mode == 2:
            if master_mode != 2 or index not in validation:
                raise ValueError(f'missing paired mode-2 validation at {index}')
            pair = validation[index]
            if pair['shape'] != shape or pair['device'] != 4 + index % 4:
                raise ValueError(f'validation provenance mismatch at {index}')
            a = pair['audits']['master']['tiling']
            b = pair['audits']['proposed']['tiling']
            fields = ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN',
                      'singleCoreK', 'baseM', 'baseN', 'baseK', 'stepM',
                      'stepN', 'stepKa', 'stepKb', 'block_dim', 'tiling_key')
            if any(a.get(field) != b.get(field) for field in fields):
                raise ValueError(f'mode-2 tiling mismatch at {index}')
            if any(item['event']['max_relative_error'] > .02 or
                   item['profile']['kernel_count'] != 5
                   for item in pair['measurements']):
                raise ValueError(f'mode-2 timing/correctness failure at {index}')
            master_us = median_us(pair, 'master')
            proposed_us = median_us(pair, 'proposed')
            validated.add(index)
        elif new_mode == old_mode:
            master_us = median_us(record, 'master')
            proposed_us = median_us(record, 'proposed')
        else:
            raise ValueError(f'new family {new_mode} lacks a paired run at {index}')
        rows.append({'index': index, 'shape': shape, 'master_mode': master_mode,
                     'old_model_mode': old_mode, 'new_model_mode': new_mode,
                     'master_us': master_us, 'proposed_us': proposed_us,
                     'speedup': master_us / proposed_us})
    if validated != set(validation):
        raise ValueError('unexpected validation record')
    changed = [row for row in rows if row['new_model_mode'] != row['master_mode']]
    result = {'all': summary(rows), 'family_changed': summary(changed),
              'family_unchanged': summary([row for row in rows if row not in changed]),
              'validated_mode2_indices': sorted(validated),
              'transitions': {f'{a}->{b}': count for (a, b), count in
                              sorted(Counter((row['master_mode'], row['new_model_mode'])
                                             for row in rows).items())},
              'rows': rows}
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
