#!/usr/bin/env python3
"""Analyze source-verified paired BASE timings for the packet objective."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


T95 = {3: 3.1824463053, 4: 2.7764451052, 5: 2.5705818366,
       6: 2.4469118511, 7: 2.3646242510, 9: 2.2621571628}


def ci_log(values):
    df = len(values) - 1
    if df not in T95:
        return None
    center = statistics.mean(values)
    half = T95[df] * statistics.stdev(values) / math.sqrt(len(values))
    return [math.exp(center - half), math.exp(center + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('selection', type=Path)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--rounds', type=int, default=6)
    args = parser.parse_args()
    selected = {tuple(row['shape']): row for row in
                json.loads(args.selection.read_text())['rows']
                if row['status'] == 'selected'}
    records = [json.loads(line) for line in args.raw.read_text().splitlines()]
    paired = defaultdict(dict)
    for record in records:
        key = tuple(record['shape']), record['round']
        if key[0] not in selected or record['arm'] in paired[key]:
            raise ValueError(f'unexpected or duplicate run: {key}')
        if (record['optimizer_tile'] != selected[key[0]]['optimizer_tile'] or
                record['default_tile'] != selected[key[0]]['default_tile']):
            raise ValueError(f'selection changed during repeat: {key}')
        paired[key][record['arm']] = record['case']['kernel_median_us']
    results = []
    for shape, choice in selected.items():
        pairs = []
        for round_index in range(args.rounds):
            pair = paired.get((shape, round_index))
            if pair is None or set(pair) != {'optimizer', 'default'}:
                raise ValueError(f'missing pair: {shape} r{round_index}')
            pairs.append(pair)
        logs = [math.log(pair['optimizer'] / pair['default'])
                for pair in pairs]
        results.append({
            'shape': shape, 'default_tile': choice['default_tile'],
            'optimizer_tile': choice['optimizer_tile'],
            'default_median_us': statistics.median(p['default'] for p in pairs),
            'optimizer_median_us': statistics.median(p['optimizer'] for p in pairs),
            'geometric_mean_optimizer_over_default':
                math.exp(statistics.mean(logs)),
            'two_sided_95pct_paired_t_ci': ci_log(logs),
            'optimizer_faster_pairs': sum(x < 0 for x in logs),
            'pairs': [{'round': index, **pair,
                       'optimizer_over_default':
                           pair['optimizer'] / pair['default']}
                      for index, pair in enumerate(pairs)],
        })
    shape_means = [math.log(row['geometric_mean_optimizer_over_default'])
                   for row in results]
    print(json.dumps({
        'raw_runs': len(records), 'rounds_per_shape': args.rounds,
        'timed_shapes': len(results), 'results': results,
        'across_timed_shapes_geometric_mean_optimizer_over_default':
            math.exp(statistics.mean(shape_means)) if shape_means else None,
        'across_timed_shapes_95pct_t_ci':
            ci_log(shape_means) if len(shape_means) >= 2 else None,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
