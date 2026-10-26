#!/usr/bin/env python3
"""Analyze randomized paired optimizer/default BASE kernel task times."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


T95 = {3: 3.1824463053, 5: 2.5705818366, 6: 2.4469118511}


def interval(values):
    mean = statistics.mean(values)
    half = T95[len(values) - 1] * statistics.stdev(values) / math.sqrt(len(values))
    return [math.exp(mean - half), math.exp(mean + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('--rounds', type=int, default=6)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.raw.read_text().splitlines()]
    by_pair = defaultdict(dict)
    for row in rows:
        key = (tuple(row['shape']), row['round'])
        if row['arm'] in by_pair[key]:
            raise ValueError(f'duplicate arm {key} {row["arm"]}')
        by_pair[key][row['arm']] = row['case']['kernel_median_us']
    by_shape = defaultdict(list)
    for (shape, round_index), pair in by_pair.items():
        if set(pair) != {'optimizer', 'default'}:
            raise ValueError(f'incomplete pair: {shape} round {round_index}')
        by_shape[shape].append((round_index, pair['optimizer'], pair['default']))
    summary = {'raw_runs': len(rows), 'shapes': []}
    shape_log_ratios = []
    for shape, pairs in sorted(by_shape.items()):
        pairs.sort()
        if len(pairs) != args.rounds:
            raise ValueError(f'expected {args.rounds} independent rounds: {shape}')
        log_ratios = [math.log(optimizer / default)
                      for _, optimizer, default in pairs]
        shape_log_ratios.append(statistics.mean(log_ratios))
        summary['shapes'].append({
            'shape': shape,
            'rounds': [{'round': r, 'optimizer_us': o, 'default_us': d,
                        'optimizer_over_default': o / d}
                       for r, o, d in pairs],
            'optimizer_median_us': statistics.median(p[1] for p in pairs),
            'default_median_us': statistics.median(p[2] for p in pairs),
            'geometric_mean_optimizer_over_default':
                math.exp(statistics.mean(log_ratios)),
            'two_sided_95pct_t_ci_optimizer_over_default':
                interval(log_ratios) if len(log_ratios) - 1 in T95 else None,
            'optimizer_faster_rounds': sum(value < 0 for value in log_ratios),
        })
    summary['across_declared_shapes'] = {
        'count': len(shape_log_ratios),
        'geometric_mean_optimizer_over_default':
            math.exp(statistics.mean(shape_log_ratios)),
        'two_sided_95pct_t_ci_optimizer_over_default':
            interval(shape_log_ratios)
            if len(shape_log_ratios) - 1 in T95 else None,
        'optimizer_faster_shapes': sum(value < 0 for value in shape_log_ratios),
        'one_sided_exact_sign_p_default_faster':
            sum(math.comb(len(shape_log_ratios), k)
                for k in range(sum(value > 0 for value in shape_log_ratios),
                               len(shape_log_ratios) + 1)) /
            (2 ** len(shape_log_ratios)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
