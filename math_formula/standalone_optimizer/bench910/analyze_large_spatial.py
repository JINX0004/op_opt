#!/usr/bin/env python3
"""Analyze independent randomized 910B3 large-shape BASE M-tile pairs."""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


T95_DF5 = 2.5705818366


def main():
    path = Path(__file__).with_name('large_paired_m64_raw.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_pair = defaultdict(dict)
    for row in rows:
        key = (tuple(row['shape']), row['round'])
        if row['arm'] in by_pair[key]:
            raise ValueError(f'duplicate arm {key} {row["arm"]}')
        by_pair[key][row['arm']] = row['case']['kernel_median_us']
    by_shape = defaultdict(list)
    for (shape, round_index), pair in by_pair.items():
        if set(pair) != {'default', 'base_m_64'}:
            raise ValueError(f'incomplete pair: {shape} round {round_index}')
        by_shape[shape].append((round_index,
                                pair['default'], pair['base_m_64']))
    summary = {'raw_runs': len(rows), 'shapes': []}
    for shape, pairs in sorted(by_shape.items()):
        pairs.sort()
        if len(pairs) != 6:
            raise ValueError(f'expected six independent rounds: {shape}')
        log_alt_over_default = [math.log(alt / default)
                                for _, default, alt in pairs]
        mean = statistics.mean(log_alt_over_default)
        half_width = T95_DF5 * statistics.stdev(log_alt_over_default) / math.sqrt(6)
        positive = sum(value > 0 for value in log_alt_over_default)
        # Exact one-sided paired sign test: H0 says the default is no more
        # likely to beat the alternative than a fair coin on this shape.
        sign_p = sum(math.comb(6, count) for count in range(positive, 7)) / 64
        summary['shapes'].append({
            'shape': shape,
            'rounds': [{'round': round_index,
                        'default_us': default,
                        'base_m_64_us': alt,
                        'alternative_over_default': alt / default}
                       for round_index, default, alt in pairs],
            'default_median_us': statistics.median(p[1] for p in pairs),
            'base_m_64_median_us': statistics.median(p[2] for p in pairs),
            'geometric_mean_alternative_over_default': math.exp(mean),
            'two_sided_95pct_t_ci_alternative_over_default':
                [math.exp(mean - half_width), math.exp(mean + half_width)],
            'default_faster_rounds': positive,
            'one_sided_sign_p_default_faster': sign_p,
        })
    # Holm control for the three declared shape-specific sign tests.
    ordered = sorted(summary['shapes'],
                     key=lambda row: row['one_sided_sign_p_default_faster'])
    largest_adjusted = 0.0
    for index, row in enumerate(ordered):
        largest_adjusted = max(largest_adjusted,
                               min(1.0, (len(ordered) - index) *
                                   row['one_sided_sign_p_default_faster']))
        row['holm_adjusted_sign_p'] = largest_adjusted
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
