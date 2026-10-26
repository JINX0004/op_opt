#!/usr/bin/env python3
"""Pair source-default deterministic Split-K times across the 65535 K limit."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


T95 = {2: 4.3026527297, 3: 3.1824463053, 4: 2.7764451052,
       5: 2.5705818366, 6: 2.4469118511}
LOW_K, HIGH_K = 65408, 65536


def log_interval(values):
    if len(values) < 3:
        return None
    half = T95[len(values) - 1] * statistics.stdev(values) / math.sqrt(len(values))
    mean = statistics.mean(values)
    return [math.exp(mean - half), math.exp(mean + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('--rounds', type=int, default=4)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.raw.read_text().splitlines()]
    paired = defaultdict(dict)
    for row in rows:
        m, n, k = row['shape']
        if k not in (LOW_K, HIGH_K):
            raise ValueError(f'unexpected K: {k}')
        key = (m, n, row['round'])
        if k in paired[key]:
            raise ValueError(f'duplicate pair arm: {key}, K={k}')
        paired[key][k] = row['case']['kernel_median_us']
    results = []
    for m, n in sorted({key[:2] for key in paired}):
        pairs = []
        for round_index in range(args.rounds):
            case = paired.get((m, n, round_index))
            if case is None or set(case) != {LOW_K, HIGH_K}:
                raise ValueError(f'missing pair: {m}x{n}, r{round_index}')
            low, high = case[LOW_K], case[HIGH_K]
            pairs.append({'round': round_index, 'low_us': low,
                          'high_us': high, 'high_over_low': high / low,
                          'excess_over_k_scaled_low_us':
                              high - low * HIGH_K / LOW_K})
        log_ratios = [math.log(p['high_over_low']) for p in pairs]
        results.append({
            'm': m, 'n': n, 'low_k': LOW_K, 'high_k': HIGH_K,
            'low_median_us': statistics.median(p['low_us'] for p in pairs),
            'high_median_us': statistics.median(p['high_us'] for p in pairs),
            'paired_geometric_mean_high_over_low':
                math.exp(statistics.mean(log_ratios)),
            'two_sided_95pct_paired_t_ci': log_interval(log_ratios),
            'mean_excess_over_k_scaled_low_us':
                statistics.mean(p['excess_over_k_scaled_low_us'] for p in pairs),
            'pairs': pairs,
        })
    print(json.dumps({'raw_runs': len(rows), 'rounds_per_shape': args.rounds,
                      'results': results}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
