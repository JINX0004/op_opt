#!/usr/bin/env python3
"""Summarize independent source-matched tiling timings without refitting."""
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    path = Path(__file__).with_name('spatial_repeat_raw.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_case = defaultdict(list)
    by_shape_round = defaultdict(list)
    for row in rows:
        case = row['case']
        by_case[row['case_id']].append(case['kernel_median_us'])
        by_shape_round[(tuple(case['shape']), row['round'])].append(row)
    summary = {
        'independent_kernel_runs': len(rows),
        'per_case': {name: {'runs': len(times),
                            'median_us': statistics.median(times),
                            'min_us': min(times), 'max_us': max(times)}
                     for name, times in sorted(by_case.items())},
        'per_shape_round_winners': [],
    }
    for (shape, round_index), group in sorted(by_shape_round.items()):
        if len(group) < 2:
            continue
        best = min(group, key=lambda r: r['case']['kernel_median_us'])
        summary['per_shape_round_winners'].append({
            'shape': shape, 'round': round_index,
            'winner': best['case_id'],
            'observed_us': best['case']['kernel_median_us']})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
