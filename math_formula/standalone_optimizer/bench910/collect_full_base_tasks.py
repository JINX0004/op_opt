#!/usr/bin/env python3
"""Preserve underlying MatMulV3 task times for checked BASE comparisons."""

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/source_compatible_base_v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'repeat'), default='repeat')
    parser.add_argument('--runs', type=int, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in (ROOT / f'{args.phase}_raw.jsonl')
            .read_text().splitlines()]
    if len(rows) != args.runs:
        raise RuntimeError(f'expected {args.runs} runs, got {len(rows)}')
    with (ROOT / f'{args.phase}_tasks.jsonl').open('w') as output:
        for row in rows:
            label = 'x'.join(map(str, row['shape']))
            directory = (ROOT / args.phase /
                         f'r{row["round"]}_{label}_{row["arm"]}')
            profile = json.loads((directory / 'kernel_time.json').read_text())
            durations = profile['durations_us']
            if (len(durations) != 8 or profile['kernel_count'] != 8 or
                    abs(statistics.median(durations) -
                        row['case']['kernel_median_us']) > 1e-6):
                raise RuntimeError(f'inconsistent profiler tasks: {directory}')
            output.write(json.dumps({
                'shape': row['shape'], 'round': row['round'],
                'arm': row['arm'], 'durations_us': durations,
                'profile_csv': row['case']['profile_csv'],
            }, sort_keys=True) + '\n')
    print(f'{len(rows)} runs, {8 * len(rows)} device task times', flush=True)


if __name__ == '__main__':
    main()
