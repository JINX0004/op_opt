#!/usr/bin/env python3
"""Preserve all profiler task times behind paired packet BASE run medians."""

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/packet_objective_v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('smoke', 'repeat'))
    parser.add_argument('--runs', required=True, type=int)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    rows = [json.loads(line) for line in
            (root / f'{args.phase}_raw.jsonl').read_text().splitlines()]
    if len(rows) != args.runs:
        raise ValueError(f'expected {args.runs} runs, got {len(rows)}')
    output = root / f'{args.phase}_tasks.jsonl'
    with output.open('w') as stream:
        for row in rows:
            label = 'x'.join(map(str, row['shape']))
            arm_suffix = f'_{row["arm"]}' if 'arm' in row else ''
            case_dir = root / args.phase / f'r{row["round"]}_{label}{arm_suffix}'
            profile = json.loads((case_dir / 'kernel_time.json').read_text())
            durations = profile['durations_us']
            if (len(durations) != 8 or profile['kernel_count'] != 8 or
                    abs(statistics.median(durations) -
                        row['case']['kernel_median_us']) > 1e-6):
                raise ValueError(f'profiler mismatch: {case_dir}')
            task = {
                'shape': row['shape'], 'round': row['round'],
                'durations_us': durations,
                'profile_csv': row['case']['profile_csv'],
            }
            if 'arm' in row:
                task['arm'] = row['arm']
            stream.write(json.dumps(task, sort_keys=True) + '\n')
    print(f'{len(rows)} runs; {8 * len(rows)} verified task times')


if __name__ == '__main__':
    main()
