#!/usr/bin/env python3
"""Preserve the eight underlying profiler tasks in each large paired run."""
import csv
import json
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/optimizer_vs_default_large/paired_m64')


def main():
    rows = [json.loads(s) for s in (ROOT / 'raw.jsonl').read_text().splitlines()]
    if len(rows) != 36:
        raise RuntimeError(f'expected 36 completed runs, got {len(rows)}')
    with (ROOT / 'raw_tasks.jsonl').open('w') as output:
        for row in rows:
            with Path(row['case']['profile_csv']).open(newline='') as csvfile:
                tasks = [r for r in csv.DictReader(csvfile)
                         if 'MatMulV3' in r['kernel_name'] and
                         r['kernel_type'] == 'KERNEL_AICORE']
            durations = [float(r['task_time(us)']) for r in tasks[-8:]]
            if len(durations) != 8:
                raise RuntimeError(f'incomplete profiler tasks: {row["shape"]}')
            output.write(json.dumps({'shape': row['shape'],
                                     'round': row['round'], 'arm': row['arm'],
                                     'kernel_durations_us': durations},
                                    sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
