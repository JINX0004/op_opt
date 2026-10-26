#!/usr/bin/env python3
"""Preserve all selected default MatMulV3 profiler task times on the 910B3."""
import csv
import json
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/optimizer_vs_default')


def main():
    rows = [json.loads(s) for s in (ROOT / 'default_cases.jsonl').read_text().splitlines()]
    output = ROOT / 'default_tasks.jsonl'
    with output.open('w') as stream:
        for row in rows:
            if row['status'] != 'eligible':
                continue
            case = row['case']
            with Path(case['profile_csv']).open(newline='') as csvfile:
                tasks = [r for r in csv.DictReader(csvfile)
                         if 'MatMulV3' in r['kernel_name'] and
                         r['kernel_type'] == 'KERNEL_AICORE']
            durations = [float(r['task_time(us)']) for r in tasks[-8:]]
            if len(durations) != 8:
                raise RuntimeError(f'incomplete profiler tasks: {row["shape"]}')
            stream.write(json.dumps({'shape': row['shape'],
                                     'kernel_durations_us': durations},
                                    sort_keys=True) + '\n')
    print(output)


if __name__ == '__main__':
    main()
