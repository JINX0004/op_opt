#!/usr/bin/env python3
"""Read deterministic Split-K memory counters at 1, 10, 11 panels."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('forced_family_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
OUT = ROOT / 'cases/fp32_deterministic_panel_v1/memory'
DEVICE = 5
SHAPES = ((8192, 384, 32768), (8192, 3840, 32768),
          (8192, 4096, 32768))
METRICS = ('MemoryAccess', 'L2Cache')


def one(shape, metric):
    n = shape[1]
    directory = OUT / metric / str(n)
    output = directory / 'result.json'
    if output.exists():
        return json.loads(output.read_text())
    runner.require_idle(DEVICE, directory, 'before')
    env = runner.environment('proposed', 3, 3)
    cmd = ['msprof', '--output=' + str(directory / 'msprof'),
           '--task-time=on', '--ai-core=on', '--aic-metrics=' + metric,
           'env', str(ROOT / 'fp32_bench'), '--api', 'v3cached',
           '--device', str(DEVICE), '--m', str(shape[0]),
           '--n', str(shape[1]), '--k', str(shape[2]),
           '--warmup', '3', '--batches', '1', '--repetitions', '3']
    with (directory / 'stdout.log').open('w') as stdout, \
            (directory / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=1800)
    runner.require_idle(DEVICE, directory, 'after')
    if proc.returncode:
        raise RuntimeError((shape, metric, proc.returncode,
                            (directory / 'stderr.log').read_text()[-1500:]))
    event = runner.harness_event(directory / 'stdout.log')
    if event['max_relative_error'] > .02 or event['repetitions'] != 3:
        raise RuntimeError(('output audit', shape, metric, event))
    databases = glob.glob(str(directory / 'msprof/**/ai_core_op_summary.db'),
                          recursive=True)
    if len(databases) != 1:
        raise RuntimeError(('metrics database', shape, metric, databases))
    with sqlite3.connect(databases[0]) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        rows = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    if len(rows) < 3:
        raise RuntimeError(('metrics rows', shape, metric, len(rows)))
    rows = rows[-3:]
    metrics = {column: statistics.median(float(row[j]) for row in rows)
               for j, column in enumerate(columns)
               if column not in ('task_id', 'stream_id', 'subtask_id', 'batch_id')
               and all(row[j] is not None for row in rows)}
    data = {'shape': shape, 'device': DEVICE, 'metric': metric,
            'max_relative_error': event['max_relative_error'],
            'metrics': metrics}
    output.write_text(json.dumps(data, indent=2) + '\n')
    return data


if __name__ == '__main__':
    records = []
    for metric in METRICS:
        for shape in SHAPES:
            data = one(shape, metric)
            records.append(data)
            print(metric, shape[1], 'done', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
