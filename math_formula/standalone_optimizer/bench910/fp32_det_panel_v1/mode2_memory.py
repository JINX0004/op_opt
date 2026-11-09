#!/usr/bin/env python3
"""MemoryAccess profile of the source mode-2 comparison at N=4096."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_source_mode2_preview_100_v1/run.py'
spec = importlib.util.spec_from_file_location('source_preview_runner', SOURCE)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
runner = experiment.runner
DEST = ROOT / 'cases/fp32_deterministic_panel_v1/mode2_memory'
DEVICE = 5


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    runner.require_idle(DEVICE, DEST, 'before')
    env = runner.environment('master', 0, 3)
    cmd = ['msprof', '--output=' + str(DEST / 'msprof'),
           '--task-time=on', '--ai-core=on', '--aic-metrics=MemoryAccess',
           'env', str(ROOT / 'fp32_bench'), '--api', 'v3cached',
           '--device', str(DEVICE), '--m', '8192', '--n', '4096',
           '--k', '32768', '--warmup', '3', '--batches', '1',
           '--repetitions', '3']
    with (DEST / 'stdout.log').open('w') as stdout, \
            (DEST / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=1800)
    runner.require_idle(DEVICE, DEST, 'after')
    if proc.returncode:
        raise RuntimeError((DEST / 'stderr.log').read_text()[-1500:])
    event = runner.harness_event(DEST / 'stdout.log')
    if event['max_relative_error'] > .02 or event['repetitions'] != 3:
        raise RuntimeError(('output audit', event))
    dbs = glob.glob(str(DEST / 'msprof/**/ai_core_op_summary.db'),
                    recursive=True)
    if len(dbs) != 1:
        raise RuntimeError(('database', dbs))
    with sqlite3.connect(dbs[0]) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        rows = conn.execute('select * from ai_core_metrics order by task_id').fetchall()[-3:]
    if len(rows) != 3:
        raise RuntimeError(('tasks', len(rows)))
    metrics = {column: statistics.median(float(row[j]) for row in rows)
               for j, column in enumerate(columns)
               if column not in ('task_id', 'stream_id', 'subtask_id', 'batch_id')
               and all(row[j] is not None for row in rows)}
    result = {'shape': [8192, 4096, 32768], 'device': DEVICE,
              'family': 'single_core_split_k_mode2',
              'max_relative_error': event['max_relative_error'],
              'metrics': metrics}
    (DEST / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print({k: metrics[k] for k in
           ('aic_GM_to_L1_datas', 'aic_L0C_to_GM_datas', 'aic_total_time')})
