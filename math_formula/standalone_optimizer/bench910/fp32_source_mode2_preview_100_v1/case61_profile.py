#!/usr/bin/env python3
"""Profile source mode 2 and online deterministic Split-K for case 61."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_source_mode2_preview_100_v1/run.py'
spec = importlib.util.spec_from_file_location('case61_runner', SOURCE)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
runner = experiment.runner
SHAPE = (8192, 4096, 32768)
DEVICE = 5
DEST = ROOT / 'cases/fp32_source_mode2_preview_100_v1/case61_pipe'
METRICS = ('aic_total_time', 'aic_mac_time', 'aic_mte1_time',
           'aic_mte2_time', 'aic_fixpipe_time', 'aic_scalar_time')


def one(arm, round_index):
    directory = DEST / f'round{round_index}' / arm
    output = directory / 'result.json'
    if output.exists():
        return json.loads(output.read_text())
    runner.require_idle(DEVICE, directory, 'before')
    env = runner.environment(arm, 0, 3)
    cmd = ['msprof', '--output=' + str(directory / 'msprof'),
           '--task-time=on', '--ai-core=on', '--aic-metrics=PipeUtilization',
           'env', str(ROOT / 'fp32_bench'), '--api', 'v3cached',
           '--device', str(DEVICE), '--m', str(SHAPE[0]),
           '--n', str(SHAPE[1]), '--k', str(SHAPE[2]),
           '--warmup', '5', '--batches', '1', '--repetitions', '5']
    with (directory / 'stdout.log').open('w') as stdout, \
            (directory / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=1800)
    runner.require_idle(DEVICE, directory, 'after')
    if proc.returncode:
        raise RuntimeError((arm, proc.returncode,
                            (directory / 'stderr.log').read_text()[-1500:]))
    event = runner.harness_event(directory / 'stdout.log')
    if event['max_relative_error'] > .02 or event['repetitions'] != 5:
        raise RuntimeError(('output audit', arm, event))
    database = glob.glob(str(directory / 'msprof/**/ai_core_op_summary.db'),
                         recursive=True)
    if len(database) != 1:
        raise RuntimeError(('missing metrics database', arm, database))
    with sqlite3.connect(database[0]) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        rows = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    if len(rows) < 5 or any(metric not in columns for metric in METRICS):
        raise RuntimeError(('missing kernel metrics', arm, len(rows), columns))
    rows = rows[-5:]
    metrics = {metric: statistics.median(float(row[columns.index(metric)])
                                  for row in rows) for metric in METRICS}
    data = dict(arm=arm, round=round_index, shape=SHAPE, device=DEVICE,
                max_relative_error=event['max_relative_error'], metrics=metrics)
    output.write_text(json.dumps(data, indent=2) + '\n')
    return data


if __name__ == '__main__':
    records = []
    for round_index, order in enumerate((('master', 'online'),
                                         ('online', 'master'))):
        for arm in order:
            row = one(arm, round_index)
            records.append(row)
            print(arm, round_index, row['metrics'], flush=True)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')
