#!/usr/bin/env python3
"""Check that the paired copy routes issue the intended GM-to-L1 traffic."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/fp32_mode2_copy_route_v1')
spec = importlib.util.spec_from_file_location('copy_run', ROOT / 'run.py')
copy_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copy_run)
OUT = ROOT / 'profiles'


def profile(route, pitch, metric, env):
    dest = OUT / f'{route}_{pitch}_{metric}'
    dest.mkdir(parents=True, exist_ok=True)
    result = dest / 'result.json'
    if result.exists():
        return json.loads(result.read_text())
    copy_run.require_idle(4)
    cmd = ['msprof', '--output=' + str(dest / 'msprof'), '--task-time=on',
           '--ai-core=on', '--aic-metrics=' + metric, 'env',
           str(copy_run.BINARY), route, str(pitch), '32256', '2560',
           '1', '20', '4', '3', '3']
    with (dest / 'stdout.log').open('w') as stdout, \
         (dest / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=600)
    copy_run.require_idle(4)
    if proc.returncode:
        raise RuntimeError((cmd, (dest / 'stderr.log').read_text()[-2000:]))
    db_files = glob.glob(str(dest / 'msprof/**/ai_core_op_summary.db'), recursive=True)
    if len(db_files) != 1:
        raise RuntimeError((dest, db_files))
    with sqlite3.connect(db_files[0]) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        values = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    if len(values) < 3:
        raise RuntimeError((dest, len(values)))
    recent = values[-3:]
    metrics = {column: statistics.median(float(row[index]) for row in recent)
               for index, column in enumerate(columns)
               if column not in ('task_id', 'stream_id', 'subtask_id', 'batch_id')
               and all(row[index] is not None for row in recent)}
    record = {'route': route, 'pitch': pitch, 'metric': metric,
              'metrics': metrics}
    result.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(route, pitch, metric, flush=True)
    return record


def main():
    env = copy_run.environment()
    records = [profile(route, pitch, metric, env)
               for route in ('inline', 'plain')
               for pitch in (32512, 32768)
               for metric in ('MemoryAccess', 'L2Cache', 'PipeUtilization')]
    (ROOT / 'profile_metrics.json').write_text(json.dumps(records, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
