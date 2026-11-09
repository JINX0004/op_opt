#!/usr/bin/env python3
"""Compare source-shaped A/B probe counters with MatMulV3's input traffic."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/fp32_mode2_copy_route_v1')
spec = importlib.util.spec_from_file_location('copy_run', ROOT / 'run.py')
copy_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copy_run)
OUT = ROOT / 'ab_profiles'


def one(n, device, pitch, metric, env):
    dest = OUT / f'N{n}_P{pitch}_{metric}'
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / 'result.json'
    if output.exists():
        return json.loads(output.read_text())
    copy_run.require_idle(device)
    cmd = ['msprof', '--output=' + str(dest / 'msprof'), '--task-time=on',
           '--ai-core=on', '--aic-metrics=' + metric, 'env',
           str(ROOT / 'build/ab_route_probe'), 'inline', str(pitch),
           '32256', '2560', str(n), '1', '20', str(device), '3', '3']
    with (dest / 'stdout.log').open('w') as stdout, \
         (dest / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=900)
    copy_run.require_idle(device)
    if proc.returncode:
        raise RuntimeError((cmd, (dest / 'stderr.log').read_text()[-2000:]))
    files = glob.glob(str(dest / 'msprof/**/ai_core_op_summary.db'), recursive=True)
    if len(files) != 1:
        raise RuntimeError((dest, files))
    with sqlite3.connect(files[0]) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        values = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    if len(values) < 3:
        raise RuntimeError((dest, len(values)))
    recent = values[-3:]
    metrics = {column: statistics.median(float(row[index]) for row in recent)
               for index, column in enumerate(columns)
               if column not in ('task_id', 'stream_id', 'subtask_id', 'batch_id')
               and all(row[index] is not None for row in recent)}
    row = {'n': n, 'pitch': pitch, 'metric': metric, 'device': device,
           'metrics': metrics}
    output.write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
    print(n, pitch, metric, flush=True)
    return row


def one_n(n, device, env):
    return [one(n, device, pitch, metric, env)
            for pitch in (32512, 32768)
            for metric in ('MemoryAccess', 'L2Cache', 'PipeUtilization')]


def main():
    env = copy_run.environment()
    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(lambda item: one_n(*item, env), ((768, 4), (1024, 6))))
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'ab_profile_metrics.json').write_text(
        json.dumps([row for group in groups for row in group], indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
