#!/usr/bin/env python3
"""Profile source-built FP32 BASE at neighboring K pitches on device 4."""

import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
OUT = ROOT / 'cases/fp32_base_formula_v2/pipeline_profiles'
spec = importlib.util.spec_from_file_location('fp32_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def run(k, metric):
    dest = OUT / f'K{k}_{metric}'
    result_path = dest / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    runner.require_idle(4, dest, 'before')
    env = runner.environment('proposed', 0, 3)
    cmd = ['msprof', '--output=' + str(dest / 'msprof'),
           '--task-time=on', '--ai-core=on', '--aic-metrics=' + metric,
           'env', str(ROOT / 'fp32_bench'), '--api', 'v3cached',
           '--device', '4', '--m', '2560', '--n', '768', '--k', str(k),
           '--warmup', '3', '--batches', '1', '--repetitions', '3']
    with (dest / 'stdout.log').open('w') as stdout, \
         (dest / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=1200)
    runner.require_idle(4, dest, 'after')
    if proc.returncode:
        raise RuntimeError((dest / 'stderr.log').read_text()[-2000:])
    event = runner.harness_event(dest / 'stdout.log')
    if event['dtype'] != 'fp32' or event['max_relative_error'] > .02:
        raise RuntimeError(event)
    data = {'k': k, 'metric': metric, 'event': event, 'path': str(dest / 'msprof')}
    runner.write(result_path, data)
    return data


if __name__ == '__main__':
    for metric in ('PipeUtilization', 'MemoryAccess', 'ResourceConflictRatio', 'L2Cache'):
        for k in (32512, 32768, 33024):
            print(run(k, metric), flush=True)
