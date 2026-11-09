#!/usr/bin/env python3
"""Hold M/N fixed while sweeping FP32 A row pitch with a forced BASE path."""

import importlib.util
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
OUT = ROOT / 'cases/fp32_base_formula_v2/stride_grid'
spec = importlib.util.spec_from_file_location('fp32_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = OUT

# The six 32512/32768/33024 and 48896/49152/49408 cases were already
# measured twice on the same device; this sweep targets independent pitches.
KS = (16128, 16384, 16640, 24576, 40960, 57344)


def direct(k, round_index):
    dest = OUT / f'K{k}' / f'round{round_index}'
    result = dest / 'result.json'
    if result.exists():
        return json.loads(result.read_text())
    runner.require_idle(4, dest, 'before')
    env = runner.environment('proposed', 0, 3)
    cmd = [str(ROOT / 'fp32_bench'), '--api', 'v3cached', '--device', '4',
           '--m', '2560', '--n', '768', '--k', str(k),
           '--warmup', '5', '--batches', '1', '--repetitions', '5']
    with (dest / 'stdout.jsonl').open('w') as stdout, \
         (dest / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=900)
    runner.require_idle(4, dest, 'after')
    if proc.returncode:
        raise RuntimeError((dest / 'stderr.log').read_text()[-1200:])
    event = runner.harness_event(dest / 'stdout.jsonl')
    if event['dtype'] != 'fp32' or event['max_relative_error'] > .02:
        raise RuntimeError(event)
    runner.write(result, event)
    return event


def main():
    rows = []
    for k in KS:
        audit = runner.audit(k, [2560, 768, k], 4, 'proposed', 0)
        rows.append({'k': k, 'tile': {field: audit['tiling'].get(field)
                                    for field in runner.TILE_FIELDS}, 'rounds_us': []})
    for round_index, order in enumerate((KS, tuple(reversed(KS)))):
        by_k = {row['k']: row for row in rows}
        for k in order:
            event = direct(k, round_index)
            by_k[k]['rounds_us'].append(event['median_us'])
            print(k, round_index, event['median_us'], flush=True)
    for row in rows:
        row['median_us'] = statistics.median(row['rounds_us'])
    runner.write(OUT / 'records.json', rows)
    print('completed', len(rows), flush=True)


if __name__ == '__main__':
    main()
