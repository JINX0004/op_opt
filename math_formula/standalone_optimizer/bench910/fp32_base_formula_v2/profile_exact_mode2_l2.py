#!/usr/bin/env python3
"""Measure L2 and memory counters for fixed FP32 mode-2 tiles near K=32768."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
spec = importlib.util.spec_from_file_location(
    'neighbors', ROOT / 'fp32_base_formula_v2/exact_mode2_k_neighbors/measure_exact_mode2_k_neighbors.py')
neighbors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neighbors)
runner = neighbors.runner
OUT = ROOT / 'cases/fp32_base_formula_v2/exact_mode2_k_neighbors/l2_profiles'


def summarize(db_path):
    with sqlite3.connect(db_path) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        values = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    if len(values) < 3:
        raise RuntimeError((db_path, len(values)))
    recent = values[-3:]
    return {column: statistics.median(float(row[index]) for row in recent)
            for index, column in enumerate(columns)
            if column not in ('task_id', 'stream_id', 'subtask_id', 'batch_id')
            and all(row[index] is not None for row in recent)}


def profile(n, k, metric, device):
    dest = OUT / f'N{n}_K{k}_{metric}'
    result_path = dest / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    runner.require_idle(device, dest, 'before')
    env = runner.environment('proposed', 2, 3)
    cmd = ['msprof', '--output=' + str(dest / 'msprof'), '--task-time=on',
           '--ai-core=on', '--aic-metrics=' + metric, 'env',
           str(ROOT / 'fp32_bench'), '--api', 'v3cached', '--device', str(device),
           '--m', '2560', '--n', str(n), '--k', str(k),
           '--warmup', '3', '--batches', '1', '--repetitions', '3']
    with (dest / 'stdout.log').open('w') as stdout, \
         (dest / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=1200)
    runner.require_idle(device, dest, 'after')
    if proc.returncode:
        raise RuntimeError((dest / 'stderr.log').read_text()[-2000:])
    event = runner.harness_event(dest / 'stdout.log')
    if event['dtype'] != 'fp32' or event['max_relative_error'] > .02:
        raise RuntimeError(event)
    db_files = glob.glob(str(dest / 'msprof/**/ai_core_op_summary.db'), recursive=True)
    if len(db_files) != 1:
        raise RuntimeError((dest, db_files))
    row = {'n': n, 'k': k, 'metric': metric, 'device': device,
           'metrics': summarize(db_files[0])}
    runner.write(result_path, row)
    print(n, k, metric, flush=True)
    return row


def one_n(n, device):
    return [profile(n, k, metric, device)
            for metric in ('MemoryAccess', 'L2Cache')
            for k in (32512, 32768, 33024)]


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = [row for group in pool.map(lambda x: one_n(*x), ((768, 6), (1024, 7)))
                for row in group]
    runner.write(OUT / 'metrics.json', rows)
    for row in rows:
        if row['metric'] == 'L2Cache':
            c = row['metrics']
            hit = c['aic_r0_read_cache_hit'] + c['aic_r1_read_cache_hit']
            miss = c['aic_r0_read_cache_miss_allocate'] + c['aic_r1_read_cache_miss_allocate']
            print(f'N={row["n"]} K={row["k"]} L2 read hit fraction={hit/(hit+miss):.4%}', flush=True)
        else:
            print(f'N={row["n"]} K={row["k"]} GM-to-L1 KB={row["metrics"]["aic_GM_to_L1_datas"]}', flush=True)


if __name__ == '__main__':
    main()
