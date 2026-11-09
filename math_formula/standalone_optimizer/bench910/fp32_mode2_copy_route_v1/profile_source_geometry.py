#!/usr/bin/env python3
"""Check mode-2 GM-to-L1 traffic at two additional source tile geometries."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = ROOT / 'fp32_mode2_copy_route_v1'
spec = importlib.util.spec_from_file_location('sweep', ROOT / 'fp32_mode2_sweep_v1/run.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)
runner = sweep.runner
OUT = HERE / 'source_geometry_profiles'


def one(shape, device):
    m, n, k = shape
    dest = OUT / f'{m}x{n}x{k}'
    dest.mkdir(parents=True, exist_ok=True)
    runner.require_idle(device, dest, 'before')
    audit = runner.audit(0, shape, device, 'sweep', 2)
    tile = audit['tiling']
    if tile['tilingEnableSplitCore'] != 2 or tile['baseM'] != 128 or tile['baseN'] != 128:
        raise RuntimeError((shape, tile))
    env = runner.environment('sweep', 2, 3)
    cmd = ['msprof', '--output=' + str(dest / 'msprof'), '--task-time=on',
           '--ai-core=on', '--aic-metrics=MemoryAccess', 'env',
           str(ROOT / 'fp32_bench'), '--api', 'v3cached', '--device', str(device),
           '--m', str(m), '--n', str(n), '--k', str(k),
           '--warmup', '3', '--batches', '1', '--repetitions', '3']
    with (dest / 'stdout.log').open('w') as stdout, \
         (dest / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=900)
    runner.require_idle(device, dest, 'after')
    if proc.returncode:
        raise RuntimeError((dest, (dest / 'stderr.log').read_text()[-2000:]))
    event = runner.harness_event(dest / 'stdout.log')
    if event['max_relative_error'] > .02:
        raise RuntimeError(event)
    db_files = glob.glob(str(dest / 'msprof/**/ai_core_op_summary.db'), recursive=True)
    if len(db_files) != 1:
        raise RuntimeError((dest, db_files))
    with sqlite3.connect(db_files[0]) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        values = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    if len(values) < 3:
        raise RuntimeError((dest, len(values)))
    recent = values[-3:]
    metrics = {col: statistics.median(float(row[index]) for row in recent)
               for index, col in enumerate(columns)
               if col not in ('task_id', 'stream_id', 'subtask_id', 'batch_id')
               and all(row[index] is not None for row in recent)}
    m_tiles = (m + tile['singleCoreM'] - 1) // tile['singleCoreM']
    n_tiles = (n + tile['singleCoreN'] - 1) // tile['singleCoreN']
    predicted_kib = (4 * k * (m * n_tiles + n * m_tiles)) / 1024
    row = {'shape': shape, 'device': device, 'tile': tile,
           'gm_to_l1_kib': metrics['aic_GM_to_L1_datas'],
           'one_a_one_b_per_panel_kib': predicted_kib}
    (dest / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
    print(shape, tile['singleCoreM'], tile['singleCoreN'],
          row['gm_to_l1_kib'], predicted_kib, flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda x: one(*x),
                             (((3072, 1024, 32768), 5),
                              ((2048, 2048, 32768), 7))))
    (HERE / 'source_geometry_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
