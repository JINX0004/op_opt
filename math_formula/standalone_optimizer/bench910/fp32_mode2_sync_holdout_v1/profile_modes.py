#!/usr/bin/env python3
"""Check bytes and L2 hits for three copy synchronization modes."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = ROOT / 'fp32_mode2_sync_holdout_v1'
PRIOR = ROOT / 'fp32_mode2_transfer_holdout_v1'
spec = importlib.util.spec_from_file_location('audit_tiles', PRIOR / 'audit_source_tiles.py')
audit_tiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_tiles)
runner = audit_tiles.runner
LOGICAL_K = 48384
PITCH = 49152
GEOMETRIES = ((2816, 1024, 4), (8192, 4096, 6))


def one(m, n, device, mode, metric):
    dest = HERE / 'mode_profiles' / f'{m}x{n}_mode{mode}_{metric}'
    dest.mkdir(parents=True, exist_ok=True)
    result_path = dest / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    runner.require_idle(device, dest, 'before')
    audits = json.loads((PRIOR / 'tile_audits.json').read_text())
    tile = next(row['tile'] for row in audits if row['shape'] == [m, n, PITCH])
    env = runner.CANN_ENV.copy()
    cmd = [str(HERE / 'build/panel_ab_probe'), str(PITCH), str(LOGICAL_K),
           str(m), str(n), str(tile['singleCoreM']), str(tile['singleCoreN']),
           str(tile['stepM'] * tile['baseM']), str(tile['singleCoreK']),
           str(int(tile['singleCoreN'] == n)), str(mode),
           str(tile['block_dim']), str(device), '3']
    for attempt in range(3):
        output_dir = dest / f'msprof_attempt{attempt}'
        prof = ['msprof', '--output=' + str(output_dir), '--task-time=on',
                '--ai-core=on', '--aic-metrics=' + metric, 'env'] + cmd
        with (dest / f'stdout_{attempt}.log').open('w') as stdout, \
             (dest / f'stderr_{attempt}.log').open('w') as stderr:
            proc = subprocess.run(prof, env=env, cwd=ROOT, stdout=stdout,
                                  stderr=stderr, timeout=1500)
        if proc.returncode == 0:
            break
        if 'output token mismatch' not in (dest / f'stderr_{attempt}.log').read_text():
            raise RuntimeError((dest, proc.returncode,
                                (dest / f'stderr_{attempt}.log').read_text()[-2000:]))
    else:
        raise RuntimeError((dest, 'three token failures'))
    runner.require_idle(device, dest, 'after')
    files = glob.glob(str(output_dir / '**/ai_core_op_summary.db'), recursive=True)
    if len(files) != 1:
        raise RuntimeError((dest, files))
    with sqlite3.connect(files[0]) as conn:
        columns = [r[1] for r in conn.execute('pragma table_info(ai_core_metrics)')]
        values = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    if len(values) < 3:
        raise RuntimeError((dest, len(values)))
    recent = values[-3:]
    metrics = {col: statistics.median(float(row[index]) for row in recent)
               for index, col in enumerate(columns)
               if col not in ('task_id', 'stream_id', 'subtask_id', 'batch_id')
               and all(row[index] is not None for row in recent)}
    row = {'m': m, 'n': n, 'device': device, 'pitch': PITCH,
           'sync_mode': mode, 'metric': metric, 'metrics': metrics}
    result_path.write_text(json.dumps(row, indent=2) + '\n')
    print(m, n, mode, metric, flush=True)
    return row


def one_geometry(item):
    m, n, device = item
    return [one(m, n, device, mode, metric)
            for mode in (0, 1, 2)
            for metric in ('MemoryAccess', 'L2Cache')]


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(one_geometry, GEOMETRIES))
    (HERE / 'mode_profile_metrics.json').write_text(
        json.dumps([row for group in groups for row in group], indent=2) + '\n')


if __name__ == '__main__':
    main()
