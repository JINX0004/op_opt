#!/usr/bin/env python3
"""Compare source-kernel and A/B-probe GM/L2 counters at 191/192 KiB."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = ROOT / 'fp32_mode2_transfer_holdout_v1'
spec = importlib.util.spec_from_file_location('audit_tiles', HERE / 'audit_source_tiles.py')
audit_tiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_tiles)
runner = audit_tiles.runner
LOGICAL_K = 48384
PITCHES = (48896, 49152)
METRICS = ('MemoryAccess', 'L2Cache')
JOBS = {
    4: (('source', 3072, 1024), ('source', 8192, 4096)),
    5: (('probe', 3072, 1024), ('probe', 8192, 4096)),
    6: (('source', 2048, 2048),),
    7: (('probe', 2048, 2048),),
}


def one(kind, m, n, device, pitch, metric):
    dest = HERE / 'traffic_profiles' / f'{kind}_{m}x{n}_p{pitch}_{metric}'
    dest.mkdir(parents=True, exist_ok=True)
    result_path = dest / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    runner.require_idle(device, dest, 'before')
    tile = runner.audit(pitch, [m, n, pitch], device, 'sweep', 2)['tiling']
    if kind == 'source':
        env = runner.environment('sweep', 2, 3)
        cmd = [str(ROOT / 'fp32_bench'), '--api', 'v3cached',
               '--device', str(device), '--m', str(m), '--n', str(n),
               '--k', str(pitch), '--warmup', '3', '--batches', '1',
               '--repetitions', '3']
    else:
        env = runner.CANN_ENV.copy()
        cmd = [str(HERE / 'build/panel_ab_probe'), str(pitch),
               str(LOGICAL_K), str(m), str(n),
               str(tile['singleCoreM']), str(tile['singleCoreN']),
               str(tile['stepM'] * tile['baseM']),
               str(tile['singleCoreK']), str(int(tile['singleCoreN'] == n)),
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
        if kind != 'probe' or 'output token mismatch' not in \
                (dest / f'stderr_{attempt}.log').read_text():
            raise RuntimeError((dest, proc.returncode,
                                (dest / f'stderr_{attempt}.log').read_text()[-2000:]))
    else:
        raise RuntimeError((dest, 'three probe token failures'))
    runner.require_idle(device, dest, 'after')
    if kind == 'source':
        event = runner.harness_event(dest / f'stdout_{attempt}.log')
        if event['max_relative_error'] > .02:
            raise RuntimeError((dest, event))
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
    result = {'kind': kind, 'm': m, 'n': n, 'device': device,
              'pitch': pitch, 'metric': metric, 'metrics': metrics,
              'tile': {key: tile.get(key) for key in runner.TILE_FIELDS}}
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print(kind, m, n, pitch, metric, flush=True)
    return result


def on_device(device, metrics):
    return [one(kind, m, n, device, pitch, metric)
            for kind, m, n in JOBS[device]
            for pitch in PITCHES for metric in metrics]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=METRICS, required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(lambda d: on_device(d, (args.metric,)), JOBS))
    (HERE / f'traffic_{args.metric}.json').write_text(
        json.dumps([row for group in groups for row in group], indent=2) + '\n')


if __name__ == '__main__':
    main()
