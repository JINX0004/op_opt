#!/usr/bin/env python3
"""Measure deterministic Split-K outer-N panel cost at fixed source tiling."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('forced_family_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

OUT = ROOT / 'cases/fp32_deterministic_panel_v1'
runner.OUT = OUT
DEVICE = 5
M, K = 8192, 32768
NS = tuple(range(384, 3841, 384)) + (4096,)
AUDIT_NS = (4096,) + tuple(range(384, 3841, 384))
TILE_FIELDS = ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN',
               'singleCoreK', 'baseM', 'baseN', 'baseK', 'stepM', 'stepN',
               'stepKa', 'stepKb', 'depthA1', 'depthB1', 'block_dim',
               'workspace_bytes', 'nd2nzA', 'nd2nzB', 'tiling_key')
METRICS = ('aic_total_time', 'aic_mac_time', 'aic_mte1_time',
           'aic_mte2_time', 'aic_fixpipe_time', 'aic_scalar_time')


def audit_tile(n):
    index = AUDIT_NS.index(n)
    audit_device = 4 + index % 4
    path = OUT / f'device{audit_device}' / f'{index:03d}_{M}x{n}x{K}' \
        / 'audit/proposed/result.json'
    row = json.loads(path.read_text())
    tile = row['tiling']
    expected = (3, 2816, 384, 192, 128, 128, 64, 1, 3, 3, 3, 6,
                9, 20, 193986560, 0, 0, 65584)
    observed = tuple(tile[key] for key in TILE_FIELDS)
    if observed != expected or row['event']['max_relative_error'] > .02:
        raise RuntimeError(('tiling/output audit', n, observed))
    return dict(zip(TILE_FIELDS, observed))


def profile(n, round_index):
    directory = OUT / 'timings' / f'round{round_index}' / str(n)
    output = directory / 'result.json'
    if output.exists():
        return json.loads(output.read_text())
    runner.require_idle(DEVICE, directory, 'before')
    env = runner.environment('proposed', 3, 3)
    cmd = ['msprof', '--output=' + str(directory / 'msprof'),
           '--task-time=on', '--ai-core=on', '--aic-metrics=PipeUtilization',
           'env', str(ROOT / 'fp32_bench'), '--api', 'v3cached',
           '--device', str(DEVICE), '--m', str(M), '--n', str(n),
           '--k', str(K), '--warmup', '5', '--batches', '1',
           '--repetitions', '5']
    with (directory / 'stdout.log').open('w') as stdout, \
            (directory / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=1800)
    runner.require_idle(DEVICE, directory, 'after')
    if proc.returncode:
        raise RuntimeError((n, round_index, proc.returncode,
                            (directory / 'stderr.log').read_text()[-1500:]))
    event = runner.harness_event(directory / 'stdout.log')
    if event['max_relative_error'] > .02 or event['repetitions'] != 5:
        raise RuntimeError(('output audit', n, round_index, event))
    parsed = subprocess.run(
        ['python3', str(ROOT / 'bf16_online_vs_master/parse_profile.py'),
         str(directory / 'msprof'), '--last', '5', '--kernel-type', 'MIX_AIC'],
        capture_output=True, text=True, check=True)
    timing = json.loads(parsed.stdout)
    if timing['kernel_count'] != 5:
        raise RuntimeError(('task count', n, timing))
    database = glob.glob(str(directory / 'msprof/**/ai_core_op_summary.db'),
                         recursive=True)
    if len(database) != 1:
        raise RuntimeError(('metrics database', n, database))
    with sqlite3.connect(database[0]) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        rows = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    if len(rows) < 5 or any(metric not in columns for metric in METRICS):
        raise RuntimeError(('missing metrics', n, len(rows), columns))
    rows = rows[-5:]
    metrics = {metric: statistics.median(float(row[columns.index(metric)])
                                  for row in rows) for metric in METRICS}
    result = {'shape': [M, n, K], 'panels': (n + 383) // 384,
              'device': DEVICE, 'round': round_index,
              'host_sha': runner.HASHES['proposed'],
              'kernel_us': timing['median_us'],
              'kernel_tasks_us': timing['durations_us'],
              'max_relative_error': event['max_relative_error'],
              'metrics_us': metrics}
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    host = runner.PACKAGES['proposed'] / \
        'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
    if runner.sha(host) != runner.HASHES['proposed']:
        raise RuntimeError('forced-family host package hash changed')
    for n in NS:
        audit_tile(n)
    print('all source tiles audited', flush=True)
    records = []
    for round_index, ns in enumerate((NS, tuple(reversed(NS)))):
        for n in ns:
            row = profile(n, round_index)
            records.append(row)
            print('round', round_index, 'N', n, 'panels', row['panels'],
                  'kernel_us', round(row['kernel_us'], 3),
                  'MTE2_us', round(row['metrics_us']['aic_mte2_time'], 3),
                  'Fixpipe_us', round(row['metrics_us']['aic_fixpipe_time'], 3),
                  flush=True)
    (OUT / 'panel_records.json').write_text(json.dumps(records, indent=2) + '\n')
