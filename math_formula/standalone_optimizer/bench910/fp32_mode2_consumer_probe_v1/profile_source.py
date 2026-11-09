#!/usr/bin/env python3
"""Profile the real fixed-work mode-2 source path at all three A pitches."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'fp32_mode2_in_context_pitch_v1/run.py'
spec = importlib.util.spec_from_file_location('fixed_work_runner', SOURCE)
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
PACKAGE = ROOT / 'fp32_mode2_in_context_pitch_v1/ops_nn_pitch_overlay_package'
PITCHES = (48896, 49152, 49408)
METRICS = ('aic_total_time', 'aic_mte1_time', 'aic_mte2_time',
           'aic_mac_time', 'aic_fixpipe_time', 'aic_scalar_time')


def one(m, n, device, pitch, round_index):
    dest = HERE / 'source_profiles' / f'{m}x{n}_device{device}' / \
           f'{pitch}_round{round_index}'
    dest.mkdir(parents=True, exist_ok=True)
    result_path = dest / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    fixed.require_idle(device)
    env = fixed.environment(PACKAGE, 3)
    cmd = fixed.benchmark_args(m, n, device, pitch, 3, 3)
    prof = ['msprof', '--output=' + str(dest / 'msprof'),
            '--task-time=on', '--ai-core=on',
            '--aic-metrics=PipeUtilization', 'env'] + cmd
    with (dest / 'stdout.log').open('w') as stdout, \
         (dest / 'stderr.log').open('w') as stderr:
        result = subprocess.run(prof, env=env, cwd=HERE,
                                stdout=stdout, stderr=stderr, timeout=1500)
    fixed.require_idle(device)
    if result.returncode:
        raise RuntimeError((dest, (dest / 'stderr.log').read_text()[-2000:]))
    event = fixed.harness_event(dest / 'stdout.log')
    if (event['dtype'] != 'fp32' or event['a_pitch'] != pitch or
            event['max_relative_error'] > .02):
        raise RuntimeError(('correctness gate', dest, event))
    databases = glob.glob(str(dest / 'msprof/**/ai_core_op_summary.db'), recursive=True)
    if len(databases) != 1:
        raise RuntimeError((dest, databases))
    with sqlite3.connect(databases[0]) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        values = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    if len(values) < 3:
        raise RuntimeError((dest, 'missing AIC tasks', len(values)))
    rows = values[-3:]
    metrics = {}
    for metric in METRICS:
        if metric not in columns:
            raise RuntimeError((dest, metric, columns))
        index = columns.index(metric)
        metrics[metric] = statistics.median(float(row[index]) for row in rows)
    data = {'shape': [m, n, fixed.K], 'device': device,
            'pitch': pitch, 'round': round_index,
            'max_relative_error': event['max_relative_error'],
            'metrics': metrics}
    result_path.write_text(json.dumps(data, indent=2) + '\n')
    return data


def one_geometry(case):
    m, n, _ = case['shape']
    device = case['device']
    return [one(m, n, device, pitch, round_index)
            for round_index in range(2)
            for pitch in (PITCHES if round_index == 0 else tuple(reversed(PITCHES)))]


if __name__ == '__main__':
    cases = json.loads((HERE / 'fixed_work_summary.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(one_geometry, cases))
    records = [row for group in groups for row in group]
    (HERE / 'source_profile_records.json').write_text(json.dumps(records, indent=2) + '\n')
    for case in cases:
        m, n, _ = case['shape']
        for metric in METRICS:
            times = {pitch: statistics.median(
                row['metrics'][metric] for row in records
                if row['shape'][:2] == [m, n] and row['pitch'] == pitch)
                for pitch in PITCHES}
            excess = times[49152] - (times[48896] + times[49408]) / 2
            print(m, n, metric, round(excess, 1), flush=True)
