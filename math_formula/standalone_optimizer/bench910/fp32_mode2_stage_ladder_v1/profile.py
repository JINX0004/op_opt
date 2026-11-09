#!/usr/bin/env python3
"""Confirm each ladder stage appears in AIC pipe-utilization counters."""

import glob
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path

from run import BINARY, HERE, fixed, PITCHES

METRICS = ('aic_total_time', 'aic_mte1_time', 'aic_mte2_time',
           'aic_mac_time', 'aic_fixpipe_time', 'aic_scalar_time')


def one(case, mode, pitch):
    m, n, k = case['shape']
    tile, device = case['tile'], case['device']
    directory = HERE / 'profiles' / f'{m}x{n}' / f'mode{mode}'
    if pitch != PITCHES[0]:
        directory = directory / f'pitch{pitch}'
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    fixed.require_idle(device)
    cmd = [str(BINARY), str(pitch), str(k), str(m), str(n),
           str(tile['singleCoreM']), str(tile['singleCoreN']),
           str(tile['stepM'] * tile['baseM']), str(tile['singleCoreK']),
           str(mode), str(tile['block_dim']), str(device), '3']
    prof = ['msprof', '--output=' + str(directory / 'msprof'),
            '--task-time=on', '--ai-core=on',
            '--aic-metrics=PipeUtilization', 'env'] + cmd
    with (directory / 'stdout.log').open('w') as stdout, \
         (directory / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(prof, cwd=HERE, env=fixed.source_cann_env(),
                              stdout=stdout, stderr=stderr, timeout=1200)
    fixed.require_idle(device)
    if proc.returncode:
        raise RuntimeError((directory, proc.returncode,
                            (directory / 'stderr.log').read_text()[-2000:]))
    events = [json.loads(line) for line in (directory / 'stdout.log').read_text().splitlines()
              if line.startswith('{"pitch":')]
    if len(events) != 1 or events[0]['pitch'] != pitch or \
            (mode == 3 and events[0]['scratch_sample'] <= 0):
        raise RuntimeError(('bad probe event', directory, events))
    databases = glob.glob(str(directory / 'msprof/**/ai_core_op_summary.db'),
                          recursive=True)
    if len(databases) != 1:
        raise RuntimeError(('profiler database missing', directory, databases))
    with sqlite3.connect(databases[0]) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        values = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    if len(values) < 3:
        raise RuntimeError(('AIC tasks missing', directory, len(values)))
    last = values[-3:]
    metrics = {key: statistics.median(float(row[columns.index(key)]) for row in last)
               for key in METRICS}
    result = {'shape': case['shape'], 'mode': mode, 'device': device,
              'pitch': pitch, 'metrics': metrics, 'runner': events[0]}
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    cases = [x for x in json.loads((HERE / 'fixed_work_summary.json').read_text())
             if tuple(x['shape'][:2]) in ((3072, 1024), (2048, 2048))]
    rows = [one(case, mode, pitch) for case in cases for mode in range(4)
            for pitch in PITCHES]
    (HERE / 'profile_pitch_summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print(row['shape'][:2], row['mode'], row['pitch'], row['metrics'], flush=True)
