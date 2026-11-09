#!/usr/bin/env python3
"""Measure L2 read-hit fractions for selected modes and A row pitches."""

import glob
import json
import sqlite3
import statistics
import subprocess

from run import BINARY, HERE, fixed, PITCHES

KEYS = ('aic_r0_read_cache_hit', 'aic_r0_read_cache_miss_allocate',
        'aic_r1_read_cache_hit', 'aic_r1_read_cache_miss_allocate',
        'aic_write_cache_hit', 'aic_write_cache_miss_allocate')


def one(case, mode, pitch):
    m, n, k = case['shape']
    tile, device = case['tile'], case['device']
    directory = HERE / 'l2_profiles' / f'{m}x{n}' / f'mode{mode}' / f'pitch{pitch}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'result.json'
    if path.exists():
        return json.loads(path.read_text())
    fixed.require_idle(device)
    cmd = [str(BINARY), str(pitch), str(k), str(m), str(n),
           str(tile['singleCoreM']), str(tile['singleCoreN']),
           str(tile['stepM'] * tile['baseM']), str(tile['singleCoreK']),
           str(mode), str(tile['block_dim']), str(device), '3']
    prof = ['msprof', '--output=' + str(directory / 'msprof'),
            '--task-time=on', '--ai-core=on', '--aic-metrics=L2Cache', 'env'] + cmd
    with (directory / 'stdout.log').open('w') as stdout, \
         (directory / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(prof, env=fixed.source_cann_env(), cwd=HERE,
                              stdout=stdout, stderr=stderr, timeout=1200)
    fixed.require_idle(device)
    if proc.returncode:
        raise RuntimeError((directory, proc.returncode,
                            (directory / 'stderr.log').read_text()[-1500:]))
    events = [json.loads(line) for line in (directory / 'stdout.log').read_text().splitlines()
              if line.startswith('{"pitch":')]
    if len(events) != 1 or events[0]['pitch'] != pitch:
        raise RuntimeError(('bad event', directory, events))
    databases = glob.glob(str(directory / 'msprof/**/ai_core_op_summary.db'),
                          recursive=True)
    if len(databases) != 1:
        raise RuntimeError(('database', directory, databases))
    with sqlite3.connect(databases[0]) as conn:
        cols = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        values = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    rows = values[-3:]
    if len(rows) != 3:
        raise RuntimeError(('task count', directory, len(values)))
    metrics = {key: statistics.median(float(row[cols.index(key)]) for row in rows)
               for key in KEYS}
    hit = metrics['aic_r0_read_cache_hit'] + metrics['aic_r1_read_cache_hit']
    miss = (metrics['aic_r0_read_cache_miss_allocate'] +
            metrics['aic_r1_read_cache_miss_allocate'])
    result = {'shape': case['shape'], 'mode': mode, 'pitch': pitch,
              'device': device, 'read_hit_fraction': hit / (hit + miss),
              'metrics': metrics, 'runner': events[0]}
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    cases = [x for x in json.loads((HERE / 'fixed_work_summary.json').read_text())
             if tuple(x['shape'][:2]) in ((3072, 1024), (2048, 2048))]
    rows = [one(case, mode, pitch) for case in cases for mode in (0, 2, 3)
            for pitch in PITCHES]
    (HERE / 'l2_summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print(row['shape'][:2], row['mode'], row['pitch'],
              round(row['read_hit_fraction'], 6), flush=True)
