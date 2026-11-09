#!/usr/bin/env python3
"""Confirm that the consuming probe executes MTE1 work on 910B3."""

import glob
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path

from run import BINARY, HERE, source_env, require_idle


def one(mode):
    dest = HERE / 'profiles' / f'mode{mode}'
    dest.mkdir(parents=True, exist_ok=True)
    databases = glob.glob(str(dest / 'msprof/**/ai_core_op_summary.db'), recursive=True)
    if not databases:
        require_idle(4)
        cmd = [str(BINARY), '48896', '48384', '2816', '1024', '144',
               '1024', '256', '256', '1', str(mode), '20', '4', '3']
        prof = ['msprof', '--output=' + str(dest / 'msprof'),
                '--task-time=on', '--ai-core=on',
                '--aic-metrics=PipeUtilization', 'env'] + cmd
        with (dest / 'stdout.log').open('w') as stdout, \
             (dest / 'stderr.log').open('w') as stderr:
            result = subprocess.run(prof, env=source_env(), cwd=HERE,
                                    stdout=stdout, stderr=stderr, timeout=1200)
        require_idle(4)
        if result.returncode:
            raise RuntimeError((dest / 'stderr.log').read_text()[-2000:])
        databases = glob.glob(str(dest / 'msprof/**/ai_core_op_summary.db'), recursive=True)
    if len(databases) != 1:
        raise RuntimeError(databases)
    with sqlite3.connect(databases[0]) as conn:
        columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
        values = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
    rows = values[-3:]
    if len(rows) != 3:
        raise RuntimeError(('unexpected task count', len(values)))
    metrics = {}
    for index, key in enumerate(columns):
        if key in ('task_id', 'stream_id', 'subtask_id', 'batch_id'):
            continue
        if all(row[index] is not None for row in rows):
            try:
                metrics[key] = statistics.median(float(row[index]) for row in rows)
            except (TypeError, ValueError):
                pass
    events = [json.loads(line) for line in (dest / 'stdout.log').read_text().splitlines()
              if line.startswith('{"pitch":')]
    if len(events) != 1:
        raise RuntimeError(('unexpected probe event count', mode, len(events)))
    return {'mode': mode, 'metrics': metrics, 'runner': events[0]}


if __name__ == '__main__':
    result = [one(mode) for mode in (0, 1, 2, 3)]
    (HERE / 'profile_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in result:
        print(row['mode'], row['metrics'], flush=True)
