#!/usr/bin/env python3
"""Extract median MatMulV3 AIC metrics from the controlled msprof runs."""

import glob
import json
import sqlite3
import statistics
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/fp32_base_formula_v2/pipeline_profiles')
OUT = ROOT.parent / 'pipeline_metrics.json'


def collect():
    rows = []
    for case in sorted(ROOT.glob('K*_*')):
        result = case / 'result.json'
        if not result.exists():
            continue
        info = json.loads(result.read_text())
        files = glob.glob(str(case / 'msprof/**/ai_core_op_summary.db'), recursive=True)
        if len(files) != 1:
            raise RuntimeError((case, files))
        connection = sqlite3.connect(files[0])
        columns = [row[1] for row in connection.execute('pragma table_info(ai_core_metrics)')]
        values = connection.execute('select * from ai_core_metrics order by task_id').fetchall()
        connection.close()
        if len(values) < 3:
            raise RuntimeError((case, len(values)))
        recent = values[-3:]
        metrics = {column: statistics.median(float(row[index]) for row in recent)
                   for index, column in enumerate(columns)
                   if column not in ('task_id', 'stream_id', 'subtask_id', 'batch_id')
                   and all(row[index] is not None for row in recent)}
        rows.append({'k': info['k'], 'metric': info['metric'], 'metrics': metrics})
    OUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
    print('wrote', OUT, 'cases', len(rows))


if __name__ == '__main__':
    collect()
