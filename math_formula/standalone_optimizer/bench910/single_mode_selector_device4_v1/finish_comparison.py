#!/usr/bin/env python3
"""Check structural mode-2/5 channels and the narrow-N default arm."""

import json
from pathlib import Path

from single_mode_experiment import experiment


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
experiment.DEVICE = 4
experiment.OUT = ROOT / 'cases/single_mode_selector_device4_v1_structural'
experiment.OUT.mkdir(parents=True, exist_ok=True)
rows = []
for shape in ((49152, 384, 1536), (384, 49152, 1536)):
    row = experiment.audit_one('default', shape)
    rows.append({'shape': shape, 'returncode': row['returncode'],
                 'mode': row.get('tiling', {}).get('tilingEnableSplitCore'),
                 'max_relative_error': row.get('event', {}).get('max_relative_error')})
    print('structural', rows[-1], flush=True)
(experiment.OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')

experiment.OUT = ROOT / 'cases/single_mode_selector_device4_v1'
shape = (2560, 1024, 36864)
for round_index in range(2):
    row = experiment.time_one('default', shape, round_index, 8)
    print('narrow default', round_index, row['kernel_median_us'], flush=True)
