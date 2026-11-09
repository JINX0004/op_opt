#!/usr/bin/env python3
"""Reproduce the current FP32 mode-2 objective for audited one-panel tiles."""

import csv
import json
from math import ceil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE = json.loads((HERE / 'profile_summary.json').read_text())


def slice_us(m, n, k, atomic):
    mb, nb = ceil(m / 128), ceil(n / 128)
    cube = 2 * m * n * k / 3_600_000
    read = (4 * k * (m * nb + n * mb) / 3_710_212
            + 2 * mb * nb * ceil(k / 64) * 0.109620)
    output = (4 * m * n / 91_000 if atomic else
              mb * nb * 0.2432 + 4 * m * n / 34_190)
    return max(cube, read + output)


rows = []
for item in PROFILE:
    if 'error' in item:
        raise RuntimeError(f"failed profile: {item}")
    m, n, k = item['shape']
    t = item['tile']
    cm, cn, ck = (t['singleCoreM'], t['singleCoreN'], t['singleCoreK'])
    if (ceil(m / cm) * ceil(n / cn) != 20 or t['block_dim'] != 20 or
            cm > 256 or ck != 256 or t['baseK'] != 64):
        raise RuntimeError(f"not the one-panel, one-tile-per-AIC geometry: {item}")
    p = ceil(k / ck)
    tail = k - (p - 1) * ck
    modeled = (2.68 + slice_us(cm, cn, ck, False)
               + (p - 2) * slice_us(cm, cn, ck, True)
               + slice_us(cm, cn, tail, True))
    c = item['counters']
    measured = item['median_us']
    rows.append({
        'device': item['device'], 'm': m, 'n': n, 'k': k,
        'core_m': cm, 'core_n': cn, 'k_slices': p,
        'model_us': round(modeled, 3), 'measured_us': round(measured, 3),
        'residual_us': round(measured - modeled, 3),
        'residual_per_slice_us': round((measured - modeled) / p, 3),
        'aicore_us': round(c['aicore_time(us)'], 3),
        'mac_us': round(c['aic_mac_time(us)'], 3),
        'mte1_us': round(c['aic_mte1_time(us)'], 3),
        'mte2_us': round(c['aic_mte2_time(us)'], 3),
        'fixpipe_us': round(c['aic_fixpipe_time(us)'], 3),
        'scalar_us': round(c['aic_scalar_time(us)'], 3),
        'task_us': round(c['Task Duration(us)'], 3),
    })

anchor = next(row for row in rows if (row['m'], row['n'], row['k']) ==
              (2816, 1024, 36864))
if abs(anchor['model_us'] - 3098.38319) > .002:
    raise RuntimeError('model reproduction disagrees with C++ source-tile score')

with (HERE / 'results.csv').open('w', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(row['device'], row['m'], row['n'], row['k'],
          'score', row['model_us'], 'measured', row['measured_us'],
          'residual/slice', row['residual_per_slice_us'])
