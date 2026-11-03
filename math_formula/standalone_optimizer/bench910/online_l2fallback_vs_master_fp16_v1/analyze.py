#!/usr/bin/env python3
"""Validate and merge the paired 910B3 L2-fallback experiment."""

import csv
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).parent
TILE_FIELDS = (
    'singleCoreM', 'singleCoreN', 'singleCoreK', 'baseM', 'baseN',
    'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb', 'depthA1',
    'depthB1', 'tiling_key', 'block_dim', 'workspace_bytes',
    'nd2nzA', 'nd2nzB',
)


def main():
    expected = json.loads((HERE / 'shapes.json').read_text())
    device_rows = []
    for device in range(4, 8):
        device_rows.extend((device, row) for row in
                           json.loads((HERE / f'device{device}.json').read_text()))
    if len(device_rows) != len(expected) or sorted(
            row['shape_index'] for _, row in device_rows) != list(range(len(expected))):
        raise ValueError('expected exactly one row per shape')
    rows = []
    for device, row in sorted(device_rows, key=lambda item: item[1]['shape_index']):
        index = row['shape_index']
        if row.get('error') or row['shape'] != expected[index]:
            raise ValueError(f'case {index} failed or used the wrong shape')
        master, online = (row['arms'][arm] for arm in ('default', 'online'))
        if max(master['max_relative_error'], online['max_relative_error']) > 0.02:
            raise ValueError(f'case {index} failed output audit')
        same_tile = (master['mode'] == online['mode'] and all(
            master['tiling'].get(field) == online['tiling'].get(field)
            for field in TILE_FIELDS))
        if not same_tile:
            raise ValueError(f'case {index} changed visible mode or tile')
        rounds = 2 if index == 6 else 4
        if len(master['times_us']) != rounds or len(online['times_us']) != rounds:
            raise ValueError(f'case {index} has incomplete timing rounds')
        rows.append({
            'index': index,
            'shape': 'x'.join(map(str, row['shape'])),
            'device': device,
            'mode': master['mode'],
            'master_us': master['median_us'],
            'patched_us': online['median_us'],
            'speedup_master_over_patched': row['speedup_default_over_online'],
            'master_rounds_us': master['times_us'],
            'patched_rounds_us': online['times_us'],
            'max_relative_error': max(master['max_relative_error'],
                                      online['max_relative_error']),
        })
    with (HERE / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'cases': len(rows),
        'max_relative_error': max(row['max_relative_error'] for row in rows),
        'large_m_geometric_mean_speedup': math.exp(statistics.mean(
            math.log(row['speedup_master_over_patched']) for row in rows[:5])),
        'rows': rows,
    }
    (HERE / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
