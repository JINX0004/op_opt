#!/usr/bin/env python3
"""Merge the fresh 22-shape FP16 master/revised-selector comparisons."""

import csv
import json
import math
import random
import statistics
from pathlib import Path


HERE = Path(__file__).parent
TILE_FIELDS = (
    'singleCoreM', 'singleCoreN', 'singleCoreK', 'baseM', 'baseN',
    'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb', 'tiling_key',
    'block_dim', 'workspace_bytes',
)


def main():
    rows = []
    for device in range(4, 8):
        path = HERE / f'device{device}.json'
        rows.extend((device, row) for row in json.loads(path.read_text()))
    if len(rows) != 22 or sorted(row['shape_index'] for _, row in rows) != list(range(22)):
        raise ValueError('expected exactly one result for each of the 22 shapes')
    output = []
    for device, row in sorted(rows, key=lambda item: item[1]['shape_index']):
        if row.get('error'):
            raise ValueError(f"case {row['shape_index']}: {row['error']}")
        master, online = (row['arms'][key] for key in ('default', 'online'))
        if any(arm['max_relative_error'] > 0.02 for arm in (master, online)):
            raise ValueError(f"case {row['shape_index']}: correctness failure")
        same_mode = master['mode'] == online['mode']
        same_tile = all(master['tiling'].get(key) == online['tiling'].get(key)
                        for key in TILE_FIELDS)
        if same_mode and not same_tile:
            raise ValueError(f"case {row['shape_index']}: equal mode changed tile")
        output.append({
            'shape_index': row['shape_index'],
            'shape': 'x'.join(map(str, row['shape'])),
            'device': device,
            'master_mode': master['mode'],
            'online_mode': online['mode'],
            'same_tile': same_tile,
            'master_us': master['median_us'],
            'online_us': online['median_us'],
            'speedup_master_over_online': row['speedup_default_over_online'],
            'master_rounds_us': master['times_us'],
            'online_rounds_us': online['times_us'],
        })
    with (HERE / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'shape_index', 'shape', 'device', 'master_mode', 'online_mode',
            'same_tile', 'master_us', 'online_us', 'speedup_master_over_online',
            'master_rounds_us', 'online_rounds_us'))
        writer.writeheader()
        writer.writerows(output)
    changed = [r for r in output if r['master_mode'] != r['online_mode']]
    log_speedups = [math.log(r['speedup_master_over_online']) for r in output]
    rng = random.Random(20261102)
    bootstrap = sorted(math.exp(sum(log_speedups[rng.randrange(len(output))]
                                    for _ in output) / len(output))
                       for _ in range(20000))
    result = {
        'cases': len(output),
        'changed_family': len(changed),
        'changed_family_wins': sum(r['speedup_master_over_online'] > 1 for r in changed),
        'same_family_identical_tiles': sum(
            r['master_mode'] == r['online_mode'] and r['same_tile'] for r in output),
        'geometric_mean_speedup': math.exp(statistics.mean(log_speedups)),
        'shape_bootstrap_95_speedup': [bootstrap[499], bootstrap[19499]],
        'sum_master_us': sum(r['master_us'] for r in output),
        'sum_online_us': sum(r['online_us'] for r in output),
        'rows': output,
    }
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
