#!/usr/bin/env python3
"""Merge the expanded 64-shape FP16 master/online comparisons."""

import csv
import json
import math
import random
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
    rows = []
    for device in range(4, 8):
        path = HERE / f'device{device}.json'
        rows.extend((device, row) for row in json.loads(path.read_text()))
    if len(rows) != 64 or sorted(row['shape_index'] for _, row in rows) != list(range(64)):
        raise ValueError('expected exactly one result for each of the 64 shapes')
    output = []
    max_relative_error = 0.0
    for device, row in sorted(rows, key=lambda item: item[1]['shape_index']):
        if row.get('error'):
            raise ValueError(f"case {row['shape_index']}: {row['error']}")
        master, online = (row['arms'][key] for key in ('default', 'online'))
        if any(arm['max_relative_error'] > 0.02 for arm in (master, online)):
            raise ValueError(f"case {row['shape_index']}: correctness failure")
        max_relative_error = max(max_relative_error,
                                 master['max_relative_error'],
                                 online['max_relative_error'])
        same_mode = master['mode'] == online['mode']
        same_tile = all(master['tiling'].get(key) == online['tiling'].get(key)
                        for key in TILE_FIELDS)
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
    repeats = {}
    for index, device in ((44, 5), (46, 4), (47, 6)):
        path = HERE / f'repeat_{index}_device{device}.json'
        saved = json.loads(path.read_text())
        if len(saved) != 1 or saved[0].get('error') or saved[0]['shape_index'] != index:
            raise ValueError(f'invalid repeat {index}')
        a, b = (saved[0]['arms'][key] for key in ('default', 'online'))
        if len(a['times_us']) != 4 or len(b['times_us']) != 4:
            raise ValueError(f'repeat {index} must have four rounds')
        repeats[index] = {
            'master_rounds_us': a['times_us'],
            'online_rounds_us': b['times_us'],
            'master_median_us': a['median_us'],
            'online_median_us': b['median_us'],
            'speedup': saved[0]['speedup_default_over_online'],
        }
    control = json.loads((HERE / 'control_44_selector_disabled_device7.json').read_text())
    if len(control) != 1 or control[0].get('error') or control[0]['shape_index'] != 44:
        raise ValueError('invalid selector-disabled control')
    adjusted = [math.log(repeats[r['shape_index']]['speedup'])
                if r['shape_index'] in repeats else
                math.log(r['speedup_master_over_online']) for r in output]
    rng = random.Random(20261102)
    adjusted_bootstrap = sorted(math.exp(sum(adjusted[rng.randrange(len(output))]
                                             for _ in output) / len(output))
                                for _ in range(20000))
    result = {
        'cases': len(output),
        'max_relative_error': max_relative_error,
        'changed_family': len(changed),
        'changed_family_gt_5_percent_wins': sum(
            r['speedup_master_over_online'] >= 1.05 for r in changed),
        'changed_family_gt_5_percent_regressions': sum(
            r['speedup_master_over_online'] <= 1 / 1.05 for r in changed),
        'changed_tile_same_family': sum(
            r['master_mode'] == r['online_mode'] and not r['same_tile'] for r in output),
        'same_family_identical_tiles': sum(
            r['master_mode'] == r['online_mode'] and r['same_tile'] for r in output),
        'geometric_mean_speedup': math.exp(statistics.mean(log_speedups)),
        'shape_bootstrap_95_speedup': [bootstrap[499], bootstrap[19499]],
        'sum_master_us': sum(r['master_us'] for r in output),
        'sum_online_us': sum(r['online_us'] for r in output),
        'four_round_repeats': repeats,
        'selector_disabled_control_44': {
            'master_rounds_us': control[0]['arms']['default']['times_us'],
            'online_rounds_us': control[0]['arms']['online']['times_us'],
            'master_median_us': control[0]['arms']['default']['median_us'],
            'online_median_us': control[0]['arms']['online']['median_us'],
        },
        'repeat_substituted_geometric_mean_speedup':
            math.exp(statistics.mean(adjusted)),
        'repeat_substituted_shape_bootstrap_95_speedup':
            [adjusted_bootstrap[499], adjusted_bootstrap[19499]],
        'rows': output,
    }
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
