#!/usr/bin/env python3
"""Validate and summarize the 100-shape paired 910B3 comparison."""

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
    'nd2nzA', 'nd2nzB', 'tilingEnableFullLoad',
    'tilingEnableFixOpti',
)
# Keep the repeat criterion identical to the device runner. Other tiling flags
# are reported below, but the runner does not use them to trigger a second round.
GEOMETRY_FIELDS = (
    'singleCoreM', 'singleCoreN', 'singleCoreK', 'baseM', 'baseN',
    'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb', 'depthA1',
    'depthB1', 'tiling_key', 'block_dim', 'workspace_bytes',
    'tilingEnableSplitCore', 'nd2nzA', 'nd2nzB',
)
GROUPS = (
    ('previous_general', 0, 64),
    ('previous_mode5_boundary', 64, 72),
    ('new_narrow_long_k', 72, 79),
    ('new_family_transition', 79, 86),
    ('new_mode5_boundary', 86, 93),
    ('new_large_mn', 93, 100),
)


def geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def main():
    shapes = json.loads((HERE / 'dataset.json').read_text())
    if len(shapes) != 100 or len({tuple(shape) for shape in shapes}) != 100:
        raise ValueError('dataset must have exactly 100 unique shapes')
    raw = []
    for device in range(4, 8):
        path = HERE / f'device{device}.json'
        raw.extend((device, row) for row in json.loads(path.read_text()))
    retry = json.loads((HERE / 'retry92b.json').read_text())
    if len(retry) != 1 or retry[0]['shape_index'] != 92 or retry[0].get('error'):
        raise ValueError('case-92 rerun is missing or failed')
    original = next(row for device, row in raw if device == 4 and
                    row['shape_index'] == 92)
    if not original.get('error'):
        raise ValueError('expected to retain the original failed harness record')
    raw = [(device, retry[0] if device == 4 and row['shape_index'] == 92
            else row) for device, row in raw]
    initial_by_index = {row['shape_index']: row for _, row in raw}
    reruns = {}
    for device, index in ((7, 39), (6, 86)):
        rerun = json.loads((HERE / f'rerun_device{device}.json').read_text())
        if len(rerun) != 1 or rerun[0]['shape_index'] != index or rerun[0].get('error'):
            raise ValueError(f'case-{index} eight-round rerun is missing or failed')
        for arm in ('default', 'online'):
            before = initial_by_index[index]['arms'][arm]['tiling']
            after = rerun[0]['arms'][arm]['tiling']
            if any(before.get(field) != after.get(field) for field in TILE_FIELDS):
                raise ValueError(f'case-{index} {arm} tiling changed in rerun')
        reruns[index] = rerun[0]
    raw = [(device, reruns.get(row['shape_index'], row)) for device, row in raw]
    if len(raw) != 100 or sorted(row['shape_index'] for _, row in raw) != list(range(100)):
        raise ValueError('expected exactly one result for each of 100 shapes')
    rows = []
    for device, row in sorted(raw, key=lambda pair: pair[1]['shape_index']):
        index = row['shape_index']
        if row.get('error') or row['shape'] != shapes[index]:
            raise ValueError(f'case {index} failed or used the wrong shape: {row.get("error")}')
        master, online = (row['arms'][key] for key in ('default', 'online'))
        error = max(master['max_relative_error'], online['max_relative_error'])
        if error > 0.02:
            raise ValueError(f'case {index} output error {error}')
        same_mode = master['mode'] == online['mode']
        same_tile = same_mode and all(
            master['tiling'].get(field) == online['tiling'].get(field)
            for field in TILE_FIELDS)
        same_geometry = all(
            master['tiling'].get(field) == online['tiling'].get(field)
            for field in GEOMETRY_FIELDS)
        rounds = len(master['times_us'])
        expected_rounds = (8 if index in reruns else
                           2 if not same_geometry or master['mode'] == 5 else 1)
        if rounds != len(online['times_us']) or rounds != expected_rounds:
            raise ValueError(f'case {index} has unexpected timing rounds')
        ratio = master['median_us'] / online['median_us']
        if not math.isclose(ratio, row['speedup_default_over_online'], rel_tol=1e-12):
            raise ValueError(f'case {index} speedup mismatch')
        rows.append({
            'index': index,
            'measurement_source': ('eight_round_rerun' if index in reruns else
                                   'retry92b' if index == 92 else f'device{device}'),
            'shape': 'x'.join(map(str, row['shape'])),
            'device': device,
            'master_mode': master['mode'],
            'online_mode': online['mode'],
            'same_tile': same_tile,
            'same_geometry': same_geometry,
            'master_us': master['median_us'],
            'online_us': online['median_us'],
            'speedup': ratio,
            'rounds': rounds,
            'master_rounds_us': master['times_us'],
            'online_rounds_us': online['times_us'],
            'max_relative_error': error,
        })
    with (HERE / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    changed = [row for row in rows if row['master_mode'] != row['online_mode']]
    unchanged = [row for row in rows if row['master_mode'] == row['online_mode']]
    changed_tile = [row for row in rows if row['master_mode'] == row['online_mode']
                    and not row['same_tile']]
    for name, category_rows in (('changed', changed), ('unchanged', unchanged)):
        with (HERE / f'family_{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(),
                                    lineterminator='\n')
            writer.writeheader()
            writer.writerows(category_rows)
    categories = {
        name: {
            'cases': len(category_rows),
            'geometric_mean_speedup': geomean(
                [row['speedup'] for row in category_rows]),
            'median_speedup': statistics.median(
                row['speedup'] for row in category_rows),
            'online_faster_cases': sum(row['speedup'] > 1 for row in category_rows),
            'online_slower_cases': sum(row['speedup'] < 1 for row in category_rows),
            'identical_tiling_cases': sum(row['same_tile'] for row in category_rows),
        } for name, category_rows in (('changed', changed), ('unchanged', unchanged))
    }
    values = [row['speedup'] for row in rows]
    initial_values = [initial_by_index[index]['speedup_default_over_online']
                      for index in range(100)]
    logs = [math.log(value) for value in values]
    rng = random.Random(20261103)
    bootstrap = sorted(math.exp(sum(logs[rng.randrange(len(logs))]
                                    for _ in logs) / len(logs))
                       for _ in range(20000))
    repeat_raw = [row for device in (5, 6) for row in json.loads(
        (HERE / f'repeat_controls_device{device}.json').read_text())]
    if sorted(row['shape_index'] for row in repeat_raw) != [4, 11, 32, 33, 39, 86]:
        raise ValueError('expected six targeted identical-tiling controls')
    diagnostic_repeats = []
    for repeat in sorted(repeat_raw, key=lambda row: row['shape_index']):
        index = repeat['shape_index']
        master, online = (repeat['arms'][key] for key in ('default', 'online'))
        if repeat.get('error') or repeat['shape'] != shapes[index] or any(
                master['tiling'].get(field) != online['tiling'].get(field)
                for field in TILE_FIELDS) or any(
                    len(arm['times_us']) != 2 or arm['max_relative_error'] > 0.02
                    for arm in (master, online)):
            raise ValueError(f'diagnostic repeat {index} failed validation')
        diagnostic_repeats.append({
            'index': index,
            'shape': rows[index]['shape'],
            'original_speedup': initial_by_index[index]['speedup_default_over_online'],
            'repeat_speedup': repeat['speedup_default_over_online'],
            'master_rounds_us': master['times_us'],
            'online_rounds_us': online['times_us'],
        })
    result = {
        'cases': len(rows),
        'harness_retry_cases': [92],
        'eight_round_rerun_cases': [39, 86],
        'geometric_mean_speedup_before_reruns': geomean(initial_values),
        'rerun_summary': [{
            'index': index,
            'shape': rows[index]['shape'],
            'initial_speedup': initial_by_index[index]['speedup_default_over_online'],
            'rerun_speedup': rows[index]['speedup'],
            'online_faster_rounds': sum(a > b for a, b in zip(
                rows[index]['master_rounds_us'], rows[index]['online_rounds_us'])),
            'rounds': rows[index]['rounds'],
        } for index in (39, 86)],
        'max_relative_error': max(row['max_relative_error'] for row in rows),
        'changed_family': len(changed),
        'family_selection_categories': categories,
        'changed_tile_same_family': len(changed_tile),
        'same_tile': sum(row['same_tile'] for row in rows),
        'same_geometry': sum(row['same_geometry'] for row in rows),
        'changed_family_gt_5_percent_wins': sum(row['speedup'] >= 1.05
                                                for row in changed),
        'changed_family_gt_5_percent_regressions': sum(row['speedup'] <= 1 / 1.05
                                                       for row in changed),
        'changed_family_both_rounds_win': sum(
            all(a > b for a, b in zip(row['master_rounds_us'],
                                     row['online_rounds_us'])) for row in changed),
        'geometric_mean_speedup': geomean(values),
        'shape_bootstrap_95_speedup': [bootstrap[499], bootstrap[19499]],
        'sum_master_us': sum(row['master_us'] for row in rows),
        'sum_online_us': sum(row['online_us'] for row in rows),
        'groups': {name: {
            'cases': end - start,
            'geometric_mean_speedup': geomean(values[start:end]),
            'changed_family': sum(row['master_mode'] != row['online_mode']
                                  for row in rows[start:end]),
        } for name, start, end in GROUPS},
        'diagnostic_repeats': diagnostic_repeats,
        'rows': rows,
    }
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('rows', 'diagnostic_repeats')}, indent=2))


if __name__ == '__main__':
    main()
