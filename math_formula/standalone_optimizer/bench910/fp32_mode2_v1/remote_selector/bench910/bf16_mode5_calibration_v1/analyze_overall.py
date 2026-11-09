#!/usr/bin/env python3
"""Combine new paired mode-5 runs with unchanged-family BF16 baseline runs."""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'bf16_online_vs_master_20261103' / 'measurements.json'
MODE5 = {(49152, 384, 1536), (65536, 384, 1536), (81920, 384, 1536)}


def median_by_arm(rows, time_key):
    values = defaultdict(list)
    for row in rows:
        values[row['arm']].append(row[time_key])
    return {arm: statistics.median(times) for arm, times in values.items()}


def main():
    previous = defaultdict(list)
    for row in json.loads(OLD.read_text()):
        previous[tuple(row['shape'])].append(row)

    cases = []
    for shape, rows in previous.items():
        old = median_by_arm(rows, 'kernel_median_us')
        if shape in MODE5:
            path = HERE / 'validation' / f'm{shape[0]}_summary.json'
            fresh = json.loads(path.read_text())
            assert len(fresh) == 4
            assert len({json.dumps(row['tiling'], sort_keys=True) for row in fresh}) == 1
            latest = median_by_arm(fresh, 'median_us')
            source = str(path.relative_to(HERE))
        else:
            latest = old
            source = str(OLD.relative_to(HERE.parent))
        cases.append({'shape': list(shape), 'old_master_us': old['master'],
                      'old_online_us': old['online'],
                      'latest_master_us': latest['master'],
                      'latest_online_us': latest['online'],
                      'old_speedup': old['master'] / old['online'],
                      'latest_speedup': latest['master'] / latest['online'],
                      'latest_source': source})

    def geo(key):
        return math.exp(statistics.mean(math.log(row[key]) for row in cases))

    result = {'shape_count': len(cases),
              'old_geomean_master_over_online': geo('old_speedup'),
              'composite_latest_geomean_master_over_online': geo('latest_speedup'),
              'fresh_mode5_shapes': [list(x) for x in sorted(MODE5)],
              'cases': sorted(cases, key=lambda row: tuple(row['shape']))}
    (HERE / 'overall_status.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
