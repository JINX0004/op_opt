#!/usr/bin/env python3
"""Project the 72-case result by substituting patched paired reruns."""

import json
import math
import random
import statistics
from pathlib import Path


HERE = Path(__file__).parent
PREVIOUS = HERE.parent / 'online_expanded_vs_master_fp16_v1'


def geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def main():
    previous = json.loads((PREVIOUS / 'analysis.json').read_text())['rows']
    boundary = [row for device in range(4, 8) for row in json.loads(
        (PREVIOUS / f'mode5_boundary_device{device}.json').read_text())]
    patched = json.loads((HERE / 'analysis.json').read_text())['rows']
    if len(previous) != 64 or len(boundary) != 8 or len(patched) != 7:
        raise ValueError('incomplete experiment inputs')
    replacements = {row['shape']: row['speedup_master_over_patched']
                    for row in patched}
    old = {row['shape']: row['speedup_master_over_online'] for row in previous}
    old.update({'x'.join(map(str, row['shape'])):
                row['speedup_default_over_online'] for row in boundary})
    if len(old) != 72 or not replacements.keys() <= old.keys():
        raise ValueError('shapes are duplicated or missing')
    values = [replacements.get(shape, ratio) for shape, ratio in old.items()]
    rng = random.Random(20261103)
    logs = [math.log(value) for value in values]
    bootstrap = sorted(math.exp(sum(logs[rng.randrange(len(logs))]
                                    for _ in logs) / len(logs))
                       for _ in range(20000))
    output = {
        'method': 'hybrid substitution, not a fresh 72-case patched run',
        'shape_count': len(values),
        'replaced_shapes': sorted(replacements),
        'old_package_geometric_mean_speedup': geomean(list(old.values())),
        'hybrid_geometric_mean_speedup': geomean(values),
        'hybrid_shape_bootstrap_95': [bootstrap[499], bootstrap[19499]],
        'hybrid_gt_5_percent_wins': sum(value >= 1.05 for value in values),
        'hybrid_gt_5_percent_regressions': sum(
            value <= 1 / 1.05 for value in values),
    }
    (HERE / 'overall_projection.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
