#!/usr/bin/env python3
"""Analyze the frozen mode-2/6 selector against device-4 source timings."""

import csv
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def quantile(values, q):
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    with (ROOT / 'model_predictions.csv').open(newline='') as stream:
        pred = {tuple(int(r[x]) for x in ('m', 'n', 'k')): r for r in csv.DictReader(stream)}
    eligibility = json.loads((ROOT / 'eligibility.json').read_text())
    cases = {}
    for line in (ROOT / 'compact_cases.jsonl').read_text().splitlines():
        row = json.loads(line)
        case = row['case']
        assert case['device'] == manifest['device']
        assert case['kernel_count'] == row['profile']['kernel_count'] == manifest['repetitions_per_round']
        assert case['max_relative_error'] <= 0.02
        assert math.isclose(case['kernel_median_us'], statistics.median(row['profile']['durations_us']), abs_tol=1e-6)
        assert case['split_core_mode'] == row['tiling']['tilingEnableSplitCore']
        key = (tuple(row['shape']), row['arm'])
        cases.setdefault(key, {})[row['round']] = case['kernel_median_us']
    records = []
    for item in eligibility:
        if item['reasons']:
            continue
        shape = tuple(item['shape'])
        expected = set(range(manifest['rounds']))
        assert set(cases.get((shape, 'mode2'), {})) == expected
        assert set(cases.get((shape, 'mode6'), {})) == expected
        t2 = statistics.median(cases[shape, 'mode2'].values())
        t6 = statistics.median(cases[shape, 'mode6'].values())
        selected = int(pred[shape]['selected_mode'])
        assert selected in (2, 6)
        default_runs = cases.get((shape, 'default'), {})
        if default_runs:
            assert set(default_runs) == expected
        default_us = statistics.median(default_runs.values()) if default_runs else None
        records.append({
            'shape': list(shape), 'default_mode': item['modes']['default'],
            'predicted_mode2_us': float(pred[shape]['mode2_us']),
            'predicted_mode6_us': float(pred[shape]['mode6_us']),
            'selected_mode': selected,
            'measured_mode2_us': t2, 'measured_mode6_us': t6,
            'measured_faster_mode': 2 if t2 < t6 else 6,
            'mode6_vs_mode2_speedup': t2 / t6,
            'default_us': default_us,
            'mode6_vs_default_speedup': default_us / t6 if default_us else None,
        })
    logs = [math.log(row['mode6_vs_mode2_speedup']) for row in records]
    rng = random.Random(manifest['seed'])
    bootstrap = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
                       for _ in range(20000))
    summary = {
        'declared_shapes': len(manifest['shapes']),
        'eligible_shapes': len(records),
        'excluded': [x for x in eligibility if x['reasons']],
        'timed_runs': sum(len(x) for x in cases.values()),
        'mip_accuracy_within_mode2_6': sum(x['selected_mode'] == x['measured_faster_mode'] for x in records) / len(records),
        'mode6_vs_mode2_geomean_speedup': math.exp(statistics.mean(logs)),
        'mode6_vs_mode2_bootstrap_95pct': [quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)],
        'records': records,
    }
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
