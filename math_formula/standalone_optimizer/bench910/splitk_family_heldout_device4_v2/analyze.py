#!/usr/bin/env python3
"""Summarize frozen family choices against source-backed device-4 timings."""

import csv
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def percentile(values, quantile):
    index = (len(values) - 1) * quantile
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    with (ROOT / 'model_predictions.csv').open(newline='') as stream:
        predictions = {(int(r['m']), int(r['n']), int(r['k'])): r
                       for r in csv.DictReader(stream)}
    audits = json.loads((ROOT / 'eligibility.json').read_text())
    audit_map = {tuple(row['shape']): row for row in audits}
    compact = {}
    for line in (ROOT / 'compact_cases.jsonl').read_text().splitlines():
        row = json.loads(line)
        shape = tuple(row['shape'])
        key = (shape, row['arm'])
        case, profile, tiling = row['case'], row['profile'], row['tiling']
        assert case['device'] == manifest['device']
        assert case['kernel_count'] == profile['kernel_count'] == manifest['repetitions_per_round']
        assert case['split_core_mode'] == tiling['tilingEnableSplitCore']
        assert case['max_relative_error'] <= 0.02
        assert math.isclose(case['kernel_median_us'], statistics.median(profile['durations_us']), abs_tol=1e-6)
        compact.setdefault(key, {})[row['round']] = case['kernel_median_us']
    records = []
    for shape_list in manifest['shapes']:
        shape = tuple(shape_list)
        audit = audit_map[shape]
        if audit['reasons']:
            continue
        default_family = 'deterministic' if audit['modes']['default'] == 3 else 'single'
        same = compact.get((shape, default_family), {})
        default = compact.get((shape, 'default'), {})
        opposite_family = 'single' if default_family == 'deterministic' else 'deterministic'
        opposite = compact.get((shape, opposite_family), {})
        expected_rounds = set(range(manifest['rounds']))
        assert set(default) == set(opposite) == expected_rounds, shape
        if same:
            assert set(same) == expected_rounds, shape
        # The source audit establishes equal key, tiling, workspace and block count
        # for default and matching forced family when the latter was not timed.
        default_us = statistics.median(default.values())
        times = {default_family: default_us,
                 opposite_family: statistics.median(opposite.values())}
        selected = predictions[shape]['selected']
        assert selected in times, shape
        best_family = min(times, key=times.get)
        selected_us = times[selected]
        records.append({
            'shape': shape_list, 'default_family': default_family,
            'selected_family': selected, 'best_family': best_family,
            'default_us': default_us, 'single_us': times['single'],
            'deterministic_us': times['deterministic'],
            'selected_us': selected_us,
            'speedup_default_over_model': default_us / selected_us,
            'model_regret_pct': 100 * (selected_us / times[best_family] - 1),
            'default_regret_pct': 100 * (default_us / times[best_family] - 1),
            'model_correct': selected == best_family,
            'default_correct': default_family == best_family,
        })
    assert len(audits) == len(manifest['shapes'])
    ratios = [r['speedup_default_over_model'] for r in records]
    cluster = {}
    for r in records:
        cluster.setdefault(tuple(r['shape'][:2]), []).append(math.log(r['speedup_default_over_model']))
    groups = list(cluster.values())
    rng = random.Random(manifest['seed'])
    bootstrap = sorted(math.exp(statistics.mean(value for group in rng.choices(groups, k=len(groups))
                                                for value in group)) for _ in range(20000))
    summary = {
        'declared_shapes': len(manifest['shapes']),
        'eligible_shapes': len(records),
        'excluded': [row for row in audits if row['reasons']],
        'timed_arm_shape_pairs': len(compact),
        'timed_runs': sum(len(rounds) for rounds in compact.values()),
        'model_family_accuracy': sum(r['model_correct'] for r in records) / len(records),
        'default_family_accuracy': sum(r['default_correct'] for r in records) / len(records),
        'model_vs_default_geomean_speedup': math.exp(statistics.mean(math.log(x) for x in ratios)),
        'model_vs_default_bootstrap_95pct': [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        'median_model_regret_pct': statistics.median(r['model_regret_pct'] for r in records),
        'median_default_regret_pct': statistics.median(r['default_regret_pct'] for r in records),
        'records': records,
    }
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
