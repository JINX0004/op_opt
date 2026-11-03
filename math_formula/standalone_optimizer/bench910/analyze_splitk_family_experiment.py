#!/usr/bin/env python3
"""Summarize paired 910B3 branch timings against frozen MIP decisions."""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


def median(values):
    return statistics.median(values)


def percentile(sorted_values, p):
    pos = (len(sorted_values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    parser.add_argument('predictions', type=Path)
    args = parser.parse_args()
    with args.predictions.open() as stream:
        model = {(int(r['m']), int(r['n']), int(r['k'])): r
                 for r in csv.DictReader(stream)}
    compact_path = args.experiment / 'compact_cases.jsonl'
    compact = {}
    if compact_path.exists():
        for line in compact_path.read_text().splitlines():
            row = json.loads(line)
            key = (tuple(row['shape']), row['arm'])
            if (row['profile']['kernel_count'] != 8 or
                    row['case']['split_core_mode'] !=
                    row['tiling']['tilingEnableSplitCore'] or
                    not math.isclose(row['case']['kernel_median_us'],
                                     median(row['profile']['durations_us']),
                                     rel_tol=0, abs_tol=1e-6)):
                raise RuntimeError(f'invalid compact profile: {key}')
            compact.setdefault(key, []).append(row['case'])
    records = []
    for shape in json.loads((args.experiment / 'eligible_shapes.json').read_text()):
        shape = tuple(shape)
        times = {}
        default_audit = json.loads((args.experiment / 'audit' /
            'x'.join(map(str, shape)) / 'default/result.json').read_text())
        default_mode = default_audit['tiling']['tilingEnableSplitCore']
        for arm in ('default', 'single', 'deterministic'):
            if compact:
                cases = compact.get((shape, arm), [])
            else:
                cases = sorted((args.experiment / 'timing').glob(
                    f'r*/{"x".join(map(str, shape))}/{arm}/case.json'))
            if cases:
                values = []
                for entry in cases:
                    if isinstance(entry, Path):
                        if (entry.parent / 'interference.flag').exists():
                            raise RuntimeError(f'device interference flagged: {entry.parent}')
                        try:
                            case = json.loads(entry.read_text())
                        except json.JSONDecodeError:
                            # run_case.sh opens the output path before writing it.
                            continue
                    else:
                        case = entry
                    audit = json.loads((args.experiment / 'audit' /
                        'x'.join(map(str, shape)) / arm / 'result.json').read_text())
                    if (case['shape'] != list(shape) or
                            case['device'] != audit['event']['device'] or
                            case['split_core_mode'] != audit['tiling']['tilingEnableSplitCore'] or
                            case['max_relative_error'] > 0.02 or
                            case['kernel_count'] != 8 or
                            'ops_nn_source_build/matmul/mat_mul_v3/' not in case['tiling_source']):
                        raise RuntimeError(f'invalid device record: {shape} {arm}')
                    values.append(case['kernel_median_us'])
                if values:
                    times[arm] = median(values)
        # The audited default/matching forced pair has the same kernel key,
        # tiling, workspace, and block count. Treat it as the same arm even
        # if an early pilot happened to time that duplicate separately.
        if default_mode in (2, 5, 6):
            times['single'] = times.get('default')
        if default_mode == 3:
            times['deterministic'] = times.get('default')
        if len(times) != 3 or any(value is None for value in times.values()):
            continue
        selected = model[shape]['selected']
        if selected not in ('single', 'deterministic'):
            continue
        best = min(times['single'], times['deterministic'])
        records.append({
            'shape': list(shape), 'selected': selected,
            'default_mode': default_mode,
            'default_us': times['default'],
            'single_us': times['single'],
            'deterministic_us': times['deterministic'],
            'mip_chosen_us': times[selected],
            'oracle_us': best,
            'mip_vs_default_speedup': times['default'] / times[selected],
            'mip_regret_pct': 100 * (times[selected] / best - 1),
            'mip_correct': times[selected] == best,
            'default_regret_pct': 100 * (times['default'] / best - 1),
            'default_correct': times['default'] == best,
        })
    if not records:
        print('No complete paired timing records yet')
        return
    ratios = [r['mip_vs_default_speedup'] for r in records]
    logs = [math.log(x) for x in ratios]
    gm = math.exp(statistics.mean(logs))
    # K variants at the same M,N are correlated. Resample spatial shapes,
    # keeping their K variants together, rather than treating eight points
    # as independent workloads.
    clusters = {}
    for row in records:
        clusters.setdefault(tuple(row['shape'][:2]), []).append(
            math.log(row['mip_vs_default_speedup']))
    grouped = list(clusters.values())
    rng = random.Random(20261030)
    bootstrap = sorted(math.exp(statistics.mean(
        value for group in rng.choices(grouped, k=len(grouped))
        for value in group)) for _ in range(10000))
    by_regime = {}
    for label, subset in {
            'below_l2': [r for r in records if r['shape'][1] == 768],
            'above_l2': [r for r in records if r['shape'][1] != 768],
    }.items():
        if subset:
            by_regime[label] = {
                'n_shapes': len(subset),
                'geomean_speedup': math.exp(statistics.mean(
                    math.log(r['mip_vs_default_speedup']) for r in subset)),
                'family_accuracy': sum(r['mip_correct'] for r in subset) / len(subset),
                'default_family_accuracy': sum(r['default_correct'] for r in subset) / len(subset),
            }
    summary = {
        'n_shapes': len(records),
        'mip_family_accuracy': sum(r['mip_correct'] for r in records) / len(records),
        'default_family_accuracy': sum(r['default_correct'] for r in records) / len(records),
        'mip_vs_default_geomean_speedup': gm,
        'mip_vs_default_cluster_bootstrap_95pct': [percentile(bootstrap, 0.025),
                                                  percentile(bootstrap, 0.975)],
        'median_mip_regret_pct': median([r['mip_regret_pct'] for r in records]),
        'median_default_regret_pct': median([r['default_regret_pct'] for r in records]),
        'by_regime': by_regime,
        'records': records,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
