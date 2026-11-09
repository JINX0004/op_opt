#!/usr/bin/env python3
"""Summarize the 2x2 in-context A/B cache-hint factorial."""

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
PITCHES = (48896, 49152, 49408)
POLICIES = (0, 1, 2, 3)
METRICS = ('device_median_us', 'aic_total_time', 'aic_mte2_time',
           'aic_mte1_time', 'aic_mac_time', 'aic_fixpipe_time',
           'aic_scalar_time')


def center(values):
    return values[49152] - (values[48896] + values[49408]) / 2


def value(row, metric):
    return row[metric] if metric == 'device_median_us' else row['metrics'][metric]


def main():
    cases = json.loads((HERE / 'fixed_work_summary.json').read_text())
    rows = json.loads((HERE / 'records.json').read_text())
    controls = json.loads((HERE / 'control_records.json').read_text())
    result = []
    for case in cases:
        shape = case['shape']
        subset = [row for row in rows if row['shape'] == shape]
        rounds = sorted(set(row['round'] for row in subset))
        assert len(subset) == len(rounds) * len(PITCHES) * len(POLICIES)
        assert all(row['max_relative_error'] <= .02 for row in subset)
        by_policy = {}
        for policy in POLICIES:
            values = {}
            for metric in METRICS:
                medians = {
                    pitch: statistics.median(
                        value(row, metric)
                        for row in subset
                        if row['policy'] == policy and row['pitch'] == pitch)
                    for pitch in PITCHES
                }
                round_excess = {
                    round_index: center({pitch: value(next(
                        row for row in subset if row['policy'] == policy and
                        row['pitch'] == pitch and row['round'] == round_index), metric)
                        for pitch in PITCHES})
                    for round_index in rounds
                }
                values[metric] = {'pitch_median_us': medians,
                                  'center_excess_us': center(medians),
                                  'paired_round_excess_us': round_excess,
                                  'paired_round_median_us': statistics.median(round_excess.values()),
                                  'paired_round_range_us': [min(round_excess.values()),
                                                            max(round_excess.values())]}
            by_policy[policy] = values
        factorial = {}
        for metric in METRICS:
            t = {policy: by_policy[policy][metric]['center_excess_us']
                 for policy in POLICIES}
            factorial[metric] = {
                'a_disable_effect_us': t[1] - t[0],
                'b_disable_effect_us': t[2] - t[0],
                'interaction_us': t[3] - t[1] - t[2] + t[0],
            }
            effects = {'a_disable_effect_us': [], 'b_disable_effect_us': [],
                       'interaction_us': []}
            for round_index in rounds:
                e = {policy: by_policy[policy][metric]['paired_round_excess_us'][round_index]
                     for policy in POLICIES}
                effects['a_disable_effect_us'].append(e[1] - e[0])
                effects['b_disable_effect_us'].append(e[2] - e[0])
                effects['interaction_us'].append(e[3] - e[1] - e[2] + e[0])
            factorial[metric]['paired_round_effects_us'] = effects
            factorial[metric]['paired_round_median_us'] = {
                name: statistics.median(samples) for name, samples in effects.items()}
            factorial[metric]['paired_round_range_us'] = {
                name: [min(samples), max(samples)] for name, samples in effects.items()}
        prior_control = {
            pitch: statistics.median(
                row['device_median_us'] for row in controls
                if row['shape'] == shape and row['pitch'] == pitch)
            for pitch in PITCHES
        }
        assert all(sum(1 for row in controls if row['shape'] == shape and
                       row['pitch'] == pitch) == len(rounds)
                   for pitch in PITCHES)
        control_round_excess = {
            round_index: center({pitch: next(
                row['device_median_us'] for row in controls if row['shape'] == shape
                and row['pitch'] == pitch and row['round'] == round_index)
                for pitch in PITCHES})
            for round_index in rounds
        }
        patched_round_excess = by_policy[0]['device_median_us']['paired_round_excess_us']
        result.append({'shape': shape, 'rounds': len(rounds),
                       'prior_normal_normal_excess_us': case['center_excess_us'],
                       'concurrent_prior_control_us': prior_control,
                       'concurrent_prior_control_excess_us': center(prior_control),
                       'concurrent_prior_round_excess_us': control_round_excess,
                       'policy0_minus_prior_round_excess_us': {
                           round_index: patched_round_excess[round_index] -
                           control_round_excess[round_index]
                           for round_index in rounds},
                       'policy': by_policy, 'factorial': factorial})
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in result:
        t = {int(p): v['device_median_us']['center_excess_us']
             for p, v in row['policy'].items()}
        f = row['factorial']['device_median_us']
        print(row['shape'][:2], 'prior', round(row['prior_normal_normal_excess_us']),
              'concurrent prior', round(row['concurrent_prior_control_excess_us']),
              'policies', {p: round(v) for p, v in t.items()},
              'A', round(f['a_disable_effect_us']),
              'B', round(f['b_disable_effect_us']),
              'interaction', round(f['interaction_us']))


if __name__ == '__main__':
    main()
