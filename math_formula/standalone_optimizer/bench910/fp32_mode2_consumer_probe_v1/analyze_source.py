#!/usr/bin/env python3
"""Summarize stage-busy pitch excess from checked source profiles."""

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
PITCHES = (48896, 49152, 49408)
METRICS = ('aic_total_time', 'aic_mte2_time', 'aic_mte1_time',
           'aic_mac_time', 'aic_fixpipe_time', 'aic_scalar_time')


def main():
    records = json.loads((HERE / 'source_profile_records.json').read_text())
    fixed = json.loads((HERE.parent / 'fp32_mode2_in_context_pitch_v1/summary.json').read_text())
    result = []
    for case in fixed:
        m, n, k = case['shape']
        subset = [row for row in records if row['shape'] == [m, n, k]]
        assert len(subset) == 6
        assert all(row['max_relative_error'] <= .02 for row in subset)
        deltas = {}
        medians = {}
        for metric in METRICS:
            times = {pitch: statistics.median(
                row['metrics'][metric] for row in subset if row['pitch'] == pitch)
                for pitch in PITCHES}
            medians[metric] = times
            deltas[metric] = times[49152] - (times[48896] + times[49408]) / 2
        result.append({'shape': [m, n, k],
                       'full_kernel_center_excess_us': case['center_excess_us'],
                       'stage_center_excess_us': deltas,
                       'stage_median_us': medians})
    (HERE / 'source_profile_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in result:
        print(row['shape'][:2], 'full', round(row['full_kernel_center_excess_us']),
              'AIC', round(row['stage_center_excess_us']['aic_total_time']),
              'MTE2', round(row['stage_center_excess_us']['aic_mte2_time']),
              'MTE1', round(row['stage_center_excess_us']['aic_mte1_time']),
              'MAC', round(row['stage_center_excess_us']['aic_mac_time']),
              'FIX', round(row['stage_center_excess_us']['aic_fixpipe_time']))


if __name__ == '__main__':
    main()
