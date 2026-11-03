#!/usr/bin/env python3
"""Analyze device-4 output-only and source-in-context Fixpipe timing."""

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).parent


def rate(rows):
    # The wide source tiles have one output tile per AIC, so these bytes are
    # the exact requested FP32 output bytes on a full, critical AIC.
    xs = [4 * r['source_core_m'] * r['source_core_n'] * r['k_slices']
          for r in rows]
    ys = [r['metrics_us']['aic_fixpipe_time(us)'] for r in rows]
    return sum(x * x for x in xs) / sum(x * y for x, y in zip(xs, ys))


def main():
    sweep = [json.loads(line) for line in
             (HERE / 'output_sweep_device4.jsonl').read_text().splitlines()]
    source = [json.loads(line) for line in
              (HERE / 'source_pipeline_device4.jsonl').read_text().splitlines()]
    holdout = json.loads((HERE / 'holdout_2048x4096x32768_device4.json').read_text())
    assert len(sweep) == 68 and len(source) == 16
    wide = [r for r in source if r['shape'][1] > 768]
    fitted = rate(wide)
    shapes = sorted(set(tuple(r['shape']) for r in wide))
    rng = random.Random(20261030)
    bootstrap = sorted(rate([r for shape in (rng.choice(shapes)
                            for _ in shapes) for r in wide
                            if tuple(r['shape']) == shape])
                       for _ in range(10000))

    by_condition = defaultdict(list)
    for r in sweep:
        key = (r['n_tiles'], r['k_packets'], r['serial'],
               r['distinct'], r['atomic'])
        by_condition[key].append(r['median_us'])
    assert all(len(v) == 2 for v in by_condition.values())
    def median(n, k, serial, distinct, atomic):
        return statistics.median(by_condition[(n, k, serial, distinct, atomic)])

    isolated = []
    for n in (8, 12):
        for serial in (0, 1):
            plain = (median(n, 64, serial, 0, 0) -
                     median(n, 16, serial, 0, 0)) / 48
            atomic = (median(n, 64, serial, 0, 1) -
                      median(n, 16, serial, 0, 1)) / 48
            isolated.append({'n_tiles': n, 'serial_barrier': serial,
                             'plain_us_per_k': plain,
                             'atomic_us_per_k': atomic,
                             'atomic_increment_us_per_k': atomic - plain})

    predicted_holdout = (4 * holdout['source_tiling']['singleCoreM'] *
                         holdout['source_tiling']['singleCoreN'] *
                         holdout['k_slices'] / fitted)
    result = {
        'device': 4,
        'isolated_streams': isolated,
        'distinct_vs_repeated_atomic_k4_us': [
            {'n_tiles': n,
             'distinct': median(n, 4, 0, 1, 1),
             'repeated': median(n, 4, 0, 0, 1)} for n in (8, 12)],
        'source_wide_fixpipe_total_rate_bytes_per_us': fitted,
        'source_wide_shape_bootstrap_95pct': [bootstrap[250], bootstrap[9750]],
        'source_wide_metrics': [
            {'shape': r['shape'], 'round': r['round'],
             'fixpipe_us': r['metrics_us']['aic_fixpipe_time(us)'],
             'mac_us': r['metrics_us']['aic_mac_time(us)'],
             'mte2_us': r['metrics_us']['aic_mte2_time(us)'],
             'kernel_us': r['metrics_us']['Task Duration(us)']}
            for r in wide],
        'holdout': {
            'shape': holdout['shape'],
            'fixpipe_measured_us': holdout['metrics_us']['aic_fixpipe_time(us)'],
            'fixpipe_predicted_us': predicted_holdout,
            'relative_error_pct': 100 * (predicted_holdout /
                holdout['metrics_us']['aic_fixpipe_time(us)'] - 1),
            'kernel_measured_us': holdout['kernel_median_us'],
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
