#!/usr/bin/env python3
"""Compare three synchronization schedules with forced MatMulV3 tiles."""

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'fp32_mode2_transfer_holdout_v1'
PITCHES = (48896, 49152, 49408)


def excess(values):
    return values[1] - (values[0] + values[2]) / 2


def main():
    rows = json.loads((HERE / 'probe_records.json').read_text())
    source = json.loads((PRIOR / 'summary.json').read_text())
    baseline = json.loads((PRIOR / 'probe_records.json').read_text())
    out = []
    for src in source:
        m, n = src['m'], src['n']
        ref = [r for r in baseline if (r['m'], r['n']) == (m, n)]
        geom = {'m': m, 'n': n, 'tile': src['tile'],
                'kernel_center_excess_us': src['source_kernel_center_excess_us'],
                'mte2_center_excess_us': src['source_mte2_center_excess_us'],
                'modes': []}
        for mode in (0, 1, 2):
            subset = {(r['pitch'], r['round']): r for r in rows
                      if (r['m'], r['n'], r['sync_mode']) == (m, n, mode)}
            if len(subset) != 6:
                raise RuntimeError((m, n, mode, len(subset)))
            times = [statistics.median(subset[k, round_index]['device_elapsed_us']
                                       for round_index in (0, 1))
                     for k in PITCHES]
            round_excesses = [excess([subset[k, round_index]['device_elapsed_us']
                                     for k in PITCHES])
                              for round_index in (0, 1)]
            result = {'sync_mode': mode,
                      'pitch_us': dict(zip(map(str, PITCHES), times)),
                      'center_excess_us': excess(times),
                      'center_excess_rounds_us': round_excesses,
                      'absolute_error_vs_kernel_excess_us':
                          abs(excess(times) - geom['kernel_center_excess_us'])}
            if mode == 0:
                old = {k: statistics.median(r['device_elapsed_us'] for r in ref
                                             if r['pitch'] == k) for k in PITCHES}
                result['prior_copy_barrier_excess_us'] = excess(list(old.values()))
            geom['modes'].append(result)
        out.append(geom)
    (HERE / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    for geom in out:
        print(geom['m'], geom['n'],
              'kernel', round(geom['kernel_center_excess_us'], 1),
              'MTE2', round(geom['mte2_center_excess_us'], 1),
              'probe modes',
              [round(mode['center_excess_us'], 1) for mode in geom['modes']])


if __name__ == '__main__':
    main()
