#!/usr/bin/env python3
"""Compare the staged probe's pitch response with paired source controls."""

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PITCHES = (48896, 49152, 49408)
MODES = (0, 1, 2, 3)


def center(x):
    return x[49152] - (x[48896] + x[49408]) / 2


def main():
    cases = [x for x in json.loads((HERE / 'fixed_work_summary.json').read_text())
             if tuple(x['shape'][:2]) in ((3072, 1024), (2048, 2048))]
    rows = json.loads((HERE / 'records.json').read_text())
    controls = json.loads((HERE / 'source_controls.json').read_text())
    assert len(rows) == 96 and len(controls) == 24
    result = []
    for case in cases:
        m, n, _ = case['shape']
        output = {'shape': case['shape'], 'tile': case['tile'], 'modes': {}}
        for mode in MODES:
            subset = [x for x in rows if (x['m'], x['n'], x['mode']) == (m, n, mode)]
            assert len(subset) == 12
            medians = {p: statistics.median(x['device_elapsed_us'] for x in subset
                                           if x['pitch'] == p) for p in PITCHES}
            by_round = {r: center({p: next(x['device_elapsed_us'] for x in subset
                                            if x['pitch'] == p and x['round'] == r)
                                   for p in PITCHES}) for r in range(4)}
            output['modes'][mode] = {
                'pitch_median_us': medians,
                'center_excess_us': center(medians),
                'round_center_excess_us': by_round,
                'round_median_excess_us': statistics.median(by_round.values()),
                'round_range_excess_us': [min(by_round.values()), max(by_round.values())],
            }
        subset = [x for x in controls if (x['m'], x['n']) == (m, n)]
        assert len(subset) == 12 and all(x['max_relative_error'] <= .02 for x in subset)
        medians = {p: statistics.median(x['median_us'] for x in subset
                                       if x['a_pitch'] == p) for p in PITCHES}
        by_round = {r: center({p: next(x['median_us'] for x in subset
                                        if x['a_pitch'] == p and x['round'] == r)
                               for p in PITCHES}) for r in range(4)}
        output['source'] = {
            'pitch_median_us': medians,
            'center_excess_us': center(medians),
            'round_center_excess_us': by_round,
            'round_median_excess_us': statistics.median(by_round.values()),
            'round_range_excess_us': [min(by_round.values()), max(by_round.values())],
        }
        output['incremental_excess_us'] = {
            mode: output['modes'][mode]['center_excess_us'] -
                  output['modes'][mode - 1]['center_excess_us']
            for mode in (1, 2, 3)}
        result.append(output)
        print((m, n), 'source', round(output['source']['center_excess_us'], 1),
              'probe', {mode: round(output['modes'][mode]['center_excess_us'], 1)
                        for mode in MODES},
              'increments', {mode: round(x, 1)
                             for mode, x in output['incremental_excess_us'].items()})
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
