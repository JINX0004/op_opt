#!/usr/bin/env python3
"""Summarize pipe-counter pitch excess and input-byte invariance."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PITCHES = (48896, 49152, 49408)
METRICS = ('aic_total_time', 'aic_mte2_time', 'aic_mte1_time',
           'aic_mac_time', 'aic_fixpipe_time')


def center(values):
    return values[49152] - (values[48896] + values[49408]) / 2


def main():
    profiles = json.loads((HERE / 'profile_pitch_summary.json').read_text())
    traffic = json.loads((HERE / 'traffic_summary.json').read_text())
    assert len(profiles) == 24 and len(traffic) == 8
    result = []
    for shape in ((3072, 1024), (2048, 2048)):
        for mode in range(4):
            rows = {row['pitch']: row for row in profiles
                    if tuple(row['shape'][:2]) == shape and row['mode'] == mode}
            assert set(rows) == set(PITCHES)
            measured = {metric: center({pitch: rows[pitch]['metrics'][metric]
                                        for pitch in PITCHES})
                        for metric in METRICS}
            t = next(row for row in traffic if tuple(row['shape'][:2]) == shape
                     and row['mode'] == mode)
            result.append({'shape': shape, 'mode': mode,
                           'pipe_center_excess_us': measured,
                           'traffic_at_192kib': t['metrics']})
            print(shape, mode,
                  'AIC total', round(measured['aic_total_time'], 1),
                  'MTE2', round(measured['aic_mte2_time'], 1),
                  'GM->L1 units', t['metrics']['aic_GM_to_L1_datas'])
    (HERE / 'profile_analysis.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
