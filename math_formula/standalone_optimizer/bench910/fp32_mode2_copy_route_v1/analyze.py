#!/usr/bin/env python3
"""Summarize paired route and pitch results without fitting a model."""

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PITCHES = (32512, 32768, 33024)


def main():
    records = json.loads((HERE / 'records.json').read_text())
    profiles = json.loads((HERE / 'profile_metrics.json').read_text())
    rows = []
    for cores in (20, 1):
        for route in ('inline', 'plain'):
            values = {}
            for pitch in PITCHES:
                samples = [row['device_elapsed_us'] for row in records
                           if (row['cores'], row['route'], row['pitch']) ==
                           (cores, route, pitch)]
                if len(samples) != 4:
                    raise RuntimeError((cores, route, pitch, len(samples)))
                values[pitch] = {'median_us': statistics.median(samples),
                                 'range_us': [min(samples), max(samples)]}
            excess = []
            for device in (4, 6):
                for round_index in (0, 1):
                    def elapsed(pitch):
                        return next(row['device_elapsed_us'] for row in records
                                    if (row['cores'], row['route'], row['pitch'],
                                        row['device'], row['round']) ==
                                    (cores, route, pitch, device, round_index))
                    excess.append(elapsed(32768) -
                                  (elapsed(32512) + elapsed(33024)) / 2)
            rows.append({'cores': cores, 'route': route, 'pitches': values,
                         'center_excess_us': {'median': statistics.median(excess),
                                              'range': [min(excess), max(excess)],
                                              'samples': excess}})
    counters = []
    for route in ('inline', 'plain'):
        for pitch in (32512, 32768):
            by_group = {row['metric']: row['metrics'] for row in profiles
                        if (row['route'], row['pitch']) == (route, pitch)}
            memory = by_group['MemoryAccess']
            cache = by_group['L2Cache']
            pipe = by_group['PipeUtilization']
            hit = sum(cache[f'aic_r{i}_read_cache_hit'] for i in (0, 1))
            miss = sum(cache[f'aic_r{i}_read_cache_miss_allocate'] for i in (0, 1))
            counters.append({'route': route, 'pitch': pitch,
                             'gm_to_l1_kib': memory['aic_GM_to_L1_datas'],
                             'l2_read_hit_fraction': hit / (hit + miss),
                             'mte2_busy_fraction': pipe['aic_mte2_ratio']})
    summary = {'logical_k': 32256, 'm': 2560, 'repeats': 1,
               'timings': rows, 'counters': counters}
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for row in rows:
        print(row['cores'], row['route'],
              [round(row['pitches'][p]['median_us'], 2) for p in PITCHES],
              'center excess', round(row['center_excess_us']['median'], 2),
              'range', [round(x, 2) for x in row['center_excess_us']['range']])
    for row in counters:
        print(row)


if __name__ == '__main__':
    main()
