#!/usr/bin/env python3
"""Summarize the A-only concurrency sweep and source-shaped A/B probe."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PITCHES = (32512, 32768, 33024)


def grouped_timings(name, keys):
    rows = json.loads((HERE / name).read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row['device_elapsed_us'])
    return {key: statistics.median(values) for key, values in groups.items()}


def pitch_table(groups, prefix_keys):
    out = []
    for prefix in sorted({key[:-1] for key in groups}):
        times = [groups[prefix + (pitch,)] for pitch in PITCHES]
        out.append({**dict(zip(prefix_keys, prefix)),
                    'pitch_32512_us': times[0],
                    'pitch_32768_us': times[1],
                    'pitch_33024_us': times[2],
                    'center_excess_us': times[1] - (times[0] + times[2]) / 2})
    return out


def main():
    a_rows = json.loads((HERE / 'records.json').read_text()) + json.loads(
        (HERE / 'core_sweep_records.json').read_text())
    a_groups = defaultdict(list)
    for row in a_rows:
        a_groups[row['cores'], row['route'], row['pitch']].append(
            row['device_elapsed_us'])
    a = pitch_table({key: statistics.median(values)
                     for key, values in a_groups.items()}, ('cores', 'route'))
    ab = pitch_table(grouped_timings('ab_records.json', ('n', 'cores', 'route', 'pitch')),
                     ('n', 'cores', 'route'))
    source = json.loads((HERE / 'source_geometry_metrics.json').read_text())
    source_check = [
        {'shape': row['shape'], 'core_m': row['tile']['singleCoreM'],
         'core_n': row['tile']['singleCoreN'],
         'measured_kib': row['gm_to_l1_kib'],
         'predicted_kib': row['one_a_one_b_per_panel_kib'],
         'difference_kib': row['gm_to_l1_kib'] - row['one_a_one_b_per_panel_kib']}
        for row in source]
    metrics = json.loads((HERE / 'ab_profile_metrics.json').read_text())
    profile = []
    for n in (768, 1024):
        for pitch in (32512, 32768):
            x = {row['metric']: row['metrics'] for row in metrics
                 if row['n'] == n and row['pitch'] == pitch}
            cache = x['L2Cache']
            hits = sum(cache.get(f'aic_r{i}_read_cache_hit', 0) for i in (0, 1))
            misses = sum(cache.get(f'aic_r{i}_read_cache_miss_allocate', 0)
                         for i in (0, 1))
            expected = 4 * 32256 * (2560 + 20 * n) / 1024
            profile.append({'n': n, 'pitch': pitch,
                            'gm_to_l1_kib': x['MemoryAccess']['aic_GM_to_L1_datas'],
                            'expected_kib': expected,
                            'l2_read_hit_fraction': hits / (hits + misses),
                            'mte2_busy_fraction': x['PipeUtilization']['aic_mte2_ratio']})
    result = {'a_only': a, 'a_plus_b': ab,
              'ab_profile': profile, 'source_geometry': source_check}
    (HERE / 'extended_summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
