#!/usr/bin/env python3
"""Compare fixed-logical-K A/B service with forced mode-2 kernels."""

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PITCHES = (48896, 49152, 49408)


def excess(values):
    return values[1] - (values[0] + values[2]) / 2


def main():
    probes = json.loads((HERE / 'probe_records.json').read_text())
    source = json.loads((HERE / 'source_profiles.json').read_text())
    audits = json.loads((HERE / 'tile_audits.json').read_text())
    traffic = json.loads((HERE / 'traffic_MemoryAccess.json').read_text())
    cache = json.loads((HERE / 'traffic_L2Cache.json').read_text())
    groups = sorted({(row['m'], row['n']) for row in probes})
    results = []
    for m, n in groups:
        p = {(row['pitch'], row['round']): row for row in probes
             if (row['m'], row['n']) == (m, n)}
        s = {row['shape'][2]: row for row in source
             if row['shape'][:2] == [m, n]}
        a = {row['shape'][2]: row for row in audits
             if row['shape'][:2] == [m, n]}
        if (len(p), len(s), len(a)) != (6, 3, 3):
            raise RuntimeError((m, n, len(p), len(s), len(a)))
        fields = ('singleCoreM', 'singleCoreN', 'singleCoreK',
                  'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'block_dim')
        tilings = {tuple(a[k]['tile'][field] for field in fields)
                   for k in PITCHES}
        if len(tilings) != 1:
            raise RuntimeError(('source tiling changed', m, n, tilings))
        probe_us = [statistics.median(p[k, r]['device_elapsed_us']
                                      for r in (0, 1)) for k in PITCHES]
        kernel_us = [s[k]['kernel_us'] for k in PITCHES]
        mte2_us = [s[k]['mte2_us'] for k in PITCHES]
        probe_excess_rounds = [excess([p[k, r]['device_elapsed_us']
                                       for k in PITCHES]) for r in (0, 1)]
        result = {'m': m, 'n': n, 'device': s[PITCHES[0]]['device'],
                  'tile': dict(zip(fields, next(iter(tilings)))),
                  'logical_k': p[PITCHES[0], 0]['logical_k'],
                  'probe_us': dict(zip(map(str, PITCHES), probe_us)),
                  'source_kernel_us': dict(zip(map(str, PITCHES), kernel_us)),
                  'source_mte2_us': dict(zip(map(str, PITCHES), mte2_us)),
                  'probe_center_excess_us': excess(probe_us),
                  'probe_center_excess_rounds_us': probe_excess_rounds,
                  'source_kernel_center_excess_us': excess(kernel_us),
                  'source_mte2_center_excess_us': excess(mte2_us)}
        traffic_rows = [row for row in traffic if (row['m'], row['n']) == (m, n)]
        cache_rows = [row for row in cache if (row['m'], row['n']) == (m, n)]
        result['traffic_checks'] = []
        for pitch in PITCHES[:2] if traffic_rows else ():
            source_bytes = next(row['metrics']['aic_GM_to_L1_datas']
                                for row in traffic_rows
                                if row['kind'] == 'source' and row['pitch'] == pitch)
            probe_bytes = next(row['metrics']['aic_GM_to_L1_datas']
                               for row in traffic_rows
                               if row['kind'] == 'probe' and row['pitch'] == pitch)
            hit_fractions = {}
            for kind in ('source', 'probe'):
                c = next(row['metrics'] for row in cache_rows
                         if row['kind'] == kind and row['pitch'] == pitch)
                hits = sum(c.get(f'aic_r{i}_read_cache_hit', 0) for i in (0, 1))
                misses = sum(c.get(f'aic_r{i}_read_cache_miss_allocate', 0)
                             for i in (0, 1))
                hit_fractions[kind] = hits / (hits + misses)
            result['traffic_checks'].append({
                'pitch': pitch, 'source_requested_kib': source_bytes,
                'probe_requested_kib': probe_bytes,
                'source_minus_scaled_probe_kib':
                    source_bytes - probe_bytes * pitch / result['logical_k'],
                'source_l2_hit_fraction': hit_fractions['source'],
                'probe_l2_hit_fraction': hit_fractions['probe']})
        results.append(result)
    (HERE / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    for row in results:
        print(f"{row['m']}x{row['n']} core "
              f"{row['tile']['singleCoreM']}x{row['tile']['singleCoreN']} "
              f"probe={row['probe_center_excess_us']:.1f} "
              f"MTE2={row['source_mte2_center_excess_us']:.1f} "
              f"kernel={row['source_kernel_center_excess_us']:.1f} us")


if __name__ == '__main__':
    main()
