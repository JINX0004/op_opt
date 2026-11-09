#!/usr/bin/env python3
"""Compare source BASE spikes, read-only probes, and AIC counters."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def probe_time(rows, operand, m, n, k):
    def elapsed(repeats):
        return next(row['device_elapsed_us'] for row in rows
                    if (row['operand'], row['m'], row['n'], row['k'], row['repeats'])
                    == (operand, m, n, k, repeats))
    return elapsed(1) - elapsed(0)


def main():
    source = {(row['shape'][0], row['shape'][1], row['shape'][2]): row['median_us']
              for row in read('same_device_records.json')}
    probes = {name: read(name + '.json')
              for name in ('source_stream', 'combined_stream', 'region_stream')}
    for m, n, ks in ((2560, 768, (32512, 32768, 33024)),
                     (2560, 1024, (48896, 49152, 49408))):
        before, center, after = ks
        actual = source[m, n, center] - (source[m, n, before] + source[m, n, after]) / 2
        print(f'{m}x{n}: whole BASE excess={actual:.1f} us')
        for name, operand in (('source_stream', 'A'), ('source_stream', 'B'),
                              ('combined_stream', 'AB'), ('region_stream', 'AB_REGION')):
            rows = probes[name]
            times = [probe_time(rows, operand, m, n, k) for k in ks]
            excess = times[1] - (times[0] + times[2]) / 2
            print(f'  {operand:9s} {times[0]:8.1f} {times[1]:8.1f} {times[2]:8.1f} us; '
                  f'excess={excess:7.1f} us ({excess / actual:.1%} of whole)')
    print('AIC counters, 2560x768:')
    metrics = {(row['k'], row['metric']): row['metrics'] for row in read('pipeline_metrics.json')}
    for k in (32512, 32768, 33024):
        pipe = metrics[k, 'PipeUtilization']
        memory = metrics[k, 'MemoryAccess']
        cache = metrics[k, 'L2Cache']
        hit = cache['aic_r0_read_cache_hit'] + cache['aic_r1_read_cache_hit']
        miss = cache['aic_r0_read_cache_miss_allocate'] + cache['aic_r1_read_cache_miss_allocate']
        print(f'  K={k}: cycles={pipe["aic_total_cycles"] / 1e6:.2f} M, '
              f'MAC={pipe["aic_mac_ratio_extra"]:.1%}, '
              f'MTE2={pipe["aic_mte2_ratio"]:.1%}, '
              f'GM->L1={memory["aic_GM_to_L1_datas"] * 1024 / 1e9:.3f} GB, '
              f'L2-hit={hit / (hit + miss):.2%}')


if __name__ == '__main__':
    main()
