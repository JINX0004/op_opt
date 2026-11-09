#!/usr/bin/env python3
"""Summarize the fixed-tile deterministic Split-K panel sweep."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = tuple(range(384, 3841, 384)) + (4096,)
METRICS = ('aic_total_time', 'aic_mac_time', 'aic_mte1_time',
           'aic_mte2_time', 'aic_fixpipe_time', 'aic_scalar_time')


def line_fit(points):
    mx = statistics.mean(x for x, _ in points)
    my = statistics.mean(y for _, y in points)
    slope = sum((x - mx) * (y - my) for x, y in points) / \
        sum((x - mx) ** 2 for x, _ in points)
    intercept = my - slope * mx
    r2 = 1 - sum((y - intercept - slope * x) ** 2 for x, y in points) / \
        sum((y - my) ** 2 for _, y in points)
    return {'slope_us_per_panel': slope, 'intercept_us': intercept, 'r2': r2}


def main():
    records = json.loads((HERE / 'panel_records.json').read_text())
    audits = json.loads((HERE / 'audits.json').read_text())
    memory = json.loads((HERE / 'memory_records.json').read_text())
    mode2_memory = json.loads((HERE / 'mode2_memory_record.json').read_text())
    model = {int(fields[1]): float(fields[4]) for fields in
             (line.split() for line in (HERE / 'model_scores.txt').read_text().splitlines())}
    by_n = defaultdict(list)
    for row in records:
        by_n[row['shape'][1]].append(row)
        if row['max_relative_error'] > .02 or len(row['kernel_tasks_us']) != 5:
            raise RuntimeError(('profile audit failed', row['shape'], row['round']))
    if sorted(by_n) != list(NS) or any(sorted(x['round'] for x in rows) != [0, 1]
                                      for rows in by_n.values()):
        raise RuntimeError('expected 11 N values and two rounds each')
    if sorted(x['n'] for x in audits) != list(NS):
        raise RuntimeError('incomplete source tiling audits')
    reference = audits[0]['tiling']
    if any(x['tiling'] != reference or x['max_relative_error'] > .02 for x in audits):
        raise RuntimeError('source tile changed or output audit failed')
    by_memory = {(x['shape'][1], x['metric']): x for x in memory}
    table = []
    for n in NS:
        items = by_n[n]
        t = statistics.median(x['kernel_us'] for x in items)
        row = {'n': n, 'panels': (n + 383) // 384,
               'kernel_us': t, 'model_us': model[n],
               'residual_us': t - model[n],
               'round_difference_us': abs(items[0]['kernel_us'] - items[1]['kernel_us'])}
        for metric in METRICS:
            row[metric + '_us'] = statistics.median(x['metrics_us'][metric]
                                                   for x in items)
        if (n, 'MemoryAccess') in by_memory:
            m = by_memory[(n, 'MemoryAccess')]['metrics']
            row['gm_to_l1_gib'] = m['aic_GM_to_L1_datas'] / 1024**2
            row['l0c_to_gm_gib'] = m['aic_L0C_to_GM_datas'] / 1024**2
            l2 = by_memory[(n, 'L2Cache')]['metrics']
            hits = l2['aic_r0_read_cache_hit'] + l2['aic_r1_read_cache_hit']
            misses = l2['aic_r0_read_cache_miss_allocate'] + \
                l2['aic_r1_read_cache_miss_allocate']
            row['l2_read_hit_fraction'] = hits / (hits + misses)
        else:
            row['gm_to_l1_gib'] = row['l0c_to_gm_gib'] = \
                row['l2_read_hit_fraction'] = ''
        table.append(row)
    full = [row for row in table if row['n'] <= 3840]
    fits = {'kernel_median': line_fit([(x['panels'], x['kernel_us']) for x in full]),
            'model': line_fit([(x['panels'], x['model_us']) for x in full])}
    for round_index in (0, 1):
        fits[f'kernel_round{round_index}'] = line_fit([
            (n // 384, next(x['kernel_us'] for x in by_n[n]
                             if x['round'] == round_index))
            for n in NS if n <= 3840])
    for metric in METRICS:
        fits[metric] = line_fit([(x['panels'], x[metric + '_us']) for x in full])
    summary = {'shape_m_k': [8192, 32768], 'dtype': 'fp32',
               'device_for_timing': 5, 'records': len(records),
               'same_source_tile_all_shapes': True, 'source_tile': reference,
               'fits': fits,
               'missing_us_per_full_panel': fits['kernel_median']['slope_us_per_panel'] -
                                            fits['model']['slope_us_per_panel'],
               'max_forward_reverse_difference_us': max(x['round_difference_us']
                                                        for x in table),
               'n4096': table[-1],
               'mode2_n4096_memory_gib': {
                   'gm_to_l1': mode2_memory['metrics']['aic_GM_to_L1_datas'] / 1024**2,
                   'l0c_to_gm': mode2_memory['metrics']['aic_L0C_to_GM_datas'] / 1024**2}}
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (HERE / 'panel_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({'fits': fits, 'missing_us_per_full_panel':
                      summary['missing_us_per_full_panel'],
                      'n4096': table[-1]}, indent=2))


if __name__ == '__main__':
    main()
