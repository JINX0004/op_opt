#!/usr/bin/env python3
"""Fit a width-aware source-step packet law from primitive task slopes."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
records = {}
for line in (HERE / 'task_sweep.jsonl').read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    key = (row['operand'], row['tile_axis'], row['k_span'], row['cores'])
    records.setdefault(key, {})[row['copies']] = row['median_us']
slopes = {key: (v[25] - v[1]) / 24 for key, v in records.items()}
PACKET = 131072.0


def endpoint(cores):
    a_full = slopes['A', 16, 4096, cores]
    a_half = slopes['A', 32, 1024, cores]
    b_full = slopes['B', 32, 2048, cores]
    b_half = slopes['B', 32, 1024, cores]
    a_fixed = 2 * a_half - a_full
    b_fixed = 2 * b_half - b_full
    a_byte_us = (a_full - a_fixed) / PACKET
    b_byte_us = (b_full - b_fixed) / PACKET
    narrow_full = slopes['B', 16, 4096, cores]
    narrow_row_us = (narrow_full - b_full) / 4096
    return {'a_fixed_us': a_fixed, 'a_byte_us': a_byte_us,
            'b_fixed_us': b_fixed, 'b_byte_us': b_byte_us,
            'b_narrow_row_us': narrow_row_us}

params = {'one_core': endpoint(1), 'twenty_core': endpoint(20),
          'narrow_contention_exponent': 2}


def predict(operand, width, span, cores):
    occupancy = (cores - 1) / 19
    one, many = params['one_core'], params['twenty_core']
    def blend(name):
        return one[name] + occupancy * (many[name] - one[name])
    prefix = operand.lower()
    value = blend(prefix + '_fixed_us') + 2 * width * span * blend(prefix + '_byte_us')
    if operand == 'B' and 2 * width < 64:
        row_us = one['b_narrow_row_us'] + occupancy ** 2 * (
            many['b_narrow_row_us'] - one['b_narrow_row_us'])
        value += span * row_us
    return value

train = {
    ('A', 16, 4096, 1), ('A', 32, 1024, 1),
    ('A', 16, 4096, 20), ('A', 32, 1024, 20),
    ('B', 32, 2048, 1), ('B', 32, 1024, 1),
    ('B', 32, 2048, 20), ('B', 32, 1024, 20),
    ('B', 16, 4096, 1), ('B', 16, 4096, 20),
}
checks = []
for key, measured in sorted(slopes.items()):
    if key in train:
        continue
    predicted = predict(*key)
    checks.append({'operand': key[0], 'tile_axis': key[1],
                   'k_span': key[2], 'cores': key[3],
                   'measured_us_per_packet': measured,
                   'predicted_us_per_packet': predicted,
                   'relative_error': (predicted - measured) / measured})
output = {'parameters': params, 'holdout_checks': checks,
          'max_abs_holdout_relative_error': max(abs(r['relative_error']) for r in checks),
          'model_base_16x16x102400_us':
              6.2003888953010853 + 25 * (
                  predict('A', 16, 4096, 1) + predict('B', 16, 4096, 1)) + 0.162586}
(HERE / 'fit.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
