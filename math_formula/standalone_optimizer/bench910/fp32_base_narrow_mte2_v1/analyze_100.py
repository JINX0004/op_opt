#!/usr/bin/env python3
"""Combine 100 new selector decisions with fresh pairs for changed families."""

import csv
import json
import math
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'fp32_base_formula_v2/latest_selector_comparison.csv'
DECISIONS = HERE / 'selector_100.csv'
PAIRS = HERE / 'selector_change_pairs.json'


def paired_time(record, arm):
    values = [item['profile']['median_us'] for item in record['measurements']
              if item['arm'] == arm]
    if len(values) != 2:
        raise RuntimeError((record['index'], arm, values))
    return statistics.median(values)


def geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def summarize(rows):
    ratios = [row['ratio'] for row in rows]
    return {'count': len(rows), 'geomean_master_over_selector': geomean(ratios),
            'selector_faster': sum(x > 1 for x in ratios),
            'master_faster': sum(x < 1 for x in ratios)}


def main():
    prior = list(csv.DictReader(PRIOR.open()))
    decisions = list(csv.DictReader(DECISIONS.open()))
    pairs = {r['index']: r for r in json.loads(PAIRS.read_text())}
    if len(prior) != 100 or len(decisions) != 100 or set(pairs) != {97, 98, 99}:
        raise RuntimeError((len(prior), len(decisions), sorted(pairs)))
    rows = []
    for index, (old, decision) in enumerate(zip(prior, decisions)):
        shape = tuple(map(int, old['shape'].split('x')))
        if int(old['index']) != index or shape != tuple(int(decision[key])
                                                       for key in ('m', 'n', 'k')):
            raise RuntimeError(f'shape order mismatch at {index}')
        old_mode, new_mode = int(old['selector_mode']), int(decision['mode'])
        if (old_mode != new_mode) != (index in pairs):
            raise RuntimeError(f'unmeasured decision change at {index}')
        if index in pairs:
            record = pairs[index]
            if record['new_model_mode'] != new_mode:
                raise RuntimeError(f'measured wrong family at {index}')
            master_us, selector_us = (paired_time(record, arm)
                                      for arm in ('master', 'proposed'))
            for arm in ('master', 'proposed'):
                if record['audits'][arm]['event']['max_relative_error'] > .02:
                    raise RuntimeError(f'output error at {index}/{arm}')
        else:
            master_us = float(old['master_us'])
            selector_us = float(old['selector_us'])
        rows.append({'index': index, 'shape': old['shape'],
                     'master_mode': int(old['master_mode']),
                     'previous_mode': old_mode, 'selector_mode': new_mode,
                     'model_score_us': float(decision['objective_us']),
                     'master_us': master_us, 'selector_us': selector_us,
                     'ratio': master_us / selector_us,
                     'measurement_source': 'fresh paired run' if index in pairs
                                           else 'prior audited pair',
                     'family_changed_vs_master': int(old['master_mode']) != new_mode})
    with (HERE / 'selector_100_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    random.seed(9104)
    bootstrap = sorted(geomean(rows[random.randrange(100)]['ratio']
                               for _ in range(100)) for _ in range(10000))
    result = {
        'all': summarize(rows),
        'family_changed': summarize([r for r in rows if r['family_changed_vs_master']]),
        'family_unchanged': summarize([r for r in rows if not r['family_changed_vs_master']]),
        'bootstrap_95pct': [bootstrap[249], bootstrap[9749]],
        'new_decisions': [r for r in rows if r['previous_mode'] != r['selector_mode']],
        'reused_pairs': 97,
        'fresh_pairs': 3,
    }
    (HERE / 'selector_100_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
