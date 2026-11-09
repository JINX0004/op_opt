#!/usr/bin/env python3
"""Combine fresh opt-in family changes with the audited 100-case control."""

import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
PRIOR = BENCH / 'fp32_mode2_pipeline_100_v1/comparison.csv'
PAIRS = HERE / 'new_family_pairs_records.json'
DECISIONS = HERE / 'new_selector_decisions.csv'


def geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def paired_time(record, arm):
    values = [item['profile']['median_us'] for item in record['measurements']
              if item['arm'] == arm]
    if len(values) != 2:
        raise RuntimeError((record['index'], arm, values))
    return statistics.median(values)


def summarize(rows):
    ratios = [row['ratio'] for row in rows]
    return {'count': len(rows), 'geomean_master_over_selector': geomean(ratios),
            'selector_faster': sum(value > 1 for value in ratios),
            'master_faster': sum(value < 1 for value in ratios)}


def main():
    prior = list(csv.DictReader(PRIOR.open()))
    decisions = list(csv.DictReader(DECISIONS.open()))
    pairs = {record['index']: record for record in json.loads(PAIRS.read_text())}
    if len(prior) != 100 or len(decisions) != 100 or len(pairs) != 7:
        raise RuntimeError((len(prior), len(decisions), len(pairs)))
    rows = []
    for index, (old, decision) in enumerate(zip(prior, decisions)):
        if int(old['index']) != index or [int(decision[key]) for key in ('m', 'n', 'k')] != \
                [int(value) for value in old['shape'].split('x')]:
            raise RuntimeError((index, old, decision))
        old_mode = int(old['pipeline_mode'])
        new_mode = int(decision['mode'])
        changed = old_mode != new_mode
        if changed != (index in pairs):
            raise RuntimeError((index, old_mode, new_mode, index in pairs))
        if changed:
            record = pairs[index]
            if record['new_model_mode'] != new_mode:
                raise RuntimeError((index, new_mode, record['new_model_mode']))
            master_us = paired_time(record, 'master')
            selector_us = paired_time(record, 'proposed')
            if (record['audits']['master']['event']['max_relative_error'] > .02 or
                    record['audits']['proposed']['event']['max_relative_error'] > .02):
                raise RuntimeError(f'output error at {index}')
        else:
            master_us = float(old['master_us'])
            selector_us = float(old['pipeline_family_us'])
        rows.append({'index': index, 'shape': old['shape'],
                     'master_mode': int(old['master_mode']),
                     'previous_mode': old_mode, 'selector_mode': new_mode,
                     'model_score_us': float(decision['objective_us']),
                     'master_us': master_us, 'selector_us': selector_us,
                     'ratio': master_us / selector_us,
                     'measurement_source': 'fresh paired run' if changed
                                           else 'prior audited pair',
                     'family_changed_vs_master': int(old['master_mode']) != new_mode})
    with (HERE / 'latest_selector_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    random.seed(9103)
    distribution = sorted(geomean([rows[random.randrange(100)]['ratio']
                                   for _ in range(100)]) for _ in range(10000))
    result = {'all': summarize(rows),
              'family_changed': summarize([row for row in rows
                                           if row['family_changed_vs_master']]),
              'family_unchanged': summarize([row for row in rows
                                             if not row['family_changed_vs_master']]),
              'transitions': dict(Counter(f"{row['master_mode']}->{row['selector_mode']}"
                                          for row in rows)),
              'bootstrap_95pct': [distribution[249], distribution[9749]],
              'new_decisions': [row for row in rows
                                if row['previous_mode'] != row['selector_mode']]}
    (HERE / 'latest_selector_analysis.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
