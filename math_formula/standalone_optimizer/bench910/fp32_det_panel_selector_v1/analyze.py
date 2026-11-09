#!/usr/bin/env python3
"""Compare the experimental host with the existing selector and master."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent / 'fp32_source_mode2_preview_100_v1'
MODES = {0: 'BASE', 2: 'mode 2', 3: 'deterministic split-K', 5: 'mode 5', 6: 'mode 6'}
TILE_FIELDS = ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN', 'singleCoreK',
               'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb',
               'depthA1', 'depthB1', 'block_dim', 'workspace_bytes',
               'nd2nzA', 'nd2nzB', 'tiling_key')


def signature(audit):
    return tuple(audit['tiling'][key] for key in TILE_FIELDS)


def median(profiles, arm, rounds):
    values = [p for p in profiles if p['arm'] == arm]
    if len(values) != rounds or any(p['profile']['kernel_count'] != 5 or
                                  p['event']['max_relative_error'] > .02 for p in values):
        raise ValueError(f'invalid profiles for {arm}')
    return statistics.median(p['profile']['median_us'] for p in values)


def summarize(rows, key):
    values = [r[key] for r in rows]
    if not values:
        return {}
    return {'count': len(values), 'geomean': math.exp(statistics.mean(map(math.log, values))),
            'new_faster_5pct': sum(v > 1.05 for v in values),
            'new_slower_5pct': sum(v < 1 / 1.05 for v in values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('records', type=Path)
    args = parser.parse_args()
    prior = {r['index']: r for r in json.loads((PREVIEW / 'records.json').read_text())}
    records = sorted(json.loads(args.records.read_text()), key=lambda r: r['index'])
    if [r['index'] for r in records] != list(range(100)):
        raise ValueError('expected 100 distinct shape audits')

    rows = []
    for r in records:
        p = prior[r['index']]
        a = r['audits']
        if (r['shape'] != p['shape'] or r['device'] != p['device'] or
                a['online']['event']['max_relative_error'] > .02):
            raise ValueError(f"shape, device, or correctness audit failed at {r['index']}")
        if (signature(a['master']) != signature(p['audits']['master']) or
                signature(a['legacy']) != signature(p['audits']['online'])):
            raise ValueError(f"prior audit mismatch at {r['index']}")
        if r['changed'] != (signature(a['legacy']) != signature(a['online'])):
            raise ValueError(f"changed tile mismatch at {r['index']}")
        if r['profiles']:
            times = {}
            for arm, rep in r['representatives'].items():
                if signature(a[arm]) != signature(a[rep]):
                    raise ValueError(f"nonidentical tiles share a profile at {r['index']}")
                times[arm] = median(r['profiles'], rep, 2)
            timing = 'fresh'
        else:
            if r['changed']:
                raise ValueError(f"changed tile has no fresh profile at {r['index']}")
            rounds = 2 if p['changed_tiling'] else 1
            times = dict(master=median(p['measurements'], 'master', rounds),
                         legacy=median(p['measurements'], 'online', rounds))
            times['online'] = times['legacy']
            timing = 'reused after tile audit'
        row = dict(index=r['index'], shape='x'.join(map(str, r['shape'])), timing=timing)
        for arm in ('master', 'legacy', 'online'):
            t = a[arm]['tiling']
            mode = t['tilingEnableSplitCore']
            row[arm + '_family'] = MODES.get(mode, str(mode))
            row[arm + '_core'] = 'x'.join(str(t[k]) for k in
                                          ('singleCoreM', 'singleCoreN', 'singleCoreK'))
            row[arm + '_us'] = times[arm]
        row.update(tile_changed=r['changed'],
                   family_changed=row['legacy_family'] != row['online_family'],
                   family_differs_from_master=row['master_family'] != row['online_family'],
                   legacy_over_new=times['legacy'] / times['online'],
                   master_over_new=times['master'] / times['online'])
        scores = r['det_scores']
        if any(pair != scores[0] for pair in scores):
            raise ValueError(f"inconsistent deterministic scores at {r['index']}")
        row['legacy_det_score_us'], row['new_det_score_us'] = scores[0] if scores else ('', '')
        rows.append(row)

    changed = [r for r in rows if r['family_changed']]
    result = {
        'audits': len(rows),
        'rescored_deterministic_candidates': sum(r['new_det_score_us'] != '' for r in rows),
        'fresh_cases': sum(r['timing'] == 'fresh' for r in rows),
        'tile_changes_vs_legacy': sum(r['tile_changed'] for r in rows),
        'family_changes_vs_legacy': len(changed),
        'family_differences_vs_master': sum(r['family_differs_from_master'] for r in rows),
        'legacy_over_new': summarize(rows, 'legacy_over_new'),
        'master_over_new': summarize(rows, 'master_over_new'),
        'legacy_over_new_family_changes': summarize(changed, 'legacy_over_new'),
        'master_over_new_family_changes': summarize(changed, 'master_over_new'),
    }
    for filename, data in [('comparison.csv', rows), ('family_changes.csv', changed)]:
        with (HERE / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
