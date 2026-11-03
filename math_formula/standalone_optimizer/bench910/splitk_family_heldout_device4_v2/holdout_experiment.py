#!/usr/bin/env python3
"""Audit and time frozen held-out Split-K family decisions on device 4."""

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
sys.path.insert(0, str(ROOT))
import splitk_family_experiment as experiment  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('audit', 'timing'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    metadata = json.loads(args.metadata.read_text())
    if sha256(args.manifest) != metadata['manifest_sha256'] or \
       sha256(args.predictions) != metadata['predictions_sha256']:
        raise RuntimeError('frozen manifest or predictions changed')
    shapes = [tuple(s) for s in manifest['shapes']]
    with args.predictions.open(newline='') as stream:
        predictions = list(csv.DictReader(stream))
    if [(int(r['m']), int(r['n']), int(r['k'])) for r in predictions] != shapes:
        raise RuntimeError('prediction shapes differ from manifest')
    experiment.OUT = args.experiment_dir
    experiment.DEVICE = manifest['device']
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    eligible = []
    audits = []
    for shape in shapes:
        records = {arm: experiment.audit_one(arm, shape)
                   for arm in ('single', 'deterministic', 'default')}
        reasons = []
        for arm, record in records.items():
            if record['returncode'] or not record.get('tiling'):
                reasons.append(f'{arm} audit failed')
                continue
            if record['event']['device'] != manifest['device']:
                reasons.append(f'{arm} ran on wrong device')
            if record['event']['max_relative_error'] > 0.02:
                reasons.append(f'{arm} output check failed')
        modes = {arm: record.get('tiling', {}).get('tilingEnableSplitCore')
                 for arm, record in records.items()}
        if modes['single'] not in (2, 5, 6):
            reasons.append(f"single mode {modes['single']}")
        if modes['deterministic'] != 3:
            reasons.append(f"deterministic mode {modes['deterministic']}")
        if modes['default'] not in (2, 3, 5, 6):
            reasons.append(f"default mode {modes['default']}")
        audits.append({'shape': shape, 'modes': modes, 'reasons': reasons})
        if not reasons:
            eligible.append(shape)
        print('eligibility', shape, modes, reasons, flush=True)
    (args.experiment_dir / 'eligibility.json').write_text(
        json.dumps(audits, indent=2) + '\n')
    if args.phase == 'audit':
        print('eligible', len(eligible), '/', len(shapes), flush=True)
        return
    rng = random.Random(manifest['seed'])
    for round_index in range(manifest['rounds']):
        tasks = []
        for shape in eligible:
            default = experiment.audit_one('default', shape)['tiling']
            same_arm = ('single' if default['tilingEnableSplitCore'] in (2, 5, 6)
                        else 'deterministic')
            matching = experiment.audit_one(same_arm, shape)['tiling']
            if any(default.get(field) != matching.get(field)
                   for field in experiment.GEOMETRY_FIELDS):
                # Time all three arms if forcing a family changes geometry.
                tasks.extend((shape, arm) for arm in
                             ('default', 'single', 'deterministic'))
            else:
                opposite = ('deterministic' if same_arm == 'single'
                            else 'single')
                tasks.extend(((shape, 'default'), (shape, opposite)))
        rng.shuffle(tasks)
        for shape, arm in tasks:
            experiment.time_one(arm, shape, round_index,
                                manifest['repetitions_per_round'])
    print('timed', len(eligible), 'eligible shapes', flush=True)


if __name__ == '__main__':
    main()
