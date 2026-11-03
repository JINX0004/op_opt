#!/usr/bin/env python3
"""Audit source-backed MatMulV3 single-core mode 2 versus mode 6."""

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
sys.path.insert(0, str(ROOT))
import splitk_family_experiment as experiment  # noqa: E402


PACKAGE = ROOT / 'ops_nn_single_mode_package'
base_environment = experiment.environment


def environment(arm: str, shape: tuple[int, int, int]) -> dict[str, str]:
    env = base_environment('default' if arm == 'default' else 'single', shape)
    if arm in ('mode2', 'mode6'):
        env['MMV3_RESEARCH_SC_MODE'] = arm[-1]
    return env


experiment.PACKAGE = PACKAGE
experiment.environment = environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pilot', 'audit', 'timing'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, default=4)
    args = parser.parse_args()
    experiment.OUT = args.experiment_dir
    experiment.DEVICE = args.device
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == 'pilot':
        shape = (2048, 2048, 32768)
        for arm in ('mode2', 'mode6', 'default'):
            row = experiment.audit_one(arm, shape)
            print(arm, row['returncode'], row.get('tiling', {}).get('tilingEnableSplitCore'),
                  row.get('event', {}).get('max_relative_error'), flush=True)
        return
    if args.manifest is None:
        parser.error('--manifest is required for audit and timing')
    manifest = json.loads(args.manifest.read_text())
    shapes = [tuple(x) for x in manifest['shapes']]
    audits = []
    eligible = []
    for shape in shapes:
        rows = {arm: experiment.audit_one(arm, shape) for arm in ('mode2', 'mode6', 'default')}
        modes = {arm: row.get('tiling', {}).get('tilingEnableSplitCore') for arm, row in rows.items()}
        reasons = []
        for arm, row in rows.items():
            if row['returncode'] != 0 or 'event' not in row or 'tiling' not in row:
                reasons.append(f'{arm}: audit failed')
            elif row['event']['device'] != args.device or row['event']['max_relative_error'] > 0.02:
                reasons.append(f'{arm}: device or correctness failed')
        if modes['mode2'] != 2 or modes['mode6'] != 6:
            reasons.append(f'unexpected forced modes: {modes}')
        audits.append({'shape': shape, 'modes': modes, 'reasons': reasons})
        if not reasons:
            eligible.append(shape)
        print('eligibility', shape, modes, reasons, flush=True)
    (args.experiment_dir / 'eligibility.json').write_text(json.dumps(audits, indent=2) + '\n')
    if args.phase == 'audit':
        print('eligible', len(eligible), '/', len(shapes), flush=True)
        return
    rng = random.Random(manifest['seed'])
    for round_index in range(manifest['rounds']):
        tasks = [(shape, arm) for shape in eligible for arm in ('mode2', 'mode6')]
        rng.shuffle(tasks)
        for shape, arm in tasks:
            experiment.time_one(arm, shape, round_index, manifest['repetitions_per_round'])
    print('timed', len(eligible), 'eligible shapes', flush=True)


if __name__ == '__main__':
    main()
