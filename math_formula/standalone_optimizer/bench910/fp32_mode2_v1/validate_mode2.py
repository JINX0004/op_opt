#!/usr/bin/env python3
"""Paired source audit of newly selected FP32 mode-2 families on 910."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = ROOT / 'fp32_100_source_family_v1'
spec = importlib.util.spec_from_file_location('fp32_100_runner', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = ROOT / 'cases/fp32_mode2_v1'
original_environment = runner.environment


def environment(arm, model_mode, log_level):
    env = original_environment('research_default' if arm == 'proposed' else arm,
                               model_mode, log_level)
    if arm == 'proposed':
        env['MMV3_RESEARCH_SPLITK_FAMILY'] = 'single'
    return env


runner.environment = environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, choices=range(4, 8))
    args = parser.parse_args()
    for arm in ('master', 'proposed'):
        library = (runner.PACKAGES[arm] /
                   'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so')
        if runner.sha(library) != runner.HASHES[arm]:
            raise RuntimeError(f'{arm} host library changed')
    shapes = json.loads((HERE / 'dataset.json').read_text())
    decisions = list(csv.DictReader((HERE / 'mode2_decisions.csv').open()))
    if len(shapes) != 100 or len(decisions) != 100:
        raise RuntimeError('shape/decision count mismatch')
    for index, (shape, decision) in enumerate(zip(shapes, decisions)):
        if index % 4 + 4 != args.device or int(decision['mode']) != 2:
            continue
        directory = runner.OUT / f'device{args.device}' / f'{index:03d}_{runner.label(shape)}'
        result = directory / 'record.json'
        if result.exists():
            print(f'skip {index}', flush=True)
            continue
        print(f'start {index} {runner.label(shape)} device {args.device}', flush=True)
        master = runner.audit(index, shape, args.device, 'master', 2)
        proposed = runner.audit(index, shape, args.device, 'proposed', 2)
        different = {field: (master['tiling'].get(field), proposed['tiling'].get(field))
                     for field in runner.TILE_FIELDS
                     if master['tiling'].get(field) != proposed['tiling'].get(field)}
        if different:
            raise RuntimeError(f'forced mode 2 changed source tiling: {index} {different}')
        prof_master = runner.profile(index, shape, args.device, 'master', 2, 0, master)
        prof_proposed = runner.profile(index, shape, args.device, 'proposed', 2, 0, proposed)
        runner.write(result, {'index': index, 'shape': shape, 'device': args.device,
                              'decision': decision,
                              'audits': {'master': master, 'proposed': proposed},
                              'measurements': [prof_master, prof_proposed]})
        print(f'done {index}: master={prof_master["profile"]["median_us"]:.3f} '
              f'proposed={prof_proposed["profile"]["median_us"]:.3f} us', flush=True)


if __name__ == '__main__':
    main()
