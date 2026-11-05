#!/usr/bin/env python3
"""Repeat a same-tile BF16 outlier in both benchmark orders."""

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bf16_source_runner', HERE / 'run.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runner = module.runner
runner.OUT = module.BENCH / 'cases/bf16_source_base_v1_same_tile_repeat'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    shapes = json.loads((HERE / 'dataset.json').read_text())
    shape = shapes[args.index]
    runner.check_packages()
    audits = {arm: runner.audit(args.index, shape, args.device, arm)
              for arm in ('master', 'online')}
    a, b = (audits[arm]['tiling'] for arm in ('master', 'online'))
    if any(a.get(field) != b.get(field) for field in runner.TILE_FIELDS):
        raise RuntimeError('repeat target no longer emits identical tiling')
    measurements = []
    for round_index, order in enumerate((('master', 'online'),
                                         ('online', 'master'))):
        for arm in order:
            row = runner.timing(args.index, shape, args.device, arm,
                                round_index, audits[arm])
            measurements.append(row)
            print(round_index, arm, row['kernel_median_us'], flush=True)
    path = runner.OUT / f'device{args.device}' / f'{args.index:03d}_{runner.label(shape)}' / 'record.json'
    runner.write(path, {'index': args.index, 'shape': shape,
                        'device': args.device, 'audits': audits,
                        'measurements': measurements})


if __name__ == '__main__':
    main()
