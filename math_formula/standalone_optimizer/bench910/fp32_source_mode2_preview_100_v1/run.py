#!/usr/bin/env python3
"""Compare source-previewed FP32 mode-2 selection against clean master."""

import argparse
import importlib.util
import json
import traceback
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
BASELINE = ROOT / 'fp32_source_base_100_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_source_base_100_runner', BASELINE)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
runner = experiment.runner

PACKAGE = ROOT / 'ops_nn_fp32_source_mode2_preview_package'
HOST_SHA = '2df4299101331a24044cb1c15a8f6487b22d0593f79aabe0af1a451272576b3a'
runner.OUT = ROOT / 'cases/fp32_source_mode2_preview_100_v1'
runner.PACKAGES.update(research_default=PACKAGE, online=PACKAGE)
runner.HASHES.update(research_default=HOST_SHA, online=HOST_SHA)


def run_case(index, shapes):
    shape = shapes[index]
    device = 4 + index % 4
    base = runner.OUT / f'device{device}' / f'{index:03d}_{runner.label(shape)}'
    record_path = base / 'record.json'
    if record_path.exists():
        print(f'skip {index:03d} {runner.label(shape)}', flush=True)
        return
    print(f'start {index:03d} {runner.label(shape)} device={device}', flush=True)
    try:
        audits = {arm: runner.audit(index, shape, device, arm, 0)
                  for arm in ('master', 'research_default', 'online')}
        master = audits['master']['tiling']
        control = audits['research_default']['tiling']
        default_diff = {key: [master.get(key), control.get(key)]
                        for key in runner.TILE_FIELDS if master.get(key) != control.get(key)}
        if default_diff:
            raise RuntimeError(f'unforced host differs from master: {default_diff}')
        online = audits['online']['tiling']
        changed = any(master.get(key) != online.get(key) for key in runner.TILE_FIELDS)
        measured = []
        for round_index in range(2 if changed else 1):
            order = ('master', 'online') if (index + round_index) % 2 == 0 \
                else ('online', 'master')
            for arm in order:
                item = runner.profile(index, shape, device, arm, 0,
                                      round_index, audits[arm])
                measured.append(item)
                print(f'  r{round_index} {arm} mode={item["mode"]} '
                      f'{item["profile"]["median_us"]:.3f} us', flush=True)
        runner.write(record_path, dict(index=index, shape=shape, device=device,
                                       changed_tiling=changed, audits=audits,
                                       measurements=measured))
        (base / 'failure.json').unlink(missing_ok=True)
        print(f'done {index:03d} {runner.label(shape)}', flush=True)
    except Exception as error:
        runner.write(base / 'failure.json', dict(index=index, shape=shape,
                     device=device, error=str(error), traceback=traceback.format_exc()))
        print(f'FAILED {index:03d} {runner.label(shape)}: {error}', flush=True)
        if 'occupied' in str(error):
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--case', type=int)
    group.add_argument('--device', type=int, choices=(4, 5, 6, 7))
    args = parser.parse_args()
    experiment.experiment.check_package()
    shapes = json.loads((ROOT / 'fp32_source_base_100_v1/dataset.json').read_text())
    if len(shapes) != 100:
        raise RuntimeError('expected the fixed 100-case dataset')
    indices = [args.case] if args.case is not None else [
        i for i in range(100) if i % 4 == args.device - 4]
    for index in indices:
        if not 0 <= index < 100:
            raise ValueError(f'case index outside dataset: {index}')
        run_case(index, shapes)
