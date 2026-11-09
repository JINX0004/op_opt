#!/usr/bin/env python3
"""Profile the 70 added FP32 shapes against master and the fixed online host."""

import argparse
import importlib.util
import json
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/xianfei/solvebase_bench_20261025')
BASELINE = ROOT / 'fp32_online_30_new_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_online_30_runner', BASELINE)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)

PACKAGE = ROOT / 'ops_nn_fp32_source_base_fix_package'
HOST_SHA = 'b49ce309d8760e2cab3ddaf0eb286684ead170ca775fb8811b4ae1db61925c75'
runner = experiment.runner
runner.OUT = ROOT / 'cases/fp32_source_base_100_v1'
runner.PACKAGES.update(research_default=PACKAGE, online=PACKAGE)
runner.HASHES.update(research_default=HOST_SHA, online=HOST_SHA)


def worker(device, max_cases=None):
    shapes = json.loads((HERE / 'dataset.json').read_text())
    if len(shapes) != 100 or len({tuple(x) for x in shapes}) != 100:
        raise RuntimeError('expected 100 unique shapes')
    indices = [i for i in range(30, 100) if i % 4 == device - 4]
    if max_cases is not None:
        indices = indices[:max_cases]
    for position, index in enumerate(indices):
        shape = shapes[index]
        base = runner.OUT / f'device{device}' / f'{index:03d}_{runner.label(shape)}'
        record_path = base / 'record.json'
        if record_path.exists():
            print(f'skip {index:03d} {runner.label(shape)}', flush=True)
            continue
        print(f'start {index:03d} {runner.label(shape)} device={device}', flush=True)
        try:
            audits = {arm: runner.audit(index, shape, device, arm, 0)
                      for arm in ('master', 'research_default', 'online')}
            master = audits['master']['tiling']
            control = audits['research_default']['tiling']
            default_diff = {key: [master.get(key), control.get(key)]
                            for key in runner.TILE_FIELDS if master.get(key) != control.get(key)}
            if default_diff:
                raise RuntimeError(f'unforced research host differs from master: {default_diff}')
            online = audits['online']['tiling']
            changed = any(master.get(key) != online.get(key) for key in runner.TILE_FIELDS)
            measured = []
            for round_index in range(2 if changed else 1):
                order = ('master', 'online') if (position + round_index) % 2 == 0 else ('online', 'master')
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
    print(f'worker device {device} finished {len(indices)} added shapes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, choices=(4, 5, 6, 7))
    parser.add_argument('--max-cases', type=int)
    args = parser.parse_args()
    experiment.check_package()
    worker(args.device, args.max_cases)
