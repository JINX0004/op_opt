#!/usr/bin/env python3
"""Paired, source-audited FP32 comparison of master and the online selector.

The online package is copied from master with only the MatMulV3 host library
replaced. An unforced run from that package checks default-path parity first.
"""

import argparse
import csv
import importlib.util
import json
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/xianfei/solvebase_bench_20261025')
OLD = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_source_family_runner', OLD)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

NEW_PACKAGE = ROOT / 'ops_nn_fp32_online_30_package'
NEW_HOST_SHA = '33855e2c05271c69389365bfbf344d4c6b5382607bf806fd116d9bdba7d302c0'
SOURCE = '/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/'
runner.OUT = ROOT / 'cases/fp32_online_30_new_v1'
runner.PACKAGES.update(research_default=NEW_PACKAGE, online=NEW_PACKAGE)
runner.SOURCES.update(research_default=SOURCE, online=SOURCE)
runner.HASHES.update(research_default=NEW_HOST_SHA, online=NEW_HOST_SHA)
original_environment = runner.environment


def environment(arm, model_mode, log_level):
    env = original_environment(arm, model_mode, log_level)
    env.pop('MMV3_FP32_MODE5_EXPERIMENT', None)
    if arm == 'online':
        env['MMV3_ONLINE_SELECTOR'] = '1'
    return env


runner.environment = environment


def decisions():
    shapes = json.loads((HERE / 'dataset.json').read_text())
    if len(shapes) != 30 or len({tuple(s) for s in shapes}) != 30:
        raise RuntimeError('expected 30 unique shapes')
    rows = list(csv.DictReader((HERE / 'decisions.csv').open()))
    if len(rows) != len(shapes):
        raise RuntimeError('missing model decisions')
    for shape, row in zip(shapes, rows):
        if shape != [int(row[k]) for k in ('m', 'n', 'k')]:
            raise RuntimeError('model decision shape/order mismatch')
    return shapes, rows


def check_package():
    for arm in ('master', 'online'):
        host = runner.PACKAGES[arm] / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
        if runner.sha(host) != runner.HASHES[arm]:
            raise RuntimeError(f'{arm} host library hash changed')


def worker(device, max_cases=None):
    shapes, models = decisions()
    assigned = [(i, shape) for i, shape in enumerate(shapes) if i % 4 == device - 4]
    if max_cases is not None:
        assigned = assigned[:max_cases]
    for position, (index, shape) in enumerate(assigned):
        base = runner.OUT / f'device{device}' / f'{index:03d}_{runner.label(shape)}'
        record_path = base / 'record.json'
        if record_path.exists():
            print(f'skip {index:03d} {runner.label(shape)}', flush=True)
            continue
        model = models[index]
        print(f'start {index:03d} {runner.label(shape)} device={device} model={model["mode"]}', flush=True)
        try:
            audits = {arm: runner.audit(index, shape, device, arm, int(model['mode']))
                      for arm in ('master', 'research_default', 'online')}
            master = audits['master']['tiling']
            control = audits['research_default']['tiling']
            default_diff = {key: [master.get(key), control.get(key)]
                            for key in runner.TILE_FIELDS if master.get(key) != control.get(key)}
            if default_diff:
                raise RuntimeError(f'unforced research host differs from master: {default_diff}')
            online = audits['online']['tiling']
            changed = any(master.get(key) != online.get(key) for key in runner.TILE_FIELDS)
            rounds = 2 if changed else 1
            measured = []
            for round_index in range(rounds):
                order = ('master', 'online') if (position + round_index) % 2 == 0 else ('online', 'master')
                for arm in order:
                    item = runner.profile(index, shape, device, arm, int(model['mode']),
                                          round_index, audits[arm])
                    measured.append(item)
                    print(f'  r{round_index} {arm} mode={item["mode"]} '
                          f'{item["profile"]["median_us"]:.3f} us', flush=True)
            runner.write(record_path, dict(index=index, shape=shape, device=device,
                                           model=model, changed_tiling=changed,
                                           audits=audits, measurements=measured))
            (base / 'failure.json').unlink(missing_ok=True)
            print(f'done {index:03d} {runner.label(shape)}', flush=True)
        except Exception as error:
            runner.write(base / 'failure.json', dict(index=index, shape=shape,
                         device=device, error=str(error), traceback=traceback.format_exc()))
            print(f'FAILED {index:03d} {runner.label(shape)}: {error}', flush=True)
            if 'occupied' in str(error):
                raise
    print(f'worker device {device} finished {len(assigned)} shapes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, choices=(4, 5, 6, 7))
    parser.add_argument('--max-cases', type=int)
    args = parser.parse_args()
    check_package()
    worker(args.device, args.max_cases)
