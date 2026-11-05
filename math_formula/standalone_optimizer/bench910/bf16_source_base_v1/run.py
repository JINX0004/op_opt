#!/usr/bin/env python3
"""Run the 100-shape BF16 source-BASE revision against clean master on 910."""

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'bf16_100_vs_master_v1' / 'run.py'
spec = importlib.util.spec_from_file_location('bf16_previous_runner', PREVIOUS)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

BENCH = Path('/home/xianfei/solvebase_bench_20261025')
runner.OUT = BENCH / 'cases/bf16_source_base_v1'
runner.PACKAGES['online'] = BENCH / 'ops_nn_bf16_source_base_v1_package'
LIB = Path('op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so')
runner.HOST_SHA['online'] = '1fd89d86dc5a7b37af3da56cbf0bad74da4be440d940de5b96faa8308d9ee4d1'
DEVICES = (4, 5, 6, 7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, choices=DEVICES, required=True)
    parser.add_argument('--slot', type=int, choices=range(len(DEVICES)),
                        help='dataset partition; defaults to the device index')
    parser.add_argument('--shards', type=int, default=1,
                        help='split the selected partition into this many shards')
    parser.add_argument('--shard', type=int, default=0,
                        help='zero-based shard of the selected partition')
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error('require 0 <= shard < shards')
    slot = DEVICES.index(args.device) if args.slot is None else args.slot
    shapes = json.loads((HERE / 'dataset.json').read_text())
    if len(shapes) != 100 or len({tuple(x) for x in shapes}) != 100:
        raise RuntimeError('dataset must contain 100 distinct shapes')
    if hashlib.sha256((runner.PACKAGES['online'] / LIB).read_bytes()).hexdigest() != runner.HOST_SHA['online']:
        raise RuntimeError('online package host hash changed')
    runner.check_packages()
    partition = [(i, shape) for i, shape in enumerate(shapes)
                 if i % len(DEVICES) == slot]
    assigned = [item for position, item in enumerate(partition)
                if position % args.shards == args.shard]
    for position, (index, shape) in enumerate(assigned):
        shape_dir = f'{index:03d}_{runner.label(shape)}'
        base = runner.OUT / f'device{args.device}' / shape_dir
        record_path = base / 'record.json'
        if any(runner.OUT.glob(f'device*/{shape_dir}/record.json')):
            print(f'skip {index:03d} {runner.label(shape)}', flush=True)
            continue
        print(f'start {index:03d} {runner.label(shape)} device={args.device}', flush=True)
        try:
            audits = {arm: runner.audit(index, shape, args.device, arm)
                      for arm in ('master', 'online')}
            a, b = (audits[arm]['tiling'] for arm in ('master', 'online'))
            changed = any(a.get(field) != b.get(field) for field in runner.TILE_FIELDS)
            mode5 = a['tilingEnableSplitCore'] == 5 or b['tilingEnableSplitCore'] == 5
            rounds = 2 if changed or mode5 else 1
            measured = []
            for round_index in range(rounds):
                order = ('master', 'online') if (position + round_index) % 2 == 0 else ('online', 'master')
                for arm in order:
                    row = runner.timing(index, shape, args.device, arm, round_index,
                                        audits[arm])
                    measured.append(row)
                    print(f"  r{round_index} {arm} mode={row['tiling']['tilingEnableSplitCore']} "
                          f"{row['kernel_median_us']:.3f} us", flush=True)
            runner.write(record_path, {'index': index, 'shape': shape,
                                       'device': args.device,
                                       'changed_tiling': changed,
                                       'mode5_control': mode5,
                                       'audits': audits, 'measurements': measured})
            for failure in runner.OUT.glob(f'device*/{shape_dir}/failure.json'):
                failure.unlink()
            print(f'done {index:03d} {runner.label(shape)}', flush=True)
        except Exception as exc:
            runner.write(base / 'failure.json',
                         {'index': index, 'shape': shape, 'device': args.device,
                          'error': str(exc), 'traceback': traceback.format_exc()})
            print(f'FAILED {index:03d} {runner.label(shape)}: {exc}', flush=True)
            if 'occupied' in str(exc):
                raise
    print(f'worker device {args.device} finished {len(assigned)} shapes', flush=True)


if __name__ == '__main__':
    main()
