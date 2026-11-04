#!/usr/bin/env python3
"""Pair clean master and calibrated online MatMulV3 BF16 on 100 shapes."""

import argparse
import hashlib
import json
import subprocess
import traceback
from pathlib import Path

BENCH = Path('/home/xianfei/solvebase_bench_20261025')
HERE = Path(__file__).resolve().parent
OUT = BENCH / 'cases/bf16_100_vs_master_v1'
CANN = '/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0'
PACKAGES = {
    'master': BENCH / 'ops_nn_master_bf16_proto_package',
    'online': BENCH / 'ops_nn_bf16_mode5_revised_package',
}
SOURCE = {
    'master': '/home/xianfei/dev/ops-nn-sribd-master-bf16/matmul/mat_mul_v3/',
    'online': '/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/',
}
HOST_SHA = {
    'master': 'b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad',
    'online': '3e0d3a91ad4efcf50ce5af3d05ccbd452c45b8d105fc58f9edf2883d7e053471',
}
TILE_FIELDS = ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN', 'singleCoreK',
               'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb',
               'depthA1', 'depthB1', 'block_dim', 'workspace_bytes',
               'nd2nzA', 'nd2nzB', 'tiling_key')


def sourced_cann_env():
    """Give direct audit processes the same library paths as run_case.sh."""
    proc = subprocess.run(
        ['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
         'bash', CANN], capture_output=True, check=True)
    return dict(item.decode().split('=', 1) for item in proc.stdout.split(b'\0')
                if item)


CANN_ENV = sourced_cann_env()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def label(shape):
    return 'x'.join(map(str, shape))


def require_idle(device, directory, phase):
    snapshot = subprocess.run(['npu-smi', 'info'], capture_output=True,
                              text=True, check=True).stdout
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'npu_smi_{phase}.txt').write_text(snapshot)
    if f'No running processes found in NPU {device}' not in snapshot:
        raise RuntimeError(f'device {device} occupied at {phase}')


def env_for(arm):
    package = PACKAGES[arm]
    env = CANN_ENV.copy()
    for key in ('MMV3_ONLINE_SELECTOR', 'MMV3_ONLINE_BF16_EXPERIMENT',
                'MMV3_ONLINE_ROW_READ_PROFILE', 'MMV3_ONLINE_TILE',
                'MMV3_BF16_MODE5_TILE_PROBE'):
        env.pop(key, None)
    env.update(CANN_ROOT=CANN, BENCH_API='v3cached',
               BENCH_CUSTOM_OPP_PATH=str(package),
               ASCEND_CUSTOM_OPP_PATH=str(package),
               MMV3_SKIP_LEGACY_OP_CACHE='1')
    env['LD_LIBRARY_PATH'] = str(package / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    if arm == 'online':
        env.update(MMV3_ONLINE_SELECTOR='1', MMV3_ONLINE_BF16_EXPERIMENT='1')
    return env


def audit(index, shape, device, arm):
    directory = OUT / f'device{device}' / f'{index:03d}_{label(shape)}' / 'audit' / arm
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    require_idle(device, directory, 'before')
    env = env_for(arm)
    env.update(ASCEND_GLOBAL_LOG_LEVEL='1', IGNORE_INFER_ERROR='1')
    cmd = [str(BENCH / 'bf16_online_vs_master/base_aclnn_bench'), '--api', 'v3cached',
           '--device', str(device), '--m', str(shape[0]), '--n', str(shape[1]),
           '--k', str(shape[2]), '--warmup', '0', '--batches', '1', '--repetitions', '1']
    with (directory / 'stdout.jsonl').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=BENCH / 'bf16_online_vs_master',
                              stdout=stdout, stderr=stderr, timeout=600)
    require_idle(device, directory, 'after')
    if proc.returncode:
        raise RuntimeError(f'audit {arm} {shape} exited {proc.returncode}: '
                           + (directory / 'stderr.log').read_text()[-1500:])
    event = json.loads((directory / 'stdout.jsonl').read_text().splitlines()[-1])
    parsed = subprocess.run(['python3', str(BENCH / 'parse_tiling_log.py'),
                             '--pid', str(event['pid'])], env=env,
                            capture_output=True, text=True, check=True)
    tiling = json.loads(parsed.stdout)
    if (event['device'] != device or event['dtype'] != 'bf16' or
            event['max_relative_error'] > .02 or
            not tiling.get('tiling_source', '').startswith(SOURCE[arm])):
        raise RuntimeError(f'audit invalid: {arm} {shape}: {event} {tiling}')
    result = {'index': index, 'shape': shape, 'arm': arm, 'device': device,
              'event': event, 'tiling': tiling}
    write(result_path, result)
    return result


def timing(index, shape, device, arm, round_index, audit_result):
    directory = OUT / f'device{device}' / f'{index:03d}_{label(shape)}' / f'round{round_index}' / arm
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    tiling = audit_result['tiling']
    mode = tiling['tilingEnableSplitCore']
    nd_a, nd_b = tiling['nd2nzA'], tiling['nd2nzB']
    env = env_for(arm)
    env.update(BENCH_EXPECT_SOURCE=SOURCE[arm],
               BENCH_EXPECT_SPLIT_CORE_MODE=str(mode),
               BENCH_EXPECT_ND2NZ_A=str(nd_a), BENCH_EXPECT_ND2NZ_B=str(nd_b),
               BENCH_PROFILE_KERNEL_TYPE=('KERNEL_AICORE' if mode == 0 and
                                          nd_a == 0 and nd_b == 0 else 'MIX_AIC'))
    require_idle(device, directory, 'before')
    with (directory / 'run.log').open('w') as log:
        proc = subprocess.run(['bash', str(BENCH / 'bf16_online_vs_master/run_case.sh'),
                               *map(str, shape), str(directory), str(device), '5', '5'],
                              env=env, cwd=BENCH / 'bf16_online_vs_master',
                              stdout=log, stderr=subprocess.STDOUT, timeout=900)
    require_idle(device, directory, 'after')
    if proc.returncode:
        raise RuntimeError(f'timing {arm} {shape} exited {proc.returncode}: '
                           + (directory / 'run.log').read_text()[-1500:])
    emitted = json.loads((directory / 'tiling.json').read_text())
    for field in TILE_FIELDS:
        if emitted.get(field) != tiling.get(field):
            raise RuntimeError(f'tile changed for {shape} {arm} {field}: '
                               f'{tiling.get(field)} -> {emitted.get(field)}')
    case = json.loads((directory / 'case.json').read_text())
    if (case['max_relative_error'] > .02 or case['device'] != device or
            case['split_core_mode'] != mode or case['kernel_count'] != 5):
        raise RuntimeError(f'profile/correctness mismatch: {shape} {arm}: {case}')
    result = {'index': index, 'shape': shape, 'arm': arm, 'device': device,
              'round': round_index, 'kernel_median_us': case['kernel_median_us'],
              'max_relative_error': case['max_relative_error'],
              'tiling': {field: emitted.get(field) for field in TILE_FIELDS},
              'case_path': str(directory / 'case.json'),
              'profile_csv': case['profile_csv']}
    write(result_path, result)
    return result


def check_packages():
    for arm, package in PACKAGES.items():
        library = (package / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/'
                   'libcust_opmaster_rt2.0.so')
        actual = hashlib.sha256(library.read_bytes()).hexdigest()
        if actual != HOST_SHA[arm]:
            raise RuntimeError(f'{arm} package host hash changed: {actual}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--slot', type=int, required=True, choices=(0, 1, 2))
    args = parser.parse_args()
    devices = (4, 6, 7)
    if args.device != devices[args.slot]:
        parser.error('slot/device mapping must be 0/4, 1/6, or 2/7')
    shapes = json.loads((HERE / 'dataset.json').read_text())
    if len(shapes) != 100 or len({tuple(x) for x in shapes}) != 100:
        raise RuntimeError('dataset must contain 100 distinct shapes')
    check_packages()
    assigned = [(i, shape) for i, shape in enumerate(shapes) if i % 3 == args.slot]
    for position, (index, shape) in enumerate(assigned):
        base = OUT / f'device{args.device}' / f'{index:03d}_{label(shape)}'
        record_path = base / 'record.json'
        if record_path.exists():
            print(f'skip {index:03d} {label(shape)}', flush=True)
            continue
        print(f'start {index:03d} {label(shape)} device={args.device}', flush=True)
        try:
            audits = {arm: audit(index, shape, args.device, arm)
                      for arm in ('master', 'online')}
            a = audits['master']['tiling']
            b = audits['online']['tiling']
            changed = any(a.get(field) != b.get(field) for field in TILE_FIELDS)
            mode5 = a['tilingEnableSplitCore'] == 5 or b['tilingEnableSplitCore'] == 5
            rounds = 2 if changed or mode5 else 1
            measured = []
            for round_index in range(rounds):
                order = ('master', 'online') if (position + round_index) % 2 == 0 else ('online', 'master')
                for arm in order:
                    result = timing(index, shape, args.device, arm, round_index, audits[arm])
                    measured.append(result)
                    print(f"  r{round_index} {arm} mode={result['tiling']['tilingEnableSplitCore']} "
                          f"{result['kernel_median_us']:.3f} us", flush=True)
            write(record_path, {'index': index, 'shape': shape, 'device': args.device,
                                'changed_tiling': changed, 'mode5_control': mode5,
                                'audits': audits, 'measurements': measured})
            (base / 'failure.json').unlink(missing_ok=True)
            print(f'done {index:03d} {label(shape)}', flush=True)
        except Exception as exc:
            write(base / 'failure.json', {'index': index, 'shape': shape,
                                          'device': args.device, 'error': str(exc),
                                          'traceback': traceback.format_exc()})
            print(f'FAILED {index:03d} {label(shape)}: {exc}', flush=True)
            # A busy device is no longer an isolated experiment; stop this worker.
            if 'occupied' in str(exc):
                raise
    print(f'worker device {args.device} finished {len(assigned)} assigned shapes', flush=True)


if __name__ == '__main__':
    main()
