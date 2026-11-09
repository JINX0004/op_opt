#!/usr/bin/env python3
"""Compare clean master with source-emitted FP32 model-family paths on 910."""

import argparse
import csv
import hashlib
import json
import subprocess
import traceback
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'cases/fp32_100_source_family_v1'
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
SELECTOR = Path('/home/xianfei/dev/ops-nn-sribd-mip-selector/experimental/mip_selector/build/mmopt_selector')
MASTER = ROOT / 'ops_nn_master_bf16_proto_package'
RESEARCH = ROOT / 'ops_nn_fp32_forced_family_package'
PACKAGES = {'master': MASTER, 'research_default': RESEARCH, 'proposed': RESEARCH}
SOURCES = {
    'master': '/home/xianfei/dev/ops-nn-sribd-master-bf16/matmul/mat_mul_v3/',
    'research_default': str(ROOT / 'ops_nn_source_build/matmul/mat_mul_v3/'),
    'proposed': str(ROOT / 'ops_nn_source_build/matmul/mat_mul_v3/'),
}
HASHES = {
    'master': 'b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad',
    'research_default': 'cd074481e4819bffc4e8d85d1d42e9c889d670dc3ecb213264cecb83ebbe5ad3',
    'proposed': 'cd074481e4819bffc4e8d85d1d42e9c889d670dc3ecb213264cecb83ebbe5ad3',
}
SELECTOR_HASH = 'd388c1000441711ea20d16eaba5528f461e8c78069590ef4ab1b9450b613927a'
TILE_FIELDS = ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN', 'singleCoreK',
               'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb',
               'depthA1', 'depthB1', 'block_dim', 'workspace_bytes',
               'nd2nzA', 'nd2nzB', 'tiling_key')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def label(shape):
    return 'x'.join(map(str, shape))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_binaries():
    if sha(SELECTOR) != SELECTOR_HASH:
        raise RuntimeError('selector binary changed during experiment')
    for arm, package in PACKAGES.items():
        library = package / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
        if sha(library) != HASHES[arm]:
            raise RuntimeError(f'{arm} package host library changed')


def source_cann_env():
    proc = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                           'bash', str(CANN)], capture_output=True, check=True)
    return dict(item.decode().split('=', 1) for item in proc.stdout.split(b'\0') if item)


CANN_ENV = source_cann_env()


def environment(arm, model_mode, log_level):
    package = PACKAGES[arm]
    env = CANN_ENV.copy()
    for key in ('MMV3_RESEARCH_SPLITK_FAMILY', 'MMV3_ONLINE_SELECTOR',
                'MMV3_ONLINE_BF16_EXPERIMENT', 'MMV3_ONLINE_TILE',
                'MMV3_ONLINE_ROW_READ_PROFILE'):
        env.pop(key, None)
    env.update(BENCH_CUSTOM_OPP_PATH=str(package), ASCEND_CUSTOM_OPP_PATH=str(package),
               MMV3_SKIP_LEGACY_OP_CACHE='1', IGNORE_INFER_ERROR='1',
               ASCEND_GLOBAL_LOG_LEVEL=str(log_level))
    env['LD_LIBRARY_PATH'] = str(package / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    if arm == 'proposed':
        env['MMV3_RESEARCH_SPLITK_FAMILY'] = {0: 'base', 3: 'deterministic'}[model_mode]
    return env


def require_idle(device, directory, phase):
    info = subprocess.run(['npu-smi', 'info'], capture_output=True,
                          text=True, check=True).stdout
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'npu_smi_{phase}.txt').write_text(info)
    if f'No running processes found in NPU {device}' not in info:
        raise RuntimeError(f'device {device} occupied at {phase}')


def harness_event(path):
    for line in reversed(path.read_text().splitlines()):
        if not line.startswith('{'):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get('kind') == 'matmul_v3_event':
            return data
    raise RuntimeError(f'no benchmark event in {path}')


def make_plan(shapes):
    path = HERE / 'model_decisions.json'
    if path.exists():
        return json.loads(path.read_text())
    request = ''.join(' '.join(map(str, shape)) + ' fp32\n' for shape in shapes)
    result = subprocess.run([str(SELECTOR)], input=request, text=True,
                            capture_output=True, check=True)
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if len(rows) != 100:
        raise RuntimeError(f'expected 100 model decisions, got {len(rows)}')
    decisions = []
    for shape, row in zip(shapes, rows):
        if [int(row[x]) for x in ('m', 'n', 'k')] != shape:
            raise RuntimeError('selector output shape/order mismatch')
        mode = int(row['mode'])
        if mode not in (0, 3):
            raise RuntimeError(f'unsupported FP32 proposed family {mode}')
        decisions.append({key: (row[key] if key == 'dtype' else float(row[key])
                                 if key == 'objective_us' else int(row[key]))
                          for key in row})
    write(path, decisions)
    return decisions


def audit(index, shape, device, arm, model_mode):
    directory = OUT / f'device{device}' / f'{index:03d}_{label(shape)}' / 'audit' / arm
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    require_idle(device, directory, 'before')
    env = environment(arm, model_mode, 1)
    cmd = [str(ROOT / 'fp32_bench'), '--api', 'v3cached', '--device', str(device),
           '--m', str(shape[0]), '--n', str(shape[1]), '--k', str(shape[2]),
           '--warmup', '0', '--batches', '1', '--repetitions', '1']
    with (directory / 'stdout.jsonl').open('w') as stdout, \
         (directory / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=900)
    require_idle(device, directory, 'after')
    if proc.returncode:
        raise RuntimeError(f'audit {arm} {shape} exit {proc.returncode}: '
                           + (directory / 'stderr.log').read_text()[-1200:])
    event = harness_event(directory / 'stdout.jsonl')
    parsed = subprocess.run(['python3', str(ROOT / 'parse_tiling_log.py'),
                             '--pid', str(event['pid'])], env=env,
                            capture_output=True, text=True, check=True)
    tiling = json.loads(parsed.stdout)
    if (event['device'] != device or event['dtype'] != 'fp32' or
            event['max_relative_error'] > .02 or
            not tiling.get('tiling_source', '').startswith(SOURCES[arm])):
        raise RuntimeError(f'audit provenance/correctness failure: {arm} {shape}')
    if arm == 'proposed' and tiling['tilingEnableSplitCore'] != model_mode:
        raise RuntimeError(f'forced family failed: expected {model_mode}, got '
                           f'{tiling["tilingEnableSplitCore"]}')
    data = {'index': index, 'shape': shape, 'device': device, 'arm': arm,
            'event': event, 'tiling': tiling}
    write(result_path, data)
    return data


def profile(index, shape, device, arm, model_mode, round_index, audited):
    directory = OUT / f'device{device}' / f'{index:03d}_{label(shape)}' / f'round{round_index}' / arm
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    require_idle(device, directory, 'before')
    env = environment(arm, model_mode, 3)
    mode = audited['tiling']['tilingEnableSplitCore']
    kind = ('KERNEL_AICORE' if mode == 0 and
            not audited['tiling']['nd2nzA'] and not audited['tiling']['nd2nzB']
            else 'MIX_AIC')
    cmd = ['msprof', '--output=' + str(directory / 'msprof'), '--task-time=on',
           '--ai-core=on', 'env', str(ROOT / 'fp32_bench'), '--api', 'v3cached',
           '--device', str(device), '--m', str(shape[0]), '--n', str(shape[1]),
           '--k', str(shape[2]), '--warmup', '5', '--batches', '1', '--repetitions', '5']
    with (directory / 'msprof_stdout.log').open('w') as stdout, \
         (directory / 'msprof_stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=1200)
    require_idle(device, directory, 'after')
    if proc.returncode:
        raise RuntimeError(f'profile {arm} {shape} exit {proc.returncode}: '
                           + (directory / 'msprof_stderr.log').read_text()[-1200:])
    event = harness_event(directory / 'msprof_stdout.log')
    if (event['device'] != device or event['dtype'] != 'fp32' or
            event['max_relative_error'] > .02 or event['repetitions'] != 5):
        raise RuntimeError(f'profile output failure: {arm} {shape}: {event}')
    parsed = subprocess.run(['python3', str(ROOT / 'bf16_online_vs_master/parse_profile.py'),
                             str(directory / 'msprof'), '--last', '5',
                             '--kernel-type', kind], capture_output=True, text=True, check=True)
    timing = json.loads(parsed.stdout)
    if timing['kernel_count'] != 5:
        raise RuntimeError(f'expected 5 kernel tasks: {arm} {shape}')
    data = {'index': index, 'shape': shape, 'device': device, 'arm': arm,
            'round': round_index, 'mode': mode, 'event': event, 'profile': timing}
    write(result_path, data)
    return data


def worker(shapes, decisions, device, max_cases=None):
    assigned = [(i, shape) for i, shape in enumerate(shapes) if i % 4 == device - 4]
    if max_cases is not None:
        assigned = assigned[:max_cases]
    for position, (index, shape) in enumerate(assigned):
        base = OUT / f'device{device}' / f'{index:03d}_{label(shape)}'
        record_path = base / 'record.json'
        if record_path.exists():
            print(f'skip {index:03d} {label(shape)}', flush=True)
            continue
        model = decisions[index]
        print(f'start {index:03d} {label(shape)} device={device} model={model["mode"]}', flush=True)
        try:
            audits = {arm: audit(index, shape, device, arm, model['mode'])
                      for arm in ('master', 'research_default', 'proposed')}
            master = audits['master']['tiling']
            research = audits['research_default']['tiling']
            changed_default = {field: (master.get(field), research.get(field))
                               for field in TILE_FIELDS if master.get(field) != research.get(field)}
            if changed_default:
                raise RuntimeError(f'research default differs from master: {changed_default}')
            proposed = audits['proposed']['tiling']
            changed = any(master.get(field) != proposed.get(field) for field in TILE_FIELDS)
            rounds = 2 if changed else 1
            measured = []
            for round_index in range(rounds):
                order = ('master', 'proposed') if (position + round_index) % 2 == 0 \
                    else ('proposed', 'master')
                for arm in order:
                    item = profile(index, shape, device, arm, model['mode'],
                                   round_index, audits[arm])
                    measured.append(item)
                    print(f'  r{round_index} {arm} mode={item["mode"]} '
                          f'{item["profile"]["median_us"]:.3f} us', flush=True)
            write(record_path, {'index': index, 'shape': shape, 'device': device,
                                'model': model, 'changed_tiling': changed,
                                'audits': audits, 'measurements': measured})
            (base / 'failure.json').unlink(missing_ok=True)
            print(f'done {index:03d} {label(shape)}', flush=True)
        except Exception as error:
            write(base / 'failure.json', {'index': index, 'shape': shape,
                                          'device': device, 'error': str(error),
                                          'traceback': traceback.format_exc()})
            print(f'FAILED {index:03d} {label(shape)}: {error}', flush=True)
            if 'occupied' in str(error):
                raise
    print(f'worker device {device} finished {len(assigned)} shapes', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('plan', 'run'))
    parser.add_argument('--device', type=int, choices=(4, 5, 6, 7))
    parser.add_argument('--max-cases', type=int)
    args = parser.parse_args()
    shapes = json.loads((HERE / 'dataset.json').read_text())
    if len(shapes) != 100 or len({tuple(x) for x in shapes}) != 100:
        raise RuntimeError('dataset must have 100 unique shapes')
    check_binaries()
    decisions = make_plan(shapes)
    if args.phase == 'plan':
        print(f'model decisions: {len(decisions)}', flush=True)
        return
    if args.device is None:
        parser.error('--device is required for run')
    worker(shapes, decisions, args.device, args.max_cases)


if __name__ == '__main__':
    main()
