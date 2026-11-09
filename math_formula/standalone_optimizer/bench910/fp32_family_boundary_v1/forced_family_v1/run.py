#!/usr/bin/env python3
"""Audit and profile source-emitted FP32 BASE versus deterministic Split-K."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'cases/fp32_forced_family_v1'
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
PACKAGE = ROOT / 'ops_nn_fp32_forced_family_package'
HOST_HASH = 'cd074481e4819bffc4e8d85d1d42e9c889d670dc3ecb213264cecb83ebbe5ad3'
SOURCE = str(ROOT / 'ops_nn_source_build/matmul/mat_mul_v3/')
MASTER_AUDIT = ROOT / 'cases/fp32_family_boundary_v1'
DEVICE = 4
FIELDS = ('tilingEnableSplitCore', 'singleCoreM', 'singleCoreN', 'singleCoreK',
          'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb',
          'depthA1', 'depthB1', 'block_dim', 'workspace_bytes',
          'nd2nzA', 'nd2nzB', 'tiling_key')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def label(shape):
    return 'x'.join(map(str, shape))


def harness_event(path):
    """Find the harness record among msprof's own stdout messages."""
    for line in reversed(path.read_text().splitlines()):
        if not line.startswith('{'):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get('kind') == 'matmul_v3_event':
            return record
    raise RuntimeError(f'no harness event in {path}')


def require_idle(path, phase):
    info = subprocess.run(['npu-smi', 'info'], capture_output=True,
                          text=True, check=True).stdout
    path.mkdir(parents=True, exist_ok=True)
    (path / f'npu_smi_{phase}.txt').write_text(info)
    if f'No running processes found in NPU {DEVICE}' not in info:
        raise RuntimeError(f'device {DEVICE} occupied at {phase}')


def environment(arm, logging):
    raw = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                          'bash', str(CANN)], capture_output=True, check=True).stdout
    env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
    for key in ('MMV3_RESEARCH_SPLITK_FAMILY', 'MMV3_ONLINE_SELECTOR',
                'MMV3_ONLINE_BF16_EXPERIMENT', 'MMV3_ONLINE_TILE'):
        env.pop(key, None)
    env.update(BENCH_CUSTOM_OPP_PATH=str(PACKAGE), ASCEND_CUSTOM_OPP_PATH=str(PACKAGE),
               MMV3_SKIP_LEGACY_OP_CACHE='1', IGNORE_INFER_ERROR='1',
               ASCEND_GLOBAL_LOG_LEVEL=str(logging))
    env['LD_LIBRARY_PATH'] = str(PACKAGE / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    if arm == 'base':
        env['MMV3_RESEARCH_SPLITK_FAMILY'] = 'base'
    elif arm == 'split':
        env['MMV3_RESEARCH_SPLITK_FAMILY'] = 'deterministic'
    elif arm != 'default':
        raise ValueError(arm)
    return env


def check_package():
    library = PACKAGE / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
    actual = hashlib.sha256(library.read_bytes()).hexdigest()
    if actual != HOST_HASH:
        raise RuntimeError(f'research package hash changed: {actual}')


def master_tiling(shape):
    matches = list(MASTER_AUDIT.glob(f'[0-9][0-9]_{label(shape)}/result.json'))
    if len(matches) != 1:
        raise RuntimeError(f'expected one master audit for {shape}, got {matches}')
    return json.loads(matches[0].read_text())['tiling']


def audit(shape, arm):
    path = OUT / label(shape) / 'audit' / arm
    result_path = path / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    require_idle(path, 'before')
    cmd = [str(ROOT / 'fp32_bench'), '--api', 'v3cached', '--device', str(DEVICE),
           '--m', str(shape[0]), '--n', str(shape[1]), '--k', str(shape[2]),
           '--warmup', '1', '--batches', '1', '--repetitions', '3']
    env = environment(arm, 1)
    print(f'audit {label(shape)} {arm}', flush=True)
    try:
        with (path / 'stdout.jsonl').open('w') as stdout, \
             (path / 'stderr.log').open('w') as stderr:
            proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                                  stderr=stderr, timeout=600)
        require_idle(path, 'after')
        if proc.returncode:
            raise RuntimeError(f'harness exit {proc.returncode}: '
                               + (path / 'stderr.log').read_text()[-1200:])
        event = json.loads((path / 'stdout.jsonl').read_text().splitlines()[-1])
        parsed = subprocess.run(['python3', str(ROOT / 'parse_tiling_log.py'),
                                 '--pid', str(event['pid'])], env=env,
                                capture_output=True, text=True, check=True)
        tiling = json.loads(parsed.stdout)
        if (event['dtype'] != 'fp32' or event['device'] != DEVICE or
                event['max_relative_error'] > 0.02 or
                not tiling.get('tiling_source', '').startswith(SOURCE)):
            raise RuntimeError(f'provenance/correctness failure: {event} {tiling}')
        expected_mode = {'base': 0, 'split': 3}.get(arm)
        if expected_mode is not None and tiling['tilingEnableSplitCore'] != expected_mode:
            raise RuntimeError(f'forced {arm} emitted mode {tiling["tilingEnableSplitCore"]}')
        if arm == 'default':
            baseline = master_tiling(shape)
            changed = {field: (baseline.get(field), tiling.get(field)) for field in FIELDS
                       if baseline.get(field) != tiling.get(field)}
            if changed:
                raise RuntimeError(f'research default differs from master: {changed}')
        result = {'shape': shape, 'arm': arm, 'event': event, 'tiling': tiling}
        write(result_path, result)
        (path / 'failure.json').unlink(missing_ok=True)
        print(f"  mode={tiling['tilingEnableSplitCore']} "
              f"core={tiling['singleCoreM']}x{tiling['singleCoreN']}x{tiling['singleCoreK']} "
              f"base={tiling['baseM']}x{tiling['baseN']}x{tiling['baseK']}", flush=True)
        return result
    except Exception as error:
        write(path / 'failure.json', {'shape': shape, 'arm': arm,
                                      'error': str(error), 'traceback': traceback.format_exc()})
        print(f'FAILED audit {label(shape)} {arm}: {error}', file=sys.stderr, flush=True)
        if 'occupied' in str(error):
            raise
        return None


def profile(shape, arm, round_index, audit_result):
    path = OUT / label(shape) / f'round{round_index}' / arm
    result_path = path / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    require_idle(path, 'before')
    env = environment(arm, 3)
    mode = audit_result['tiling']['tilingEnableSplitCore']
    nd_a = audit_result['tiling']['nd2nzA']
    nd_b = audit_result['tiling']['nd2nzB']
    kind = 'KERNEL_AICORE' if mode == 0 and not nd_a and not nd_b else 'MIX_AIC'
    cmd = ['msprof', '--output=' + str(path / 'msprof'), '--task-time=on',
           '--ai-core=on', 'env', str(ROOT / 'fp32_bench'), '--api', 'v3cached',
           '--device', str(DEVICE), '--m', str(shape[0]), '--n', str(shape[1]),
           '--k', str(shape[2]), '--warmup', '5', '--batches', '1',
           '--repetitions', '8']
    print(f'profile r{round_index} {label(shape)} {arm}', flush=True)
    with (path / 'msprof_stdout.log').open('w') as stdout, \
         (path / 'msprof_stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=900)
    require_idle(path, 'after')
    if proc.returncode:
        raise RuntimeError(f'{shape} {arm}: msprof exit {proc.returncode}: '
                           + (path / 'msprof_stderr.log').read_text()[-1200:])
    event = harness_event(path / 'msprof_stdout.log')
    if (event['dtype'] != 'fp32' or event['device'] != DEVICE or
            event['max_relative_error'] > 0.02):
        raise RuntimeError(f'timed run failed correctness: {shape} {arm}: {event}')
    parsed = subprocess.run(['python3', str(ROOT / 'bf16_online_vs_master/parse_profile.py'),
                             str(path / 'msprof'), '--last', '8', '--kernel-type', kind],
                            capture_output=True, text=True, check=True)
    result = {'shape': shape, 'arm': arm, 'round': round_index,
              'mode': mode, 'profile': json.loads(parsed.stdout), 'event': event}
    if result['profile']['kernel_count'] != 8:
        raise RuntimeError(f'expected 8 kernel tasks: {shape} {arm}')
    write(result_path, result)
    print(f"  median_us={result['profile']['median_us']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('audit', 'profile'))
    args = parser.parse_args()
    check_package()
    shapes = json.loads((HERE / 'dataset.json').read_text())
    eligible = []
    for shape in shapes:
        audits = {arm: audit(shape, arm) for arm in ('default', 'base', 'split')}
        if audits['base'] and audits['split'] and audits['default']:
            eligible.append((shape, audits))
    write(OUT / 'eligible.json', [shape for shape, _ in eligible])
    print(f'eligible {len(eligible)}/{len(shapes)}', flush=True)
    if args.phase == 'audit':
        return
    for round_index in range(2):
        tasks = [(shape, arm, audits[arm]) for shape, audits in eligible
                 for arm in ('base', 'split')]
        random.Random(20261106 + round_index).shuffle(tasks)
        for shape, arm, audit_result in tasks:
            profile(shape, arm, round_index, audit_result)


if __name__ == '__main__':
    main()
