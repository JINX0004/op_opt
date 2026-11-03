#!/usr/bin/env python3
"""Audit and time source-backed MatMulV3 Split-K family choices on 910B3."""

import argparse
import json
import os
import random
import subprocess
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
OUT = ROOT / 'cases/splitk_family_selection_v1'
DEVICE = 0
PACKAGE = ROOT / 'ops_nn_splitk_family_package'
SOURCE = str(ROOT / 'ops_nn_source_build/matmul/mat_mul_v3/')
SHAPES = (
    (1792, 768, 27392), (2048, 768, 27392),
    (1792, 768, 32768), (2048, 768, 32768),
    (2048, 2048, 27392), (2048, 3072, 27392),
    (2048, 2048, 32768), (2048, 3072, 32768),
)
GEOMETRY_FIELDS = ('singleCoreM', 'singleCoreN', 'singleCoreK',
                   'baseM', 'baseN', 'baseK', 'stepM', 'stepN',
                   'stepKa', 'stepKb', 'depthA1', 'depthB1',
                   'tiling_key', 'block_dim', 'workspace_bytes',
                   'tilingEnableSplitCore', 'nd2nzA', 'nd2nzB')


def environment(arm: str, shape: tuple[int, int, int]) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        'MMV3_SKIP_LEGACY_OP_CACHE': '1',
        'BENCH_API': 'v3cached',
        'BENCH_CUSTOM_OPP_PATH': str(PACKAGE),
        'BENCH_EXPECT_SOURCE': SOURCE,
        'BENCH_EXPECT_ND2NZ_A': str(int(shape[2] > 65535)),
        'BENCH_EXPECT_ND2NZ_B': '0',
        'BENCH_PROFILE_KERNEL_TYPE': 'MIX_AIC',
        'CANN_ROOT': '/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0',
        'ASCEND_CUSTOM_OPP_PATH': str(PACKAGE),
    })
    opapi = PACKAGE / 'op_api/lib'
    env['LD_LIBRARY_PATH'] = f"{opapi}:{env.get('LD_LIBRARY_PATH', '')}"
    env.pop('MMV3_RESEARCH_SPLITK_FAMILY', None)
    if arm != 'default':
        env['MMV3_RESEARCH_SPLITK_FAMILY'] = arm
    return env


def label(shape: tuple[int, int, int]) -> str:
    return 'x'.join(map(str, shape))


def require_idle(directory: Path, phase: str) -> None:
    snapshot = subprocess.run(['npu-smi', 'info'], text=True,
                              capture_output=True, check=True).stdout
    (directory / f'npu_smi_{phase}.txt').write_text(snapshot)
    if f'No running processes found in NPU {DEVICE}' not in snapshot:
        (directory / 'interference.flag').write_text(
            f'Device {DEVICE} was occupied at {phase}\n')
        raise RuntimeError(f'device {DEVICE} is occupied at {phase}: {directory}')


def audit_one(arm: str, shape: tuple[int, int, int]) -> dict:
    directory = OUT / 'audit' / label(shape) / arm
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    env = environment(arm, shape)
    env.update({'ASCEND_GLOBAL_LOG_LEVEL': '1', 'IGNORE_INFER_ERROR': '1'})
    cmd = [str(ROOT / 'base_aclnn_bench'), '--api', 'v3cached', '--device', str(DEVICE),
           '--m', str(shape[0]), '--n', str(shape[1]), '--k', str(shape[2]),
           '--warmup', '0', '--batches', '1', '--repetitions', '1']
    with (directory / 'stdout.jsonl').open('w') as stdout, \
         (directory / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr)
    result = {'arm': arm, 'shape': list(shape), 'returncode': proc.returncode}
    try:
        event = json.loads((directory / 'stdout.jsonl').read_text().splitlines()[-1])
        result['event'] = event
        parsed = subprocess.run(['python3', str(ROOT / 'parse_tiling_log.py'),
                                 '--pid', str(event['pid'])], text=True,
                                capture_output=True)
        if parsed.returncode == 0:
            result['tiling'] = json.loads(parsed.stdout)
        else:
            result['parse_error'] = parsed.stderr.strip()
    except (ValueError, IndexError, KeyError) as exc:
        result['parse_error'] = str(exc)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('audit', label(shape), arm, 'mode',
          result.get('tiling', {}).get('tilingEnableSplitCore'),
          'rc', proc.returncode, flush=True)
    return result


def eligible(shape: tuple[int, int, int]) -> bool:
    s = audit_one('single', shape)
    d = audit_one('deterministic', shape)
    default = audit_one('default', shape)
    modes = [r.get('tiling', {}).get('tilingEnableSplitCore')
             for r in (s, d, default)]
    return (s['returncode'] == d['returncode'] == default['returncode'] == 0
            and modes[0] in (2, 5, 6) and modes[1] == 3
            and all(r['event']['device'] == DEVICE for r in (s, d, default))
            and all(r['event']['max_relative_error'] <= 0.02
                    for r in (s, d, default)))


def time_one(arm: str, shape: tuple[int, int, int], round_index: int,
             repetitions: int) -> dict:
    directory = OUT / 'timing' / f'r{round_index}' / label(shape) / arm
    row_path = directory / 'case.json'
    if row_path.exists():
        return json.loads(row_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    audit = audit_one(arm, shape)
    mode = audit['tiling']['tilingEnableSplitCore']
    env = environment(arm, shape)
    env['BENCH_EXPECT_SPLIT_CORE_MODE'] = str(mode)
    require_idle(directory, 'before')
    with (directory / 'run.log').open('w') as log:
        subprocess.run(['bash', str(ROOT / 'run_case.sh'), *map(str, shape),
                        str(directory), str(DEVICE), '10', str(repetitions)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    require_idle(directory, 'after')
    (directory / 'interference.flag').unlink(missing_ok=True)
    case = json.loads(row_path.read_text())
    if (case['device'] != DEVICE or case['split_core_mode'] != mode or
            case['kernel_count'] != repetitions):
        raise RuntimeError(f'profile/mode mismatch: {arm} {shape}')
    for field in ('singleCoreM', 'singleCoreN', 'singleCoreK', 'baseM',
                  'baseN', 'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb',
                  'tiling_key', 'nd2nzA', 'nd2nzB'):
        if json.loads((directory / 'tiling.json').read_text()).get(field) != \
                audit['tiling'].get(field):
            raise RuntimeError(f'geometry changed: {field} {arm} {shape}')
    print('timing', round_index, label(shape), arm,
          f"{case['kernel_median_us']:.3f} us", flush=True)
    return case


def main() -> None:
    global OUT, DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('audit', 'timing'))
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--repetitions', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--experiment-dir', type=Path, default=OUT)
    args = parser.parse_args()
    DEVICE = args.device
    OUT = args.experiment_dir
    OUT.mkdir(parents=True, exist_ok=True)
    valid = [shape for shape in SHAPES if eligible(shape)]
    (OUT / 'eligible_shapes.json').write_text(json.dumps(valid, indent=2) + '\n')
    print(f'eligible {len(valid)} / {len(SHAPES)}', flush=True)
    if args.phase == 'audit':
        return
    rng = random.Random(20261030)
    for round_index in range(args.rounds):
        tasks = []
        for shape in valid:
            default = audit_one('default', shape)['tiling']
            same_arm = ('single' if default['tilingEnableSplitCore'] in (2, 5, 6)
                        else 'deterministic' if default['tilingEnableSplitCore'] == 3
                        else None)
            if same_arm is None:
                tasks.extend((shape, arm) for arm in
                             ('default', 'single', 'deterministic'))
                continue
            matching = audit_one(same_arm, shape)['tiling']
            if any(default.get(field) != matching.get(field)
                   for field in GEOMETRY_FIELDS):
                raise RuntimeError(f'default and {same_arm} tilings differ: {shape}')
            opposite = ('deterministic' if same_arm == 'single' else 'single')
            tasks.extend(((shape, 'default'), (shape, opposite)))
        rng.shuffle(tasks)
        for shape, arm in tasks:
            time_one(arm, shape, round_index, args.repetitions)


if __name__ == '__main__':
    main()
