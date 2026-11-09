#!/usr/bin/env python3
"""Audit clean-master FP32 MatMulV3 family and tiling near source thresholds."""

import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'cases/fp32_family_boundary_v1'
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
PACKAGE = ROOT / 'ops_nn_master_bf16_proto_package'
SOURCE = '/home/xianfei/dev/ops-nn-sribd-master-bf16/matmul/mat_mul_v3/'
DEVICE = 4


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def require_idle():
    output = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True,
                            check=True).stdout
    if f'No running processes found in NPU {DEVICE}' not in output:
        raise RuntimeError(f'NPU {DEVICE} is occupied')


def main():
    raw = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                          'bash', str(CANN)], capture_output=True, check=True).stdout
    env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
    for key in ('MMV3_ONLINE_SELECTOR', 'MMV3_ONLINE_BF16_EXPERIMENT',
                'MMV3_ONLINE_FP32_EXPERIMENT', 'MMV3_ONLINE_TILE'):
        env.pop(key, None)
    env.update(BENCH_CUSTOM_OPP_PATH=str(PACKAGE), ASCEND_CUSTOM_OPP_PATH=str(PACKAGE),
               MMV3_SKIP_LEGACY_OP_CACHE='1', ASCEND_GLOBAL_LOG_LEVEL='1')
    env['LD_LIBRARY_PATH'] = str(PACKAGE / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    dataset = json.loads((HERE / 'dataset.json').read_text())
    assert len({tuple(x['shape']) for x in dataset}) == len(dataset)
    for index, case in enumerate(dataset):
        shape = case['shape']
        label = 'x'.join(map(str, shape))
        path = OUT / f'{index:02d}_{label}'
        if (path / 'result.json').exists():
            print(f'skip {label}', flush=True)
            continue
        try:
            require_idle()
            path.mkdir(parents=True, exist_ok=True)
            cmd = [str(ROOT / 'fp32_bench'), '--api', 'v3cached', '--device', str(DEVICE),
                   '--m', str(shape[0]), '--n', str(shape[1]), '--k', str(shape[2]),
                   '--warmup', '1', '--batches', '1', '--repetitions', '3']
            print(f'start {index:02d} {label}', flush=True)
            with (path / 'stdout.jsonl').open('w') as stdout, \
                 (path / 'stderr.log').open('w') as stderr:
                proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                                      stderr=stderr, timeout=600)
            require_idle()
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
            write(path / 'result.json', {'index': index, 'group': case['group'],
                                         'shape': shape, 'event': event, 'tiling': tiling})
            print(f"  mode={tiling['tilingEnableSplitCore']} "
                  f"base={tiling['baseM']}x{tiling['baseN']}x{tiling['baseK']} "
                  f"core={tiling['singleCoreM']}x{tiling['singleCoreN']}x{tiling['singleCoreK']}",
                  flush=True)
        except Exception as exc:
            write(path / 'failure.json', {'index': index, 'shape': shape,
                                          'error': str(exc), 'traceback': traceback.format_exc()})
            print(f'FAILED {label}: {exc}', file=sys.stderr, flush=True)
            if 'occupied' in str(exc):
                raise


if __name__ == '__main__':
    main()
