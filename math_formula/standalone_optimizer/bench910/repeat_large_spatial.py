#!/usr/bin/env python3
"""Six randomized source-checked large BASE tile comparisons on 910B3."""
import json
import os
import random
import subprocess
import time
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SHAPES = [(2048, 2048, 4096),
          (4096, 4096, 8192),
          (8192, 8192, 4096)]


def main():
    base = ROOT / 'cases/optimizer_vs_default_large/paired_m64'
    base.mkdir(parents=True, exist_ok=True)
    output = base / 'raw.jsonl'
    done = {(r['round'], tuple(r['shape']), r['arm'])
            for r in map(json.loads, output.read_text().splitlines())} if output.exists() else set()
    env_base = os.environ.copy()
    env_base.update({
        'MMV3_SKIP_LEGACY_OP_CACHE': '1',
        'BENCH_API': 'v3cached',
        'BENCH_CUSTOM_OPP_PATH': str(ROOT / 'ops_nn_spatial_package'),
        'BENCH_EXPECT_SOURCE': str(ROOT / 'ops_nn_source_build/matmul/mat_mul_v3/'),
        'CANN_ROOT': '/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0',
    })
    rng = random.Random(91020261025)
    for round_index in range(6):
        order = [(shape, arm) for shape in SHAPES
                 for arm in ('default', 'base_m_64')]
        rng.shuffle(order)
        for shape, arm in order:
            if (round_index, shape, arm) in done:
                continue
            label = 'x'.join(map(str, shape))
            directory = base / f'r{round_index}_{label}_{arm}'
            env = env_base.copy()
            env.pop('MMV3_FORCE_BASE_M', None)
            if arm == 'base_m_64':
                env['MMV3_FORCE_BASE_M'] = '64'
            if not (directory / 'case.json').exists():
                subprocess.run(['bash', str(ROOT / 'run_case.sh'),
                                *map(str, shape), str(directory), '0', '10', '8'],
                               env=env, check=True, stdout=subprocess.DEVNULL)
            case = json.loads((directory / 'case.json').read_text())
            tiling = json.loads((directory / 'tiling.json').read_text())
            if (case['shape'] != list(shape) or case['max_abs_error'] != 0 or
                    case['baseM'] != (128 if arm == 'default' else 64) or
                    case['singleCoreM'] != case['baseM'] or
                    case['block_dim'] != 20 or
                    tiling['family'] != 'BASE' or tiling['nd2nzA'] != 0 or
                    tiling['nd2nzB'] != 0 or case['tiling_key'] != 65536):
                raise RuntimeError(f'failed checked tiling or output: {directory}')
            row = {'round': round_index, 'shape': shape, 'arm': arm,
                   'unix_time': time.time(), 'case': case}
            with output.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            print(f'round {round_index+1}/6 {label} {arm} '
                  f'{case["kernel_median_us"]:.3f} us', flush=True)


if __name__ == '__main__':
    main()
