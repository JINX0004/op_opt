#!/usr/bin/env python3
"""Measure kernel time for paired FP32 source-family boundary cases."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
OUT = ROOT / 'cases/fp32_family_boundary_v1'
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
PACKAGE = ROOT / 'ops_nn_master_bf16_proto_package'
DEVICE = 4
SELECTED = {
    (16, 16, 6080), (16, 16, 6144),
    (64, 64, 6080), (64, 64, 6144),
    (128, 128, 7616), (128, 128, 7680),
    (256, 256, 7616), (256, 256, 7680),
    (128, 128, 27328), (128, 128, 27392),
    (256, 256, 27328), (256, 256, 27392),
    (512, 512, 27328), (512, 512, 27392),
    (2048, 64, 27328), (2048, 64, 27392),
}


def require_idle():
    info = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True,
                          check=True).stdout
    if f'No running processes found in NPU {DEVICE}' not in info:
        raise RuntimeError(f'NPU {DEVICE} is occupied')


def main():
    raw = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                          'bash', str(CANN)], capture_output=True, check=True).stdout
    env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
    for key in ('MMV3_ONLINE_SELECTOR', 'MMV3_ONLINE_BF16_EXPERIMENT',
                'MMV3_ONLINE_FP32_EXPERIMENT', 'MMV3_ONLINE_TILE'):
        env.pop(key, None)
    env.update(BENCH_CUSTOM_OPP_PATH=str(PACKAGE), ASCEND_CUSTOM_OPP_PATH=str(PACKAGE),
               MMV3_SKIP_LEGACY_OP_CACHE='1', ASCEND_GLOBAL_LOG_LEVEL='3')
    env['LD_LIBRARY_PATH'] = str(PACKAGE / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    for case in sorted(OUT.glob('[0-9][0-9]_*')):
        audited = case / 'result.json'
        if not audited.exists():
            continue
        data = json.loads(audited.read_text())
        shape = tuple(data['shape'])
        if shape not in SELECTED or (case / 'profile.json').exists():
            continue
        require_idle()
        mode = data['tiling']['tilingEnableSplitCore']
        nd_a = data['tiling']['nd2nzA']
        nd_b = data['tiling']['nd2nzB']
        kind = 'KERNEL_AICORE' if mode == 0 and nd_a == 0 and nd_b == 0 else 'MIX_AIC'
        path = case / 'msprof'
        print(f'profile {shape} mode={mode} kind={kind}', flush=True)
        cmd = ['msprof', '--output=' + str(path), '--task-time=on', '--ai-core=on',
               'env', str(ROOT / 'fp32_bench'), '--api', 'v3cached',
               '--device', str(DEVICE), '--m', str(shape[0]), '--n', str(shape[1]),
               '--k', str(shape[2]), '--warmup', '5', '--batches', '1',
               '--repetitions', '5']
        with (case / 'msprof_stdout.log').open('w') as stdout, \
             (case / 'msprof_stderr.log').open('w') as stderr:
            proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                                  stderr=stderr, timeout=900)
        require_idle()
        if proc.returncode:
            raise RuntimeError(f'{shape}: msprof exit {proc.returncode}: '
                               + (case / 'msprof_stderr.log').read_text()[-1200:])
        parsed = subprocess.run(['python3', str(ROOT / 'bf16_online_vs_master/parse_profile.py'),
                                 str(path), '--last', '5', '--kernel-type', kind],
                                capture_output=True, text=True, check=True)
        profile = json.loads(parsed.stdout)
        (case / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
        print(f"  kernel_median_us={profile['median_us']}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise
