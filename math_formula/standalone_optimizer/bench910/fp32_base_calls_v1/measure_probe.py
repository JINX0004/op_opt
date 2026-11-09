#!/usr/bin/env python3
"""Measure isolated FP32 GM-to-L1 A/B packet service on device 4."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/fp32_base_calls_v1')
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
PROBE = ROOT / 'probe/build/l1_step_read_probe'


def main():
    raw = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                          'bash', str(CANN)], capture_output=True, check=True).stdout
    env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
    records = []
    for operand in ('A', 'B'):
        for axis, span in ((16, 512), (16, 1024), (16, 2048),
                           (32, 512), (32, 1024), (32, 2048),
                           (64, 512), (64, 1024)):
            for copies in (1, 32):
                command = [str(PROBE), operand, str(axis), str(span),
                           str(axis), str(axis), '65536', str(copies),
                           '1', '5', '100']
                result = subprocess.run(command, cwd=ROOT, env=env,
                                        text=True, capture_output=True, check=True)
                records.append(json.loads(result.stdout))
                print(json.dumps(records[-1]), flush=True)
    (ROOT / 'probe_results.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
