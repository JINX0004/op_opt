#!/usr/bin/env python3
"""Probe source-shaped FP32 GM-to-L1 A/B packets near K stride spikes."""

import json
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/fp32_base_calls_v1')
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
PROBE = ROOT / 'probe/build/l1_step_read_probe'
OUT = Path('/home/xianfei/solvebase_bench_20261025/cases/fp32_base_formula_v2/stride_probe.json')


def main():
    raw = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                          'bash', str(CANN)], capture_output=True, check=True).stdout
    env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
    records = []
    for m, n, ks in ((2560, 768, (32512, 32768, 33024)),
                     (2560, 1024, (48896, 49152, 49408))):
        for operand, axis, span in (('A', 128, 256), ('B', 256, 128)):
            for k in ks:
                for copies in (1, 128):
                    command = [str(PROBE), operand, str(axis), str(span),
                               str(m), str(n), str(k), str(copies),
                               '20', '5', '100']
                    result = subprocess.run(command, cwd=ROOT, env=env,
                                            text=True, capture_output=True, check=True)
                    record = json.loads(result.stdout)
                    records.append(record)
                    print(record, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
