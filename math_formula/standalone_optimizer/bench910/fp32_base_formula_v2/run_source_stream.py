#!/usr/bin/env python3
"""Measure one full source-sized output-tile A/B stream per AIC on device 4."""

import json
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/fp32_base_formula_v2/source_stream_probe')
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
PROBE = ROOT / 'build/base_stream_probe'
OUT = Path('/home/xianfei/solvebase_bench_20261025/cases/fp32_base_formula_v2/source_stream.json')


def main():
    raw = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                          'bash', str(CANN)], capture_output=True, check=True).stdout
    env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    for m, n, ks in ((2560, 768, (32512, 32768, 33024)),
                     (2560, 1024, (48896, 49152, 49408))):
        for operand, span in (('A', 256), ('B', 128)):
            for k in ks:
                for repeats in (0, 1):
                    cmd = [str(PROBE), operand, '128', '256', str(span),
                           str(m), str(n), str(k), str(repeats),
                           '20', '3', '50']
                    if any(row['m'] == m and row['n'] == n and row['k'] == k and
                           row['operand'] == operand and row['repeats'] == repeats
                           for row in rows):
                        continue
                    for attempt in range(3):
                        result = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                                                capture_output=True)
                        if result.returncode == 0:
                            break
                        print('retry', attempt + 1, cmd, result.stderr[-700:], flush=True)
                    result.check_returncode()
                    row = json.loads(result.stdout)
                    rows.append(row)
                    print(row, flush=True)
                    OUT.parent.mkdir(parents=True, exist_ok=True)
                    OUT.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
