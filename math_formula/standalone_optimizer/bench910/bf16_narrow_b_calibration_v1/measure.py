#!/usr/bin/env python3
"""Paired FP16/BF16 ND-to-L1 packet slopes on idle 910B3 device 4."""

import json
import random
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/bf16_narrow_b_calibration_v1')
ROOT.mkdir(parents=True, exist_ok=True)
BINARIES = {
    'fp16': '/tmp/mmv3mte2sweep/build/l1_step_read_probe',
    'bf16': '/tmp/mmv3bf16narrow/build/l1_step_read_probe',
}
K = 102400
ROWS = [('B', 16, 4096, 1), ('B', 16, 4096, 20),
        ('B', 32, 2048, 1), ('B', 32, 2048, 20),
        ('B', 128, 512, 1), ('A', 16, 4096, 1)]


def require_idle():
    snapshot = subprocess.run(['npu-smi', 'info'], text=True,
                              capture_output=True, check=True).stdout
    if 'No running processes found in NPU 4' not in snapshot:
        raise RuntimeError('device 4 occupied')


def main():
    output = ROOT / 'event_sweep.jsonl'
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row['round'], row['dtype'], row['operand'],
                          row['tile_axis'], row['k_span'], row['cores'], row['copies']))
    for round_index in range(3):
        cases = [(dtype, operand, axis, span, cores, copies)
                 for operand, axis, span, cores in ROWS
                 for dtype in BINARIES for copies in (1, 25)]
        random.Random(91020261104 + round_index).shuffle(cases)
        for dtype, operand, axis, span, cores, copies in cases:
            key = (round_index, dtype, operand, axis, span, cores, copies)
            if key in done:
                continue
            require_idle()
            cmd = [BINARIES[dtype], operand, str(axis), str(span),
                   str(axis), str(axis), str(K), str(copies), str(cores),
                   '10', '16']
            run = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
            if run.returncode:
                raise RuntimeError(f'{key} exited {run.returncode}: {run.stderr[-1000:]}')
            require_idle()
            row = json.loads(run.stdout.splitlines()[-1])
            row.update(dtype=dtype, round=round_index, packet_bytes=2 * axis * span,
                       nd_rows=axis if operand == 'A' else span,
                       unix_time=time.time())
            with output.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            done.add(key)
            print(f'{round_index} {dtype} {operand}{axis} K{span} c{cores} '
                  f'p{copies} {row["device_elapsed_us"]:.3f} us', flush=True)


if __name__ == '__main__':
    main()
