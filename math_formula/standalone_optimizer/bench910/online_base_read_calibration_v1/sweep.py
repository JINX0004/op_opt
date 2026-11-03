#!/usr/bin/env python3
"""Randomized device-4 event sweep of source-shaped ND-to-L1 A/B packets."""
import json
import random
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/online_base_read_calibration_v1')
ROOT.mkdir(parents=True, exist_ok=True)
BINARY = '/tmp/mmv3mte2sweep/build/l1_step_read_probe'
K = 102400
pairs = []
for width in (16, 32, 64, 128, 256):
    full_span = 65536 // width  # 128 KiB FP16 packet
    for span in (full_span // 2, full_span):
        for cores in (1, 4, 20):
            pairs.append(('B', width, span, cores))
    for cores in (1, 20):
        pairs.append(('A', width, full_span, cores))
random.Random(20261102).shuffle(pairs)
output = ROOT / 'event_sweep.jsonl'
done = set()
if output.exists():
    for line in output.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            done.add((row['operand'], row['tile_axis'], row['k_span'], row['cores'], row['copies']))
for index, (operand, width, span, cores) in enumerate(pairs, 1):
    counts = (1, 25) if index % 2 else (25, 1)
    if all((operand, width, span, cores, count) in done for count in counts):
        continue
    snapshot = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True, check=True).stdout
    if 'No running processes found in NPU 4' not in snapshot:
        raise RuntimeError(f'device 4 occupied before {(operand, width, span, cores)}')
    for copies in counts:
        key = (operand, width, span, cores, copies)
        if key in done:
            continue
        cmd = [BINARY, operand, str(width), str(span), str(width), str(width),
               str(K), str(copies), str(cores), '10', '16']
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f'{key} rc={result.returncode}: {result.stderr[-1200:]}')
        row = json.loads(result.stdout.splitlines()[-1])
        row.update({'round': 0, 'packet_bytes': 2 * width * span,
                    'nd_rows': width if operand == 'A' else span,
                    'unix_time': time.time()})
        with output.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        done.add(key)
        print(index, operand, width, span, cores, copies,
              f'{row["device_elapsed_us"]:.3f} us', flush=True)
    snapshot = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True, check=True).stdout
    if 'No running processes found in NPU 4' not in snapshot:
        raise RuntimeError(f'device 4 occupied after {(operand, width, span, cores)}')
