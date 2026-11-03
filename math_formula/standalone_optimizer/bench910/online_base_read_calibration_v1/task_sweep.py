#!/usr/bin/env python3
"""Profiler task-time verification for selected event-sweep geometries."""
import csv
import json
import random
import statistics
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/online_base_read_calibration_v1/task_sweep')
ROOT.mkdir(parents=True, exist_ok=True)
BINARY = '/tmp/mmv3mte2sweep/build/l1_step_read_probe'
K = 102400
cases = []
for width in (16, 32, 64, 128, 256):
    span = 65536 // width
    for cores in (1, 20):
        for copies in (1, 25):
            cases.append(('B', width, span, cores, copies))
for width in (16, 32, 128):
    span = 32768 // width
    for copies in (1, 25):
        cases.append(('B', width, span, 1, copies))
for width in (16, 128, 256):
    span = 65536 // width
    for copies in (1, 25):
        cases.append(('A', width, span, 1, copies))
        cases.append(('A', width, span, 20, copies))
for copies in (1, 25):
    cases.append(('B', 16, 4096, 4, copies))
for width in (32, 128):
    span = 32768 // width
    for copies in (1, 25):
        cases.append(('B', width, span, 20, copies))
for cores, span in ((20, 2048), (10, 4096)):
    for copies in (1, 25):
        cases.append(('B', 16, span, cores, copies))
for width in (32, 128):
    span = 32768 // width
    for cores in (1, 20):
        for copies in (1, 25):
            cases.append(('A', width, span, cores, copies))
random.Random(20261103).shuffle(cases)
output = ROOT / 'results.jsonl'
done = {}
if output.exists():
    for line in output.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            done[(row['operand'], row['tile_axis'], row['k_span'], row['cores'], row['copies'])] = row
for index, (operand, width, span, cores, copies) in enumerate(cases, 1):
    key = (operand, width, span, cores, copies)
    if key in done:
        continue
    snapshot = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True, check=True).stdout
    if 'No running processes found in NPU 4' not in snapshot:
        raise RuntimeError(f'device 4 occupied before {key}')
    label = f'{operand}{width}_K{span}_c{cores}_p{copies}'
    directory = ROOT / label
    cmd = ['msprof', f'--output={directory}', '--task-time=on', '--ai-core=on',
           BINARY, operand, str(width), str(span), str(width), str(width),
           str(K), str(copies), str(cores), '10', '8']
    with (ROOT / f'{label}.stdout').open('w') as stdout, (ROOT / f'{label}.stderr').open('w') as stderr:
        proc = subprocess.run(cmd, stdout=stdout, stderr=stderr, timeout=240)
    if proc.returncode:
        raise RuntimeError(f'{label} rc={proc.returncode}: {(ROOT / f"{label}.stderr").read_text()[-1200:]}')
    files = list(directory.glob('PROF_*/mindstudio_profiler_output/task_time_*.csv'))
    if not files:
        raise RuntimeError(f'{label} missing task CSV')
    with files[-1].open(newline='') as source:
        times = [float(r['task_time(us)']) for r in csv.DictReader(source)
                 if r['kernel_name'].startswith('l1_step_read_kernel') and
                 r['kernel_type'] == 'AI_CORE']
    if len(times) < 8:
        raise RuntimeError(f'{label} too few tasks: {times}')
    row = {'operand': operand, 'tile_axis': width, 'k_span': span,
           'cores': cores, 'copies': copies, 'packet_bytes': 2 * width * span,
           'nd_rows': width if operand == 'A' else span,
           'task_times_us': times[-8:], 'median_us': statistics.median(times[-8:]),
           'profile_csv': str(files[-1])}
    with output.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')
    print(index, label, f'{row["median_us"]:.3f} us', flush=True)
