#!/usr/bin/env python3
"""Parallel source MatrixAtoNZV2 task-time sweep on devices 4–7."""

import csv
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/converted_a_probe_v1')
EXE = ROOT / 'build/mmv3_converted_a_probe'
OUT = ROOT / 'sweep_v1'
OUT.mkdir(parents=True, exist_ok=True)
CASES = [
    (65536, 19, 2048), (65536, 47, 1024),
    (73728, 19, 2048), (73728, 47, 1024),
]


def check_idle(device, directory, name):
    output = subprocess.run(['npu-smi', 'info'], check=True, text=True,
                            capture_output=True).stdout
    (directory / f'npu_smi_{name}.txt').write_text(output)
    if f'No running processes found in NPU {device}' not in output:
        raise RuntimeError(f'device {device} occupied at {name}')


def measure(device, k, base_an, base_ad, mode, round_index):
    directory = OUT / f'device{device}' / f'k{k}_an{base_an}_ad{base_ad}_mode{mode}_r{round_index}'
    directory.mkdir(parents=True, exist_ok=True)
    row_path = directory / 'result.json'
    if row_path.exists():
        return json.loads(row_path.read_text())
    check_idle(device, directory, 'before')
    cmd = [str(EXE), str(device), '2560', str(k), str(base_an), str(base_ad),
           str(mode), '10', '1', '8']
    with (directory / 'stdout.jsonl').open('w') as stdout, \
         (directory / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(['msprof', f'--output={directory / "profile"}',
                               '--task-time=on', '--ai-core=on', *cmd],
                              stdout=stdout, stderr=stderr)
    if proc.returncode:
        raise RuntimeError(f'profile failed: {directory}, rc={proc.returncode}')
    check_idle(device, directory, 'after')
    files = sorted(directory.glob('profile/PROF_*/mindstudio_profiler_output/task_time_*.csv'))
    if len(files) != 1:
        raise RuntimeError(f'expected one task CSV: {directory}, found {len(files)}')
    with files[0].open(newline='') as source:
        rows = [r for r in csv.DictReader(source)
                if 'mmv3_converted_a_probe_kernel' in r['kernel_name']
                and r['kernel_type'] == 'AI_VECTOR_CORE']
    if len(rows) < 8:
        raise RuntimeError(f'expected eight measured kernel tasks: {directory}')
    task_us = [float(r['task_time(us)']) for r in rows[-8:]]
    runner_lines = [line for line in (directory / 'stdout.jsonl').read_text().splitlines()
                    if line.startswith('{"device":')]
    if len(runner_lines) != 1:
        raise RuntimeError(f'expected one validated runner result: {directory}')
    payload = {'device': device, 'm': 2560, 'k': k,
               'baseAN': base_an, 'baseAD': base_ad,
               'mode': mode, 'round': round_index,
               'task_us': task_us, 'median_us': statistics.median(task_us),
               'profile_csv': str(files[0]),
               'runner': json.loads(runner_lines[0])}
    row_path.write_text(json.dumps(payload, indent=2) + '\n')
    print('result', device, k, base_an, base_ad, mode, round_index,
          payload['median_us'], flush=True)
    return payload


def device_worker(device):
    rows = []
    rows.append(measure(device, 65536, 19, 2048, 0, 0))
    order = CASES[device - 4:] + CASES[:device - 4]
    for k, base_an, base_ad in order:
        rows.append(measure(device, k, base_an, base_ad, 1, 0))
    # Repeat each source-emitted tile with the order reversed.
    for k, base_an, base_ad in [(73728, 47, 1024), (65536, 19, 2048)]:
        rows.append(measure(device, k, base_an, base_ad, 1, 1))
    return rows


all_rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(device_worker, device): device for device in (4, 5, 6, 7)}
    for future in as_completed(futures):
        all_rows.extend(future.result())
        (OUT / 'summary.json').write_text(json.dumps(all_rows, indent=2) + '\n')
