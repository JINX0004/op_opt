#!/usr/bin/env python3
"""Measure source-shaped A/B MTE2 demand at fixed logical K."""

import hashlib
import importlib.util
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/fp32_mode2_copy_route_v1')
spec = importlib.util.spec_from_file_location('copy_run', ROOT / 'run.py')
copy_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copy_run)
BINARY = ROOT / 'build/ab_route_probe'
OUT = ROOT / 'ab_sweep'


def one_geometry(n, device, env):
    dest = OUT / f'N{n}_device{device}'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'npu_smi_before.txt').write_text(copy_run.require_idle(device))
    rows = []
    for cores in (20, 12, 8):
        for round_index in range(2):
            pitches = copy_run.PITCHES if round_index == 0 else tuple(reversed(copy_run.PITCHES))
            routes = ('inline', 'plain') if (device + round_index) % 2 == 0 else ('plain', 'inline')
            for pitch in pitches:
                for route in routes:
                    cmd = [str(BINARY), route, str(pitch), str(copy_run.LOGICAL_K),
                           '2560', str(n), '1', str(cores), str(device), '3', '5']
                    proc = subprocess.run(cmd, env=env, cwd=ROOT, text=True,
                                          capture_output=True, timeout=300)
                    if proc.returncode:
                        raise RuntimeError((cmd, proc.stdout[-1000:], proc.stderr[-2000:]))
                    row = json.loads(proc.stdout.strip().splitlines()[-1])
                    row['round'] = round_index
                    rows.append(row)
                    (dest / 'records.json').write_text(json.dumps(rows, indent=2) + '\n')
                    print(n, device, cores, round_index, pitch, route,
                          row['device_elapsed_us'], flush=True)
    (dest / 'npu_smi_after.txt').write_text(copy_run.require_idle(device))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'ab_binary_sha256.txt').write_text(
        hashlib.sha256(BINARY.read_bytes()).hexdigest() + '\n')
    env = copy_run.environment()
    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(lambda item: one_geometry(*item, env), ((768, 4), (1024, 6))))
    (OUT / 'records.json').write_text(json.dumps([row for group in groups for row in group], indent=2) + '\n')


if __name__ == '__main__':
    main()
