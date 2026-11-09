#!/usr/bin/env python3
"""Measure the 128-KiB A-pitch penalty across intermediate AIC counts."""

import importlib.util
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/fp32_mode2_copy_route_v1')
spec = importlib.util.spec_from_file_location('copy_run', ROOT / 'run.py')
copy_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copy_run)
OUT = ROOT / 'core_sweep'
CORES = (2, 4, 8, 12, 16)


def one_device(device, env):
    dest = OUT / f'device{device}'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'npu_smi_before.txt').write_text(copy_run.require_idle(device))
    rows = []
    for cores in (CORES if device == 4 else tuple(reversed(CORES))):
        for pitch in (copy_run.PITCHES if device == 4 else tuple(reversed(copy_run.PITCHES))):
            for route in (('inline', 'plain') if device == 4 else ('plain', 'inline')):
                cmd = [str(copy_run.BINARY), route, str(pitch),
                       str(copy_run.LOGICAL_K), '2560', '1', str(cores),
                       str(device), '3', '12']
                proc = subprocess.run(cmd, env=env, cwd=ROOT, text=True,
                                      capture_output=True, timeout=240)
                if proc.returncode:
                    raise RuntimeError((cmd, proc.stdout[-1000:], proc.stderr[-2000:]))
                row = json.loads(proc.stdout.strip().splitlines()[-1])
                rows.append(row)
                (dest / 'records.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(device, cores, pitch, route, row['device_elapsed_us'], flush=True)
    (dest / 'npu_smi_after.txt').write_text(copy_run.require_idle(device))
    return rows


def main():
    env = copy_run.environment()
    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(lambda device: one_device(device, env), (4, 6)))
    (OUT / 'records.json').write_text(json.dumps([r for group in groups for r in group], indent=2) + '\n')


if __name__ == '__main__':
    main()
