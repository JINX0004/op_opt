#!/usr/bin/env python3
"""Compare A-only inline ND2NZ and plain GM-to-L1 copies at fixed work."""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/fp32_mode2_copy_route_v1')
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
BINARY = ROOT / 'build/copy_route_probe'
PITCHES = (32512, 32768, 33024)
LOGICAL_K = 32256


def environment():
    proc = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                           'bash', str(CANN)], capture_output=True, check=True)
    return dict(item.decode().split('=', 1) for item in proc.stdout.split(b'\0') if item)


def require_idle(device):
    info = subprocess.run(['npu-smi', 'info'], text=True,
                          capture_output=True, check=True).stdout
    if f'No running processes found in NPU {device}' not in info:
        raise RuntimeError(f'device {device} occupied')
    return info


def run_device(device, env):
    out = ROOT / f'device{device}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'npu_smi_before.txt').write_text(require_idle(device))
    rows = []
    for cores in (20, 1):
        for round_index in range(2):
            pitches = PITCHES if round_index == 0 else tuple(reversed(PITCHES))
            routes = ('inline', 'plain') if (device + round_index) % 2 == 0 else ('plain', 'inline')
            for pitch in pitches:
                for route in routes:
                    cmd = [str(BINARY), route, str(pitch), str(LOGICAL_K),
                           '2560', '1', str(cores), str(device), '3', '12']
                    proc = subprocess.run(cmd, env=env, cwd=ROOT, text=True,
                                          capture_output=True, timeout=240)
                    if proc.returncode:
                        raise RuntimeError((cmd, proc.stdout[-1000:], proc.stderr[-2000:]))
                    data = json.loads(proc.stdout.strip().splitlines()[-1])
                    data['round'] = round_index
                    rows.append(data)
                    (out / 'records.json').write_text(json.dumps(rows, indent=2) + '\n')
                    print(device, cores, round_index, pitch, route,
                          data['device_elapsed_us'], flush=True)
    (out / 'npu_smi_after.txt').write_text(require_idle(device))
    return rows


def main():
    fingerprint = hashlib.sha256(BINARY.read_bytes()).hexdigest()
    (ROOT / 'binary_sha256.txt').write_text(fingerprint + '\n')
    env = environment()
    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(lambda device: run_device(device, env), (4, 6)))
    rows = [row for group in groups for row in group]
    (ROOT / 'records.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('completed', len(rows), 'measurements', flush=True)


if __name__ == '__main__':
    main()
