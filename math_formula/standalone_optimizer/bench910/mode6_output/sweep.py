#!/usr/bin/env python3
"""Measure FP32 output-only service on device 4 with the 910B3 probe.

The probe checks every run's first FP32 output and completion token. This
stream has source spatial tiles and repeated addresses, but it precomputes
L0C once: its rate is not automatically the source kernel's exposed output
service while Cube and MTE run concurrently.
"""

import argparse
import itertools
import json
import random
import statistics
import subprocess
import time
from pathlib import Path


def ensure_idle(device: int) -> None:
    info = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True,
                          check=True).stdout
    if f'No running processes found in NPU {device}' not in info:
        raise RuntimeError(f'device {device} is occupied')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    points = list(itertools.product((8, 12), (1, 4, 16, 64), (0, 1), (0, 1)))
    # Additional distinct-K controls fit in L2 at four packets.
    points += [(n, 4, 0, 1, 1) for n in (8, 12)]
    rng = random.Random(20261030)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as output:
        for round_index in range(args.rounds):
            ensure_idle(args.device)
            order = [(n, k, serial, 0, atomic) for
                     n, k, serial, atomic in points[:32]] + points[32:]
            rng.shuffle(order)
            for n_tiles, k_packets, serial, distinct, atomic in order:
                cmd = [str(args.runner), str(args.device), str(k_packets),
                       str(n_tiles), str(serial), str(distinct), str(atomic),
                       '3', '8']
                result = subprocess.run(cmd, text=True, capture_output=True)
                if result.returncode:
                    raise RuntimeError(f'{cmd}: {result.stderr[-1000:]}')
                batch = [json.loads(line) for line in result.stdout.splitlines()]
                if len(batch) != 3:
                    raise RuntimeError(f'{cmd}: expected 3 batches, got {len(batch)}')
                row = {'round': round_index, 'n_tiles': n_tiles,
                       'k_packets': k_packets, 'serial': serial,
                       'distinct': distinct, 'atomic': atomic,
                       'us': [b['us'] for b in batch],
                       'median_us': statistics.median(b['us'] for b in batch),
                       'unix_time': time.time()}
                output.write(json.dumps(row, sort_keys=True) + '\n')
                output.flush()
                print(round_index, n_tiles, k_packets, serial, distinct,
                      atomic, f"{row['median_us']:.3f} us", flush=True)
            ensure_idle(args.device)


if __name__ == '__main__':
    main()
