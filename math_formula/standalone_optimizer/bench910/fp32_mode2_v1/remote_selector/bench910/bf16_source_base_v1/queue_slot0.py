#!/usr/bin/env python3
"""Run a shard of interrupted slot 0 after another device's worker exits."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, action='append', required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--shard', type=int, choices=range(3), required=True)
    args = parser.parse_args()
    while any(Path(f'/proc/{pid}').exists() for pid in args.wait_pid):
        time.sleep(30)
    print(f'resuming slot 0 shard {args.shard} on device {args.device}', flush=True)
    subprocess.run([sys.executable, str(HERE / 'run.py'), '--device',
                    str(args.device), '--slot', '0', '--shards', '3',
                    '--shard', str(args.shard)], cwd=HERE, check=True)


if __name__ == '__main__':
    main()
