#!/usr/bin/env python3
"""Paired FP16 single-core Split-K primitive probes on an Ascend 910B3.

Run on the device after sourcing CANN 9.0.0 set_env.sh. The sibling
primitives and splitk_primitives directories must have been built there.
Each msprof invocation is sequential to avoid competing for device 0.
"""
import argparse
import csv
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "singlecore_calibration_raw.jsonl"


def points():
    items = []
    for packet_bytes in (8192, 32768, 65536):
        for packets in (1, 4, 16, 64):
            for mode in (5, 6):
                items.append(("fixpipe", mode, packets, packet_bytes))
    for elements in (1024, 4096, 8192):
        for repetitions in (1, 4, 16, 64):
            items.append(("cast", 4, elements, repetitions))
    return items


def measure(round_index, item):
    group, mode, x, y = item
    label = f"r{round_index}_{group}_m{mode}_x{x}_y{y}"
    profile = ROOT / "singlecore_profiles" / label
    if group == "fixpipe":
        runner = ROOT / "primitives/build/solvebase_probe"
        args = [str(mode), "20", str(x), str(y), "10", "1", "8"]
        kernel_type = "AI_CORE"
        prefix = "solvebase_probe_kernel"
    else:
        runner = ROOT / "splitk_primitives/build/splitk_probe"
        args = ["4", "256", "65536", str(x), str(y), "16", "2048", "10", "8"]
        kernel_type = "AI_VECTOR_CORE"
        prefix = "splitk_probe_kernel"
    cmd = ["msprof", f"--output={profile}", "--task-time=on", "--ai-core=on",
           str(runner), *args]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{label}: {result.stderr[-2000:]}")
    files = sorted(profile.glob("PROF_*/mindstudio_profiler_output/task_time_*.csv"))
    if not files:
        raise RuntimeError(f"{label}: no task_time CSV")
    with files[-1].open(newline="") as stream:
        times = [float(row["task_time(us)"]) for row in csv.DictReader(stream)
                 if row["kernel_name"].startswith(prefix)
                 and row["kernel_type"] == kernel_type]
    if len(times) < 8:
        raise RuntimeError(f"{label}: only {len(times)} matching tasks")
    return {"round": round_index, "label": label, "group": group,
            "mode": mode, "x": x, "y": y, "task_us": times[-8:],
            "median_us": statistics.median(times[-8:]),
            "profile_csv": str(files[-1]), "unix_time": time.time()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--group", choices=("fixpipe", "cast", "all"), default="all")
    args = parser.parse_args()
    items = [p for p in points() if args.group == "all" or p[0] == args.group]
    done = {row["label"] for row in map(json.loads, RAW.read_text().splitlines())} \
        if RAW.exists() else set()
    for round_index in range(args.rounds):
        order = items.copy()
        random.Random(20261030 + round_index).shuffle(order)
        for item in order:
            group, mode, x, y = item
            label = f"r{round_index}_{group}_m{mode}_x{x}_y{y}"
            if label in done:
                continue
            row = measure(round_index, item)
            with RAW.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            done.add(label)
            print(f"{label}: {row['median_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
