#!/usr/bin/env python3
"""Randomized 910B3 task-time sweep for A/B-shaped ND-to-NZ MTE2 packets."""

import argparse
import csv
import json
import random
import statistics
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHAPE = (2048, 2048, 4096)
CORES = 20
BASE_K = 64
COPIES = (8, 16, 32, 64)


def run_point(round_index: int, operand: str, tile_axis: int, copies: int,
              root: Path) -> dict:
    label = f"r{round_index}_{operand}{tile_axis}_c{copies}"
    profile = root / label
    executable = HERE / "build/directional_read_probe"
    command = ["msprof", f"--output={profile}", "--task-time=on",
               "--ai-core=on", str(executable), operand, str(tile_axis),
               *(str(v) for v in SHAPE), str(copies), str(CORES), "10", "8"]
    result = subprocess.run(command, capture_output=True, text=True,
                            check=True, timeout=180)
    files = list(profile.glob("PROF_*/mindstudio_profiler_output/task_time_*.csv"))
    if not files:
        raise RuntimeError(f"expected a task-time CSV for {label}")
    latest = max(files, key=lambda path: path.stat().st_mtime_ns)
    with latest.open(newline="") as stream:
        durations = [float(row["task_time(us)"]) for row in csv.DictReader(stream)
                     if row["kernel_name"].startswith("directional_read_kernel")
                     and row["kernel_type"] == "AI_CORE"]
    if len(durations) < 8:
        raise RuntimeError(f"expected eight AIC task times for {label}: {durations}")
    durations = durations[-8:]
    event_rows = [json.loads(line) for line in result.stdout.splitlines()
                  if line.startswith('{"operand"')]
    if len(event_rows) != 1 or event_rows[0]["operand"] != operand:
        raise RuntimeError(f"missing runner verification for {label}")
    return {
        "label": label, "round": round_index, "operand": operand,
        "tile_axis": tile_axis, "copies": copies, "shape": list(SHAPE),
        "base_k": BASE_K, "cores": CORES,
        "packet_bytes": 2 * tile_axis * BASE_K,
        "requested_bytes_per_core": 2 * tile_axis * BASE_K * copies,
        "source_pitch_elements": SHAPE[2] if operand == "A" else SHAPE[1],
        "input_bytes": 2 * SHAPE[0] * SHAPE[2] if operand == "A"
                       else 2 * SHAPE[2] * SHAPE[1],
        "kernel_us": durations,
        "median_kernel_us": statistics.median(durations),
        "event_row": event_rows[0], "profile_csv": str(latest),
        "unix_time": time.time(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        raise SystemExit("rounds must be 1..10")
    root = HERE / "measurements"
    root.mkdir(exist_ok=True)
    output = HERE / "directional_read_raw.jsonl"
    done = set()
    if output.exists():
        done = {json.loads(line)["label"]
                for line in output.read_text().splitlines() if line.strip()}
    points = [(operand, axis, copies) for operand in ("A", "B")
              for axis in (128, 256) for copies in COPIES]
    rng = random.Random(20261026)
    for round_index in range(args.rounds):
        order = points.copy()
        rng.shuffle(order)
        for operand, axis, copies in order:
            label = f"r{round_index}_{operand}{axis}_c{copies}"
            if label in done:
                continue
            row = run_point(round_index, operand, axis, copies, root)
            with output.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            done.add(label)
            print(f"{len(done)}/{args.rounds * len(points)} {label} "
                  f"{row['median_kernel_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
