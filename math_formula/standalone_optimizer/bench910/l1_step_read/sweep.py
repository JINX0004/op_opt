#!/usr/bin/env python3
"""Profile source-emitted 128 KiB A/B L1-step transfer geometries."""

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
COPIES = (2, 4, 8, 16, 32)
CASES = (
    ("default", "A", 128, 512),
    ("default", "B", 256, 256),
    ("swapped", "A", 256, 256),
    ("swapped", "B", 128, 512),
)
HELDOUT_CASES = (
    ("heldout", "A", 128, 256),
    ("heldout", "A", 256, 128),
    ("heldout", "B", 128, 256),
    ("heldout", "B", 256, 128),
)


def run_point(round_index, orientation, operand, axis, span, copies, root):
    label = f"r{round_index}_{orientation}_{operand}{axis}k{span}_c{copies}"
    profile = root / label
    command = ["msprof", f"--output={profile}", "--task-time=on",
               "--ai-core=on", str(HERE / "build/l1_step_read_probe"),
               operand, str(axis), str(span), *(str(v) for v in SHAPE),
               str(copies), str(CORES), "10", "8"]
    result = subprocess.run(command, capture_output=True, text=True,
                            check=True, timeout=180)
    files = list(profile.glob("PROF_*/mindstudio_profiler_output/task_time_*.csv"))
    if not files:
        raise RuntimeError(f"expected profiler task CSV for {label}")
    latest = max(files, key=lambda path: path.stat().st_mtime_ns)
    with latest.open(newline="") as stream:
        durations = [float(row["task_time(us)"]) for row in csv.DictReader(stream)
                     if row["kernel_name"].startswith("l1_step_read_kernel")
                     and row["kernel_type"] == "AI_CORE"]
    if len(durations) < 8:
        raise RuntimeError(f"expected eight measured tasks for {label}")
    durations = durations[-8:]
    event_rows = [json.loads(line) for line in result.stdout.splitlines()
                  if line.startswith('{"operand"')]
    if len(event_rows) != 1 or event_rows[0]["k_span"] != span:
        raise RuntimeError(f"runner verification failed for {label}")
    return {
        "label": label, "round": round_index, "orientation": orientation,
        "operand": operand, "tile_axis": axis, "k_span": span,
        "copies": copies, "shape": list(SHAPE), "cores": CORES,
        "packet_bytes": 2 * axis * span,
        "requested_bytes_per_core": 2 * axis * span * copies,
        "source_pitch_elements": SHAPE[2] if operand == "A" else SHAPE[1],
        "kernel_us": durations,
        "median_kernel_us": statistics.median(durations),
        "event_row": event_rows[0], "profile_csv": str(latest),
        "unix_time": time.time(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--heldout", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        raise SystemExit("rounds must be 1..10")
    root = HERE / ("measurements_heldout" if args.heldout else "measurements")
    root.mkdir(exist_ok=True)
    output = HERE / ("l1_step_read_heldout_raw.jsonl" if args.heldout
                     else "l1_step_read_raw.jsonl")
    done = set()
    if output.exists():
        done = {json.loads(line)["label"]
                for line in output.read_text().splitlines() if line.strip()}
    rng = random.Random(20261026)
    for round_index in range(args.rounds):
        cases = HELDOUT_CASES if args.heldout else CASES
        counts = (4, 16, 32) if args.heldout else COPIES
        order = [(case, copies) for case in cases for copies in counts]
        rng.shuffle(order)
        for (orientation, operand, axis, span), copies in order:
            label = f"r{round_index}_{orientation}_{operand}{axis}k{span}_c{copies}"
            if label in done:
                continue
            row = run_point(round_index, orientation, operand, axis, span,
                            copies, root)
            with output.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            done.add(label)
            print(f"{len(done)}/{args.rounds * len(order)} {label} "
                  f"{row['median_kernel_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
