#!/usr/bin/env python3
"""Mode-6 source-shaped A/B GM->L1 ND->NZ packet sweep."""
import argparse
import csv
import json
import random
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "copy_raw.jsonl"


def measure(round_index, regime, operand, k_packet, copies):
    m = n = 1024 if regime == "l2" else 2048
    k = 6144 if regime == "l2" else (
        73728 if regime == "above_l2_288" else 49152)
    label = f"r{round_index}_{regime}_{operand}_k{k_packet}_p{copies}"
    path = ROOT / "profiles" / label
    cmd = ["msprof", f"--output={path}", "--task-time=on", "--ai-core=on",
           str(ROOT / "build/directional_read_probe"), operand, "128",
           str(m), str(n), str(k), str(copies), "20", "10", "8",
           str(k_packet)]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{label}: {result.stderr[-2000:]}")
    files = sorted(path.glob("PROF_*/mindstudio_profiler_output/task_time_*.csv"))
    if not files:
        raise RuntimeError(f"{label}: missing task-time CSV")
    with files[-1].open(newline="") as stream:
        times = [float(row["task_time(us)"]) for row in csv.DictReader(stream)
                 if row["kernel_name"].startswith("directional_read_kernel")
                 and row["kernel_type"] == "AI_CORE"]
    if len(times) < 8:
        raise RuntimeError(f"{label}: only {len(times)} matching tasks")
    return {"label": label, "round": round_index, "regime": regime,
            "operand": operand, "k_packet": k_packet, "copies": copies,
            "bytes_per_packet": 2 * 128 * k_packet,
            "median_us": statistics.median(times[-8:]),
            "task_us": times[-8:], "profile_csv": str(files[-1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--regime", choices=("l2", "above_l2_288", "all"), default="all")
    args = parser.parse_args()
    regimes = ("l2", "above_l2_288") if args.regime == "all" else (args.regime,)
    points = [(regime, operand, k_packet, copies)
              for regime in regimes for operand in ("A", "B")
              for k_packet in (384, 512)
              for copies in ((1, 4, 16, 64, 128, 192, 256)
                             if regime == "above_l2_288" else (1, 4, 16, 64))]
    raw = ROOT / "copy_above_l2_288_raw.jsonl" if args.regime == "above_l2_288" else RAW
    done = {row["label"] for row in map(json.loads, raw.read_text().splitlines())} \
        if raw.exists() else set()
    for round_index in range(args.rounds):
        order = points.copy()
        random.Random(20261030 + 11 * round_index).shuffle(order)
        for regime, operand, k_packet, copies in order:
            label = f"r{round_index}_{regime}_{operand}_k{k_packet}_p{copies}"
            if label in done:
                continue
            row = measure(round_index, regime, operand, k_packet, copies)
            with raw.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            done.add(label)
            print(f"{label}: {row['median_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
