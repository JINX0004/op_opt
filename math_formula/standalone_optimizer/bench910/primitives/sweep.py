#!/usr/bin/env python3
"""Collect source-independent launch and GM->L1 task-time probes on 910B3.

Every point gets its own msprof run, so task_time measures the device kernel
and avoids host submission gaps in event brackets. Run with CANN set_env.sh.
"""
import csv
import argparse
import json
import statistics
import subprocess
from pathlib import Path


def run_point(root: Path, mode: int, cores: int, packets: int, packet_bytes: int):
    label = f"m{mode}_c{cores}_p{packets}_b{packet_bytes}"
    profile = root / label
    cmd = ["msprof", f"--output={profile}", "--task-time=on", "--ai-core=on",
           str(Path(__file__).resolve().parent / "build/solvebase_probe"),
           str(mode), str(cores), str(packets), str(packet_bytes), "10", "1", "8"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = sorted(profile.glob("PROF_*/mindstudio_profiler_output/task_time_*.csv"))
    if not files:
        raise RuntimeError(f"expected a task time file for {label}")
    with files[-1].open(newline="") as stream:
        durations = [float(row["task_time(us)"]) for row in csv.DictReader(stream)
                     if row["kernel_name"].startswith("solvebase_probe_kernel")
                     and row["kernel_type"] == "AI_CORE"]
    if len(durations) < 8:
        raise RuntimeError(f"missing kernels for {label}: {durations}")
    event_rows = [json.loads(line) for line in result.stdout.splitlines()
                  if line.startswith("{\"mode\"")]
    return {"label": label, "mode": mode, "cores": cores,
            "packets": packets, "packet_bytes": packet_bytes,
            # Modes 1/4 measure input bytes; mode 3 measures output bytes.
            # Mode 2 varies Cube repeats with fixed GM traffic.
            "bytes_per_core": packets * packet_bytes if mode in (1, 3, 4) else 0,
            "flop_per_core": 2 * 128**3 * packets if mode == 2 else 0,
            "kernel_us": durations[-8:],
            "median_kernel_us": statistics.median(durations[-8:]),
            "event_rows": event_rows, "profile_csv": str(files[-1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("launch_read_synced",
                                           "cube_write", "nd2nz_read_corrected",
                                           "hbm_read"),
                        default="launch_read_synced")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "measurements"
    root.mkdir(exist_ok=True)
    if args.group == "launch_read_synced":
        points = [(0, c, 0, 32) for c in (1, 4, 20)]
        for cores in (1, 20):
            # Fixed 256 KiB total with varying packet count.
            points += [(1, cores, p, 262144 // p) for p in (4, 8, 16, 32, 64)]
            # Fixed packet count with varying amount of data.
            points += [(1, cores, 16, b) for b in (1024, 2048, 4096, 32768, 65536)]
    elif args.group == "cube_write":
        points = []
        for cores in (1, 20):
            points += [(2, cores, p, 32768) for p in (1, 2, 4, 8, 16, 32, 64, 128)]
            points += [(3, cores, p, 262144 // p) for p in (8, 16, 32, 64)]
            points += [(3, cores, 16, b) for b in (4096, 8192, 16384, 32768)]
    elif args.group == "nd2nz_read_corrected":
        points = []
        for cores in (1, 20):
            points += [(4, cores, p, 262144 // p) for p in (8, 16, 32, 64)]
            points += [(4, cores, 16, b) for b in (4096, 8192, 16384, 32768)]
    else:
        # Device 0 reports 192 MiB L2. Use large L1 packets to cross that
        # capacity with at most 64 serialized loads per core. Earlier p=128
        # attempts used an undersized source allocation and are invalid;
        # they do not establish a device limit on packet count.
        points = [(4, 20, p, b) for p, b in
                  ((32, 262144), (48, 262144),
                   (64, 196608), (64, 262144))]
    output = root / f"{args.group}_raw.jsonl"
    done = set()
    if output.exists():
        done = {json.loads(line)["label"] for line in output.read_text().splitlines()}
    for i, point in enumerate(points):
        label = f"m{point[0]}_c{point[1]}_p{point[2]}_b{point[3]}"
        if label in done:
            continue
        row = run_point(root, *point)
        with output.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        done.add(label)
        print(f"{i+1}/{len(points)} {label} {row['median_kernel_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
