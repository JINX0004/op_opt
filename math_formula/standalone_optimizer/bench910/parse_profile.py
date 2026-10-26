#!/usr/bin/env python3
"""Summarize actual device tasks from a CANN msprof export.

Use this for MatMulV3 kernel time. The AscendCL event bracket can include
host submission gaps, so it is intentionally kept as a separate measurement.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path, help="msprof --output directory or task_time CSV")
    parser.add_argument("--last", type=int, default=0,
                        help="take the last N matching kernels after warmup")
    parser.add_argument("--kernel", default="MatMulV3")
    parser.add_argument("--kernel-type", default="KERNEL_AICORE",
                        help="task_time kernel_type (MIX_AIC for mixed Split-K)")
    args = parser.parse_args()
    if args.profile.is_file():
        csv_path = args.profile
    else:
        files = sorted(args.profile.glob("PROF_*/mindstudio_profiler_output/task_time_*.csv"))
        if not files:
            raise SystemExit("no task_time CSV in profile directory")
        csv_path = files[-1]
    with csv_path.open(newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if args.kernel in row["kernel_name"] and
                row["kernel_type"] == args.kernel_type]
    if not rows:
        raise SystemExit("no matching AICore kernel in " + str(csv_path))
    if args.last:
        if args.last < 0 or args.last > len(rows):
            raise SystemExit("invalid --last count")
        rows = rows[-args.last:]
    names = sorted({row["kernel_name"] for row in rows})
    durations = [float(row["task_time(us)"]) for row in rows]
    print(json.dumps({
        "kind": "msprof_kernel",
        "profile_csv": str(csv_path),
        "kernel_names": names,
        "kernel_count": len(rows),
        "min_us": min(durations),
        "median_us": statistics.median(durations),
        "max_us": max(durations),
        "durations_us": durations,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
