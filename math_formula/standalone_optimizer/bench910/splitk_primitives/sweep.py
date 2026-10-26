#!/usr/bin/env python3
"""Profile source-shaped 40-AIV Split-K conversion and reduction probes."""
import argparse
import csv
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBE = ROOT / "build/splitk_probe"
RAW = ROOT / "repeat_raw.jsonl"


def points():
    result = [("launch", 0, 256, 65536, 256, 1, 16, 2048)]
    result += [("convert", 1, m, 65536, 256, 1, 16, 2048)
               for m in (128, 256, 384, 512, 768, 1024)]
    # Frozen host-tiler baseAN/baseAD for the three long-K source cases.
    result += [("convert_source", 1, m, 65536, 256, 1, base_n, 2048)
               for m, base_n in ((256, 18), (512, 21), (1024, 23))]
    result += [("reduce", 2, 256, 65536, e, p, 16, 2048)
               for e in (1024, 4096)
               for p in (1, 2, 4, 8, 16, 20)]
    result += [("reduce", 2, 256, 65536, e, p, 16, 2048)
               for e in (256, 8192) for p in (1, 20)]
    result += [("vector", 3, 256, 65536, e, p, 16, 2048)
               for e in (1024, 4096) for p in (1, 4, 16, 64, 128)]
    result += [("vector", 3, 256, 65536, 8192, p, 16, 2048)
               for p in (1, 128)]
    return result


def measure(round_index, item):
    label, mode, m, k, elements, partials, base_n, base_d = item
    key = f"r{round_index}_{label}_m{m}_e{elements}_p{partials}"
    output = ROOT / "profiles" / key
    command = ["msprof", f"--output={output}", "--task-time=on",
               "--ai-core=on", str(PROBE), str(mode), str(m), str(k),
               str(elements), str(partials), str(base_n), str(base_d),
               "10", "8"]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"{key}: {process.stderr[-2000:]}")
    files = sorted(output.glob("PROF_*/mindstudio_profiler_output/task_time_*.csv"))
    if not files:
        raise RuntimeError(f"{key}: missing task_time CSV")
    with files[-1].open(newline="") as stream:
        times = [float(row["task_time(us)"]) for row in csv.DictReader(stream)
                 if row["kernel_name"].startswith("splitk_probe_kernel")
                 and row["kernel_type"] == "AI_VECTOR_CORE"]
    if len(times) < 8:
        raise RuntimeError(f"{key}: expected eight AIV task times, got {times}")
    return {"round": round_index, "key": key, "group": label,
            "mode": mode, "m": m, "k": k,
            "elements_per_aiv": elements, "partials": partials,
            "base_n": base_n, "base_d": base_d,
            "kernel_us": times[-8:],
            "median_us": statistics.median(times[-8:]),
            "profile_csv": str(files[-1]), "unix_time": time.time()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    done = {(row["round"], row["key"]) for row in
            map(json.loads, RAW.read_text().splitlines())} if RAW.exists() else set()
    for round_index in range(args.rounds):
        order = points()
        random.Random(20261026 + round_index).shuffle(order)
        for item in order:
            label, _, m, _, elements, partials, _, _ = item
            key = f"r{round_index}_{label}_m{m}_e{elements}_p{partials}"
            if (round_index, key) in done:
                continue
            row = measure(round_index, item)
            with RAW.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"{key}: {row['median_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
