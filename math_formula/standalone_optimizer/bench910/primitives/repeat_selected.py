#!/usr/bin/env python3
"""Five randomized independent profiler runs for representative probe points."""
import json
import random
import time
from pathlib import Path

from sweep import run_point


def main():
    root = Path(__file__).resolve().parent / "measurements"
    output = root / "repeat_selected_corrected_raw.jsonl"
    points = [
        (0, 20, 0, 32),
        (1, 20, 16, 16384),
        (1, 20, 64, 4096),
        (4, 20, 16, 16384),
        (4, 20, 64, 4096),
        (4, 20, 16, 4096),
        (4, 20, 16, 32768),
        (2, 20, 8, 32768),
        (2, 20, 64, 32768),
        (3, 20, 16, 16384),
        (3, 20, 64, 4096),
        (3, 20, 16, 4096),
        (3, 20, 16, 32768),
    ]
    done = set()
    if output.exists():
        done = {(r["round"], r["label"])
                for r in map(json.loads, output.read_text().splitlines())}
    rng = random.Random(9102026)
    for round_index in range(5):
        order = points.copy()
        rng.shuffle(order)
        for point in order:
            label = f"m{point[0]}_c{point[1]}_p{point[2]}_b{point[3]}"
            if (round_index, label) in done:
                continue
            row = run_point(root, *point)
            row["round"] = round_index
            row["unix_time"] = time.time()
            with output.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"round {round_index + 1}/5 {label} "
                  f"{row['median_kernel_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
