#!/usr/bin/env python3
"""Five independent above-L2 ND-to-NZ read runs, in randomized order."""
import json
import random
from pathlib import Path

from sweep import run_point


def main():
    root = Path(__file__).resolve().parent / "measurements"
    output = root / "hbm_repeat_raw.jsonl"
    points = [(4, 20, 48, 262144), (4, 20, 64, 196608),
              (4, 20, 64, 262144)]
    done = set()
    if output.exists():
        done = {(r["round"], r["label"])
                for r in map(json.loads, output.read_text().splitlines())}
    rng = random.Random(9102027)
    for round_index in range(5):
        order = points.copy()
        rng.shuffle(order)
        for point in order:
            label = f"m{point[0]}_c{point[1]}_p{point[2]}_b{point[3]}"
            if (round_index, label) in done:
                continue
            row = run_point(root, *point)
            row["round"] = round_index
            with output.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"round {round_index + 1}/5 {label} "
                  f"{row['median_kernel_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
