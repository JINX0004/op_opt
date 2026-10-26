#!/usr/bin/env python3
"""Fit MTE2 packet service from primitive probes; check held-out geometries.

The response is per AIC and excludes a whole-MatMul timing fit. Its three
features are one issued packet, requested KiB, and ND source rows. The two
operands have separate coefficients because their ND layouts and pitches
differ. The fit is descriptive for this 20-AIC, L2-hot 2048^2x4096 setting.
"""

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "directional_read" / "directional_read_raw.jsonl"
TRAIN = HERE / "l1_step_read_raw.jsonl"
HOLDOUT = HERE / "l1_step_read_heldout_raw.jsonl"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def slope(rows):
    counts = sorted({row["copies"] for row in rows})
    if len(counts) < 3:
        raise ValueError("at least three copy counts are required")
    # Median across independent runs at each count suppresses isolated
    # profiler excursions before fitting the slope in copy count.
    times = [statistics.median(row["median_kernel_us"] for row in rows
                               if row["copies"] == count)
             for count in counts]
    mean_x, mean_y = statistics.mean(counts), statistics.mean(times)
    return sum((x - mean_x) * (y - mean_y)
               for x, y in zip(counts, times)) / sum(
                   (x - mean_x) ** 2 for x in counts)


def observations(rows, name):
    groups = {}
    for row in rows:
        span = row.get("k_span", row.get("base_k"))
        key = (row["operand"], row["tile_axis"], span)
        groups.setdefault(key, []).append(row)
    return [{"dataset": name, "operand": op, "tile_axis": axis,
             "rows": axis if op == "A" else span,
             "k_span": span, "bytes": 2 * axis * span,
             "measured_us": slope(group)}
            for (op, axis, span), group in sorted(groups.items())]


def solve_3(matrix, vector):
    a = [list(row) + [value] for row, value in zip(matrix, vector)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("rank-deficient packet probe design")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(3):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [row[3] for row in a]


def features(point):
    return (1.0, point["bytes"] / 1024.0, float(point["rows"]))


def fit(points):
    x = [features(point) for point in points]
    y = [point["measured_us"] for point in points]
    gram = [[sum(row[i] * row[j] for row in x) for j in range(3)]
            for i in range(3)]
    rhs = [sum(row[i] * value for row, value in zip(x, y))
           for i in range(3)]
    return solve_3(gram, rhs)


def predicted(point, coefficients):
    return sum(a * b for a, b in zip(features(point), coefficients))


def main():
    training = observations(read(OLD), "baseK64") + \
               observations(read(TRAIN), "L1-step")
    heldout = observations(read(HOLDOUT), "held-out") if HOLDOUT.exists() else []
    result = {"model": "per-AIC packet_us = intercept_us + "
                       "us_per_kib * packet_kib + us_per_row * ND_rows",
              "scope": "20-AIC, L2-hot, 2048x2048x4096 FP16 ND->L1 NZ",
              "coefficients": {}, "training": [], "heldout": []}
    for operand in ("A", "B"):
        points = [point for point in training if point["operand"] == operand]
        coefficients = fit(points)
        result["coefficients"][operand] = dict(zip(
            ("intercept_us", "us_per_kib", "us_per_row"), coefficients))
        for point in points + [point for point in heldout
                               if point["operand"] == operand]:
            estimate = predicted(point, coefficients)
            record = {**point, "predicted_us": estimate,
                      "error_us": estimate - point["measured_us"],
                      "relative_error": estimate / point["measured_us"] - 1}
            result["heldout" if point["dataset"] == "held-out"
                   else "training"].append(record)
    if heldout:
        result["heldout_max_abs_relative_error"] = max(
            abs(point["relative_error"]) for point in result["heldout"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
