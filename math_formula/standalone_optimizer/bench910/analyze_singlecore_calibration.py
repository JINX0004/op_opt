#!/usr/bin/env python3
"""Fit exploratory atomic Fixpipe and resident FP32->FP16 Cast increments."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROWS = [json.loads(s) for s in
        (ROOT / "singlecore_calibration_raw.jsonl").read_text().splitlines()]


def solve_linear(matrix, rhs):
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    n = len(rhs)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        if abs(divisor) < 1e-20:
            raise ValueError("singular probe design")
        a[col] = [value / divisor for value in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [a[row][-1] for row in range(n)]


def least_squares(points):
    width = len(points[0][0])
    normal = [[sum(x[i] * x[j] for x, _ in points)
               for j in range(width)] for i in range(width)]
    rhs = [sum(x[i] * y for x, y in points) for i in range(width)]
    coefficients = solve_linear(normal, rhs)
    residuals = [y - sum(c * v for c, v in zip(coefficients, x))
                 for x, y in points]
    return coefficients, statistics.median(map(abs, residuals))


def atomic_points(rounds=None):
    data = {(r["round"], r["x"], r["y"], r["mode"]): r["median_us"]
            for r in ROWS if r["group"] == "fixpipe"}
    result = []
    for round_index, packets, bytes_, mode in sorted(data):
        if mode != 5 or (rounds is not None and round_index not in rounds):
            continue
        other = data.get((round_index, packets, bytes_, 6))
        if other is None:
            continue
        # delta = intercept + packets * fixed + packets * bytes * beta
        result.append(((1.0, float(packets), float(packets * bytes_)),
                       other - data[round_index, packets, bytes_, 5]))
    return result


def store_points(rounds=None):
    return [((1.0, float(r["x"]), float(r["x"] * r["y"])), r["median_us"])
            for r in ROWS if r["group"] == "fixpipe" and r["mode"] == 5
            and (rounds is None or r["round"] in rounds)]


def cast_points(rounds=None):
    data = {(r["round"], r["x"], r["y"]): r["median_us"]
            for r in ROWS if r["group"] == "cast"}
    result = []
    for round_index, elements, repeats in sorted(data):
        if repeats != 16 or (rounds is not None and round_index not in rounds):
            continue
        later = data.get((round_index, elements, 64))
        if later is None:
            continue
        # Per extra resident cast = fixed + elements * beta.
        result.append(((1.0, float(elements)),
                       (later - data[round_index, elements, 16]) / 48.0))
    return result


def summary(points, rounds, point_builder):
    coefficients, median_abs_residual = least_squares(points)
    by_round = {}
    for round_index in rounds:
        subset = point_builder({round_index})
        if len(subset) >= len(coefficients):
            by_round[str(round_index)] = least_squares(subset)[0]
    return {"points": len(points), "coefficients": coefficients,
            "median_abs_residual_us": median_abs_residual,
            "coefficients_by_round": by_round}


rounds = sorted({r["round"] for r in ROWS})
store = summary(store_points(), rounds, store_points)
atomic = summary(atomic_points(), rounds, atomic_points)
cast = summary(cast_points(), rounds, cast_points)
store["coefficient_order"] = ["intercept_us", "plain_us_per_packet",
                              "plain_us_per_byte"]
atomic["coefficient_order"] = ["intercept_us", "extra_us_per_packet",
                                "extra_us_per_byte"]
cast["coefficient_order"] = ["cast_us_per_call", "cast_us_per_element"]
print(json.dumps({"fp32_store": store, "atomic": atomic, "cast": cast},
                 indent=2, sort_keys=True))
