#!/usr/bin/env python3
"""Fit each source-emitted L1-step packet service, without kernel-time fitting."""

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent


def fit(rows):
    x = [row["copies"] for row in rows]
    y = [row["median_kernel_us"] for row in rows]
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / \
            sum((a - mean_x) ** 2 for a in x)
    intercept = mean_y - slope * mean_x
    errors = [b - intercept - slope * a for a, b in zip(x, y)]
    return {"intercept_us": intercept, "us_per_128kib_packet": slope,
            "effective_rate_bytes_per_us": 131072 / slope if slope > 0 else None,
            "median_abs_residual_us": statistics.median(map(abs, errors)),
            "max_abs_residual_us": max(map(abs, errors))}


def fit_round_medians(rows):
    # One whole-run profiler excursion should not determine a slope. Take
    # the median of three independent run medians at each copy count, then
    # fit the five predeclared copy counts with equal weight.
    copies = sorted({r["copies"] for r in rows})
    compressed = [{"copies": count,
                   "median_kernel_us": statistics.median(
                       r["median_kernel_us"] for r in rows
                       if r["copies"] == count)} for count in copies]
    return fit(compressed)


def main():
    rows = [json.loads(line) for line in
            (HERE / "l1_step_read_raw.jsonl").read_text().splitlines()
            if line.strip()]
    labels = [r["label"] for r in rows]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate probe point")
    rounds = sorted({r["round"] for r in rows})
    result = {"scope": "20-AIC, L2-hot, 2048x2048x4096 source-emitted "
                       "128KiB A/B L1-step ND->NZ transfers",
              "points": len(rows), "rounds": rounds, "cases": {}}
    for orientation, operand, axis, span in (
            ("default", "A", 128, 512), ("default", "B", 256, 256),
            ("swapped", "A", 256, 256), ("swapped", "B", 128, 512)):
        group = [r for r in rows if r["orientation"] == orientation and
                 r["operand"] == operand]
        if len(group) != 5 * len(rounds):
            raise ValueError(f"incomplete {orientation} {operand} probe")
        case = {"operand": operand, "axis": axis, "k_span": span,
                "pooled": fit(group),
                "round_medians": fit_round_medians(group),
                "by_round": [fit([r for r in group if r["round"] == round])
                             for round in rounds]}
        result["cases"][f"{orientation}_{operand}"] = case
    per_orientation = {}
    for orientation in ("default", "swapped"):
        # Each output tile traverses all K once. With 128 output tiles and
        # 20 AICs, this is the mean service demand per occupied AIC. It is
        # not a makespan estimate because A/B overlap and task assignment
        # imbalance are unresolved.
        a = result["cases"][f"{orientation}_A"]
        b = result["cases"][f"{orientation}_B"]
        a_copies = 128 * (4096 / a["k_span"]) / 20
        b_copies = 128 * (4096 / b["k_span"]) / 20
        a_us = a_copies * a["round_medians"]["us_per_128kib_packet"]
        b_us = b_copies * b["round_medians"]["us_per_128kib_packet"]
        per_orientation[orientation] = {"a_requested_steps_per_aic": a_copies,
                                        "b_requested_steps_per_aic": b_copies,
                                        "serial_read_service_us": a_us + b_us,
                                        "a_service_us": a_us, "b_service_us": b_us}
    result["orientation_service"] = per_orientation
    result["swapped_minus_default_serial_service_us"] = (
        per_orientation["swapped"]["serial_read_service_us"] -
        per_orientation["default"]["serial_read_service_us"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
