#!/usr/bin/env python3
"""Diagnose whether independently probed A/B MTE2 services resolve a tile tie."""

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
MIB = 1048576


def solve(matrix, vector):
    a = [list(row) + [value] for row, value in zip(matrix, vector)]
    for col in range(len(vector)):
        pivot = max(range(col, len(vector)), key=lambda i: abs(a[i][col]))
        if abs(a[pivot][col]) < 1e-10:
            raise ValueError("rank-deficient probe design")
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        a[col] = [v / divisor for v in a[col]]
        for row in range(len(vector)):
            if row == col:
                continue
            multiple = a[row][col]
            a[row] = [x - multiple * y for x, y in zip(a[row], a[col])]
    return [row[-1] for row in a]


def fit(rows):
    # The kernel launch/loop overhead is the intercept. The two fitted
    # slopes correspond to serial MTE2 packets per AIC and aggregate bytes
    # across 20 AICs. They are descriptive unless held-out validation passes.
    design = [[1.0, row["copies"], row["requested_bytes_per_core"] / MIB]
              for row in rows]
    times = [row["median_kernel_us"] for row in rows]
    gram = [[sum(x[i] * x[j] for x in design) for j in range(3)]
            for i in range(3)]
    rhs = [sum(x[i] * y for x, y in zip(design, times))
           for i in range(3)]
    intercept, packet_us, us_per_core_mib = solve(gram, rhs)
    residuals = [time - sum(a * b for a, b in zip(x, (intercept, packet_us,
                                                       us_per_core_mib)))
                 for x, time in zip(design, times)]
    return {
        "intercept_us": intercept, "packet_us_per_aic": packet_us,
        "us_per_core_mib": us_per_core_mib,
        "aggregate_rate_bytes_per_us":
            20 * MIB / us_per_core_mib if us_per_core_mib > 0 else None,
        "median_abs_residual_us": statistics.median(map(abs, residuals)),
        "max_abs_residual_us": max(map(abs, residuals)),
        "nonphysical_slope": packet_us < 0 or us_per_core_mib <= 0,
    }


def main() -> None:
    rows = [json.loads(line) for line in
            (HERE / "directional_read_raw.jsonl").read_text().splitlines()
            if line.strip()]
    keys = [(r["round"], r["operand"], r["tile_axis"], r["copies"])
            for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate probe point")
    rounds = sorted({r["round"] for r in rows})
    result = {"scope": "20-AIC, 2048x2048x4096 FP16 ND operand-shaped MTE2; "
                       "L2-hot input allocation; no whole-MatMul fit",
              "points": len(rows), "rounds": rounds, "fits": {}}
    for operand in ("A", "B"):
        operand_rows = [r for r in rows if r["operand"] == operand]
        if len(operand_rows) < 6:
            raise ValueError(f"too few {operand} points")
        pooled = fit(operand_rows)
        per_round = [fit([r for r in operand_rows if r["round"] == round_index])
                     for round_index in rounds]
        result["fits"][operand] = {"pooled": pooled, "by_round": per_round}
    a = result["fits"]["A"]["pooled"]
    b = result["fits"]["B"]["pooled"]
    if a["aggregate_rate_bytes_per_us"] and b["aggregate_rate_bytes_per_us"]:
        # The prior MatMul pair has 128 MiB A + 256 MiB B for 128x256,
        # versus 256 MiB A + 128 MiB B for 256x128. Packet counts tie.
        default_read_us = (128 * MIB / a["aggregate_rate_bytes_per_us"] +
                           256 * MIB / b["aggregate_rate_bytes_per_us"])
        swapped_read_us = (256 * MIB / a["aggregate_rate_bytes_per_us"] +
                           128 * MIB / b["aggregate_rate_bytes_per_us"])
        result["2048_square_read_component"] = {
            "default_128x256_us": default_read_us,
            "swapped_256x128_us": swapped_read_us,
            "swapped_minus_default_us": swapped_read_us - default_read_us,
            "read_only_preference": "128x256" if default_read_us < swapped_read_us
                                   else "256x128" if swapped_read_us < default_read_us
                                   else "tie",
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
