#!/usr/bin/env python3
"""Fit exploratory Split-K conversion and compound-reduction probe costs."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (ROOT / "repeat_raw.jsonl").read_text().splitlines()]
by_round = defaultdict(list)
for row in rows:
    by_round[row["round"]].append(row)


def linear(points):
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    beta = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    alpha = (sy - beta * sx) / n
    return alpha, beta


def compound(points):
    # delta_us = (partials - 1) * (alpha_us + beta_us_per_element * e)
    s11 = sum(a * a for a, b, y in points)
    s12 = sum(a * b for a, b, y in points)
    s22 = sum(b * b for a, b, y in points)
    t1 = sum(a * y for a, b, y in points)
    t2 = sum(b * y for a, b, y in points)
    det = s11 * s22 - s12 * s12
    return (t1 * s22 - t2 * s12) / det, (t2 * s11 - t1 * s12) / det


result = {"rounds": {}, "point_count": len(rows)}
for round_index, group in sorted(by_round.items()):
    lookup = {(r["group"], r["elements_per_aiv"], r["partials"]): r
              for r in group}
    conversion = [(r["m"] * r["k"] / 1e6, r["median_us"])
                  for r in group if r["group"] == "convert"]
    if len(conversion) < 3:
        continue
    conv_intercept, conv_beta = linear(conversion)
    source_conversion = [(r["m"] * r["k"] / 1e6, r["median_us"])
                         for r in group if r["group"] == "convert_source"]
    source_intercept, source_beta = linear(source_conversion) if len(
        source_conversion) >= 3 else (None, None)
    reduction = []
    for r in group:
        if r["group"] != "reduce" or r["partials"] == 1:
            continue
        base = lookup.get(("reduce", r["elements_per_aiv"], 1))
        if base is None:
            continue
        a = r["partials"] - 1
        e = r["elements_per_aiv"]
        reduction.append((a, a * e, r["median_us"] - base["median_us"]))
    reduce_fixed, reduce_beta = compound(reduction)
    vector = []
    for r in group:
        if r["group"] != "vector" or r["partials"] == 1:
            continue
        base = lookup.get(("vector", r["elements_per_aiv"], 1))
        if base is None:
            continue
        a = r["partials"] - 1
        vector.append((a, a * r["elements_per_aiv"],
                       r["median_us"] - base["median_us"]))
    vector_fixed, vector_beta = compound(vector) if len(vector) >= 2 else (None, None)
    result["rounds"][str(round_index)] = {
        "conversion_intercept_us": conv_intercept,
        "conversion_rate_elements_per_us": 1e6 / conv_beta,
        "conversion_fit_us_per_million_elements": conv_beta,
        "reduction_extra_partial_fixed_us": reduce_fixed,
        "reduction_extra_partial_us_per_element_per_aiv": reduce_beta,
        "reduction_compound_elements_per_us_per_aiv": (
            1 / reduce_beta if reduce_beta > 0 else None),
        "conversion_points": len(conversion),
        "source_conversion_intercept_us": source_intercept,
        "source_conversion_rate_elements_per_us": (
            1e6 / source_beta if source_beta and source_beta > 0 else None),
        "source_conversion_points": len(source_conversion),
        "reduction_points": len(reduction),
        "vector_add_fixed_us_per_call": vector_fixed,
        "vector_add_us_per_element_per_aiv": vector_beta,
        "vector_add_elements_per_us_per_aiv": (
            1 / vector_beta if vector_beta and vector_beta > 0 else None),
        "vector_points": len(vector),
        "launch_us": lookup.get(("launch", 256, 1), {}).get("median_us"),
    }

for field in ("conversion_rate_elements_per_us",
              "conversion_intercept_us",
              "source_conversion_rate_elements_per_us",
              "reduction_extra_partial_fixed_us",
              "reduction_extra_partial_us_per_element_per_aiv",
              "vector_add_fixed_us_per_call",
              "vector_add_us_per_element_per_aiv"):
    values = [r[field] for r in result["rounds"].values() if r[field] is not None]
    if values:
        result[field + "_median"] = statistics.median(values)
        result[field + "_range"] = [min(values), max(values)]

(ROOT / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
