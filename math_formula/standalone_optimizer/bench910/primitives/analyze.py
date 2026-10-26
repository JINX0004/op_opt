#!/usr/bin/env python3
"""Fit probe slopes and evaluate the unchanged reduced SolveBase equation.

The fit is descriptive. A device profile is accepted only after repeated
independent runs and held-out tiling-rank validation. This script reports
negative slopes, nonmonotonic points, and validation residuals explicitly.
"""
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name):
    path = HERE / name
    return [json.loads(line) for line in path.read_text().splitlines()]


def solve_linear(matrix, vector):
    n = len(vector)
    a = [list(row) + [v] for row, v in zip(matrix, vector)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(a[i][j]))
        if abs(a[pivot][j]) < 1e-12:
            raise ValueError("rank-deficient probe design")
        a[j], a[pivot] = a[pivot], a[j]
        scale = a[j][j]
        a[j] = [x / scale for x in a[j]]
        for i in range(n):
            if i != j:
                scale = a[i][j]
                a[i] = [x - scale * y for x, y in zip(a[i], a[j])]
    return [row[-1] for row in a]


def fit(rows, features):
    # Scale work axes to MiB and MFLOP for numerical conditioning. The
    # returned rates are converted back to native bytes/µs and FLOP/µs.
    design = [[1.0] + [f(row) for f in features] for row in rows]
    y = [row["median_kernel_us"] for row in rows]
    size = len(design[0])
    gram = [[sum(x[i] * x[j] for x in design) for j in range(size)]
            for i in range(size)]
    rhs = [sum(x[i] * t for x, t in zip(design, y)) for i in range(size)]
    coefficients = solve_linear(gram, rhs)
    predictions = [sum(a * b for a, b in zip(x, coefficients)) for x in design]
    residuals = [a - b for a, b in zip(y, predictions)]
    return {"coefficients": coefficients,
            "median_abs_residual_us": statistics.median(map(abs, residuals)),
            "max_abs_residual_us": max(map(abs, residuals)),
            "residuals": dict(zip((row["label"] for row in rows), residuals)),
            "negative_coefficient": any(c < 0 for c in coefficients[1:])}


def main():
    launch_read = load("launch_read_synced_raw.jsonl")
    cube_write = load("cube_write_raw.jsonl")
    nd2nz_read = load("nd2nz_read_corrected_raw.jsonl")
    results = {"scope": "FP16 910B3, L2-hot primitive pilot; not an accepted model profile",
               "raw_count": len(launch_read) + len(cube_write) + len(nd2nz_read),
               "fits": {}}
    launch = [r for r in launch_read if r["mode"] == 0]
    results["launch_by_cores_us"] = {
        str(r["cores"]): r["median_kernel_us"] for r in launch}
    for cores in (1, 20):
        read = [r for r in launch_read if r["mode"] == 1 and r["cores"] == cores]
        rf = fit(read, [lambda r: r["bytes_per_core"] / 1048576,
                        lambda r: r["packets"]])
        rf["aggregate_rate_bytes_per_us"] = cores * 1048576 / rf["coefficients"][1]
        rf["per_packet_us"] = rf["coefficients"][2]
        results["fits"][f"flat_read_{cores}"] = rf

        matching_read = [r for r in nd2nz_read if r["mode"] == 4 and r["cores"] == cores]
        mr = fit(matching_read, [lambda r: r["bytes_per_core"] / 1048576,
                                 lambda r: r["packets"]])
        mr["aggregate_rate_bytes_per_us"] = cores * 1048576 / mr["coefficients"][1]
        mr["per_packet_us"] = mr["coefficients"][2]
        results["fits"][f"read_{cores}"] = mr

        cube = [r for r in cube_write if r["mode"] == 2 and r["cores"] == cores]
        cf = fit(cube, [lambda r: 2 * 128**3 * r["packets"] / 1000000])
        cf["per_core_rate_flop_per_us"] = 1000000 / cf["coefficients"][1]
        results["fits"][f"cube_{cores}"] = cf

        write = [r for r in cube_write if r["mode"] == 3 and r["cores"] == cores]
        wf = fit(write, [lambda r: r["bytes_per_core"] / 1048576,
                         lambda r: r["packets"]])
        wf["aggregate_rate_bytes_per_us"] = cores * 1048576 / wf["coefficients"][1]
        wf["per_packet_us"] = wf["coefficients"][2]
        results["fits"][f"write_{cores}"] = wf

    # These cases were measured before probe fitting, so they are not part
    # of the regression. They are a shape check; no same-shape tiling rank
    # can be computed until spatial overrides are measured.
    cases = json.loads((HERE.parent / "source_matched_results.json").read_text())["cases"]
    rc = results["fits"]["cube_20"]["per_core_rate_flop_per_us"]
    rr = results["fits"]["read_20"]["aggregate_rate_bytes_per_us"]
    rw = results["fits"]["write_20"]["aggregate_rate_bytes_per_us"]
    lr = results["fits"]["read_20"]["per_packet_us"]
    lw = results["fits"]["write_20"]["per_packet_us"]
    launch_us = results["launch_by_cores_us"]["20"]
    def predict(case, wave=False):
        c = case["estimated_nonidle_aics"]
        cube_us = case["model_F_flops"] / (c * rc)
        packets = 2 * case["model_S_spatial_packets"] * case["model_P_k_packets"]
        output_packets = case["output_tiles"]
        if wave:
            spatial_per_output = (math.ceil(case["singleCoreM"] / case["baseM"]) *
                                  math.ceil(case["singleCoreN"] / case["baseN"]))
            output_packets = math.ceil(case["output_tiles"] / c)
            packets = 2 * output_packets * spatial_per_output * case["model_P_k_packets"]
        memory_us = (case["model_D_read_bytes"] / rr +
                     case["model_output_bytes"] / rw +
                     packets * lr + output_packets * lw)
        return launch_us + max(cube_us, memory_us), cube_us, memory_us, packets

    results["source_matched_shape_check"] = []
    for case in cases:
        if case["forced_used_cores"] is not None:
            continue
        estimated, cube_us, memory_us, packets = predict(case)
        wave_estimated, _, _, wave_packets = predict(case, wave=True)
        results["source_matched_shape_check"].append({
            "shape": case["shape"], "observed_us": case["kernel_median_us"],
            "predicted_reduced_us": estimated,
            "predicted_wave_us": wave_estimated,
            "relative_error": (estimated / case["kernel_median_us"] - 1),
            "cube_branch_us": cube_us, "memory_branch_us": memory_us,
            "read_packets_global": packets, "read_packets_wave": wave_packets})
    controlled = json.loads((HERE.parent / "spatial_results.json").read_text())["cases"]
    by_shape = {}
    for case in controlled:
        case = dict(case)
        case["predicted_reduced_us"] = predict(case)[0]
        case["predicted_wave_us"] = predict(case, wave=True)[0]
        by_shape.setdefault(tuple(case["shape"]), []).append(case)
    results["spatial_rankings"] = []
    for shape, group in by_shape.items():
        best = min(group, key=lambda x: x["kernel_median_us"])
        reduced = min(group, key=lambda x: x["predicted_reduced_us"])
        wave = min(group, key=lambda x: x["predicted_wave_us"])
        results["spatial_rankings"].append({
            "shape": shape, "observed_best": best["case_id"],
            "reduced_choice": reduced["case_id"],
            "wave_choice": wave["case_id"],
            "reduced_regret": reduced["kernel_median_us"] / best["kernel_median_us"] - 1,
            "wave_regret": wave["kernel_median_us"] / best["kernel_median_us"] - 1,
            "candidates": [{"case_id": c["case_id"], "baseM": c["baseM"],
                            "active_cores": c["estimated_nonidle_aics"],
                            "observed_us": c["kernel_median_us"],
                            "predicted_reduced_us": c["predicted_reduced_us"],
                            "predicted_wave_us": c["predicted_wave_us"]}
                           for c in group]})
    spatial_repeat_path = HERE.parent / "spatial_repeat_raw.jsonl"
    if spatial_repeat_path.exists():
        repeated = [json.loads(line) for line in
                    spatial_repeat_path.read_text().splitlines()]
        times_by_case = {}
        for row in repeated:
            times_by_case.setdefault(row["case_id"], []).append(
                row["case"]["kernel_median_us"])
        median_by_case = {name: statistics.median(times)
                          for name, times in times_by_case.items()}
        results["spatial_repeat_rankings"] = []
        for ranking in results["spatial_rankings"]:
            candidates = [c for c in ranking["candidates"]
                          if c["case_id"] in median_by_case]
            if len(candidates) < 2:
                continue
            best = min(candidates, key=lambda c: median_by_case[c["case_id"]])
            reduced = min(candidates, key=lambda c: c["predicted_reduced_us"])
            wave = min(candidates, key=lambda c: c["predicted_wave_us"])
            best_time = median_by_case[best["case_id"]]
            results["spatial_repeat_rankings"].append({
                "shape": ranking["shape"],
                "observed_best_median": best["case_id"],
                "observed_best_median_us": best_time,
                "reduced_choice": reduced["case_id"],
                "wave_choice": wave["case_id"],
                "reduced_regret_on_repeat_medians":
                    median_by_case[reduced["case_id"]] / best_time - 1,
                "wave_regret_on_repeat_medians":
                    median_by_case[wave["case_id"]] / best_time - 1,
                "candidate_medians_us": {c["case_id"]:
                                         median_by_case[c["case_id"]]
                                         for c in candidates},
            })
    repeat_path = HERE / "repeat_selected_corrected_raw.jsonl"
    if repeat_path.exists():
        repeated = load("repeat_selected_corrected_raw.jsonl")
        by_label = {}
        by_round = {}
        for row in repeated:
            by_label.setdefault(row["label"], []).append(row["median_kernel_us"])
            by_round[(row["round"], row["label"])] = row["median_kernel_us"]
        results["repeat_selected_summary"] = {
            label: {"independent_runs": len(times),
                    "median_us": statistics.median(times),
                    "min_us": min(times), "max_us": max(times)}
            for label, times in by_label.items()}

        def paired(low, high, transform):
            values = [transform(by_round[(round_index, low)],
                                by_round[(round_index, high)])
                      for round_index in range(5)
                      if (round_index, low) in by_round and
                         (round_index, high) in by_round]
            return {"paired_rounds": len(values),
                    "values": values,
                    "median": statistics.median(values),
                    "min": min(values), "max": max(values)} if values else None

        results["repeat_derived_slopes"] = {
            "cube_flop_per_us_per_aic": paired(
                "m2_c20_p8_b32768", "m2_c20_p64_b32768",
                lambda a, b: (64 - 8) * 2 * 128**3 / (b - a)),
            "nd2nz_read_packet_us": paired(
                "m4_c20_p16_b16384", "m4_c20_p64_b4096",
                lambda a, b: (b - a) / (64 - 16)),
            "nd2nz_read_aggregate_bytes_per_us": paired(
                "m4_c20_p16_b4096", "m4_c20_p16_b32768",
                lambda a, b: 20 * 16 * (32768 - 4096) / (b - a)),
            "flat_read_packet_us": paired(
                "m1_c20_p16_b16384", "m1_c20_p64_b4096",
                lambda a, b: (b - a) / (64 - 16)),
            "fixpipe_write_packet_us": paired(
                "m3_c20_p16_b16384", "m3_c20_p64_b4096",
                lambda a, b: (b - a) / (64 - 16)),
            "fixpipe_write_aggregate_bytes_per_us": paired(
                "m3_c20_p16_b4096", "m3_c20_p16_b32768",
                lambda a, b: 20 * 16 * (32768 - 4096) / (b - a)),
        }
    hbm_path = HERE / "hbm_read_raw.jsonl"
    if hbm_path.exists():
        large = load("hbm_read_raw.jsonl")
        by_point = {r["label"]: r for r in large}
        summary = [{"packets": r["packets"],
                    "packet_bytes": r["packet_bytes"],
                    "total_working_set_mib": 20 * r["bytes_per_core"] / 1048576,
                    "median_kernel_us": r["median_kernel_us"]}
                   for r in large]
        a = by_point.get("m4_c20_p64_b196608")
        b = by_point.get("m4_c20_p64_b262144")
        if a and b and b["median_kernel_us"] > a["median_kernel_us"]:
            summary.append({"comparison": "fixed_packets_above_l2",
                            "differential_aggregate_bytes_per_us":
                            20 * (b["bytes_per_core"] - a["bytes_per_core"]) /
                            (b["median_kernel_us"] - a["median_kernel_us"])})
        results["hbm_probe_summary"] = summary
    hbm_repeat_path = HERE / "hbm_repeat_raw.jsonl"
    if hbm_repeat_path.exists():
        repeated = load("hbm_repeat_raw.jsonl")
        by_round = {(r["round"], r["label"]): r["median_kernel_us"]
                    for r in repeated}
        rates = []
        packet_slopes = []
        for round_index in range(5):
            try:
                same_packet_small = by_round[(round_index, "m4_c20_p64_b196608")]
                same_packet_large = by_round[(round_index, "m4_c20_p64_b262144")]
                same_bytes_fewer = by_round[(round_index, "m4_c20_p48_b262144")]
            except KeyError:
                continue
            if same_packet_large > same_packet_small:
                rates.append(20 * 64 * (262144 - 196608) /
                             (same_packet_large - same_packet_small))
            packet_slopes.append((same_packet_small - same_bytes_fewer) / 16)
        results["hbm_repeat_derived"] = {
            "aggregate_bytes_per_us_fixed_64_packets": rates,
            "rate_median": statistics.median(rates) if rates else None,
            "rate_min": min(rates) if rates else None,
            "rate_max": max(rates) if rates else None,
            "same_240mib_packet_slope_us": packet_slopes,
            "packet_slope_median": statistics.median(packet_slopes) if packet_slopes else None,
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
