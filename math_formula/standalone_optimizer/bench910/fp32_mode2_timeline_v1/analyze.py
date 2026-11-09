#!/usr/bin/env python3
"""Compare the independent-input event schedule with fixed-work holdouts.

Only the copy probe supplies an input duration. Full-kernel measurements are
loaded after prediction and never passed to the C++ evaluator.
"""

import argparse
import csv
import io
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
PITCHES = (48896, 49152, 49408)


def center_excess(values):
    return values[49152] - (values[48896] + values[49408]) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path, help="compiled timeline.cpp")
    args = parser.parse_args()

    holdout = json.loads((BENCH / "fp32_mode2_in_context_pitch_v1/summary.json").read_text())
    probe = json.loads((BENCH / "fp32_mode2_sync_holdout_v1/probe_records.json").read_text())
    probe_times = defaultdict(list)
    for record in probe:
        if record["sync_mode"] == 1:
            key = (record["m"], record["n"], record["pitch"])
            probe_times[key].append(record["device_elapsed_us"])

    requests = []
    for case in holdout:
        m, n, k = case["shape"]
        tile = case["tile"]
        assert k == 48384 and tile["block_dim"] == 20
        for pitch in PITCHES:
            samples = probe_times[(m, n, pitch)]
            assert len(samples) == 2, (m, n, pitch, samples)
            duration = statistics.median(samples)
            for order in ("mn", "nm"):
                requests.append(
                    f"{m}x{n} {pitch} {tile['singleCoreM']} "
                    f"{tile['singleCoreN']} {k} {tile['singleCoreK']} "
                    f"{tile['baseM']} {tile['baseN']} {tile['baseK']} "
                    f"{duration:.9f} {order}"
                )
    evaluated = subprocess.run(
        [str(args.binary)], input="\n".join(requests) + "\n",
        capture_output=True, text=True, check=True,
    )
    rows = list(csv.DictReader(io.StringIO(evaluated.stdout)))
    assert len(rows) == len(requests)

    results = []
    for case in holdout:
        m, n, _ = case["shape"]
        observed = {p: case["median_us"][str(p)] for p in PITCHES}
        for order in ("mn", "nm"):
            subset = {
                int(row["pitch"]): row for row in rows
                if row["id"] == f"{m}x{n}" and row["order"] == order
            }
            assert set(subset) == set(PITCHES)
            predicted = {p: float(subset[p]["predicted_us"]) for p in PITCHES}
            probe_by_pitch = {
                p: statistics.median(probe_times[(m, n, p)]) for p in PITCHES
            }
            for pitch in PITCHES:
                row = subset[pitch]
                critical = sum(
                    float(row[f"{stage}_critical_us"])
                    for stage in ("mte2", "l0a", "l0b", "cube", "fix")
                )
                # A critical path consists of exactly one chain; launch is
                # outside the graph. This checks the event tracing itself.
                assert abs(critical + 2.68 - predicted[pitch]) < 0.01
                results.append({
                    "shape": f"{m}x{n}x48384",
                    "pitch": pitch,
                    "order": order,
                    "probe_us": probe_by_pitch[pitch],
                    "observed_us": observed[pitch],
                    "predicted_us": predicted[pitch],
                    "error_us": predicted[pitch] - observed[pitch],
                    "observed_center_excess_us": center_excess(observed),
                    "predicted_center_excess_us": center_excess(predicted),
                    "probe_center_excess_us": center_excess(probe_by_pitch),
                    "mte2_critical_us": float(row["mte2_critical_us"]),
                    "l0a_critical_us": float(row["l0a_critical_us"]),
                    "l0b_critical_us": float(row["l0b_critical_us"]),
                    "cube_critical_us": float(row["cube_critical_us"]),
                    "fix_critical_us": float(row["fix_critical_us"]),
                })

    fieldnames = list(results[0])
    with (HERE / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary = []
    for shape in sorted(set(row["shape"] for row in results)):
        for order in ("mn", "nm"):
            r = next(row for row in results if row["shape"] == shape and row["order"] == order)
            sub = [row for row in results if row["shape"] == shape and row["order"] == order]
            summary.append({
                "shape": shape, "order": order,
                "observed_center_excess_us": r["observed_center_excess_us"],
                "predicted_center_excess_us": r["predicted_center_excess_us"],
                "probe_center_excess_us": r["probe_center_excess_us"],
                "absolute_mae_us": statistics.mean(abs(row["error_us"]) for row in sub),
                "absolute_mape_percent": 100 * statistics.mean(
                    abs(row["error_us"]) / row["observed_us"] for row in sub
                ),
            })
    (HERE / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in summary:
        print(
            f"{row['shape']:19s} {row['order']} "
            f"observed={row['observed_center_excess_us']:7.1f} "
            f"predicted={row['predicted_center_excess_us']:7.1f} "
            f"probe={row['probe_center_excess_us']:7.1f} "
            f"abs-MAPE={row['absolute_mape_percent']:5.1f}%"
        )


if __name__ == "__main__":
    main()
