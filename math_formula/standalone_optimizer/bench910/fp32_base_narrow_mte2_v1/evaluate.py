#!/usr/bin/env python3
"""Build the opt-in scorer and compare audited source-BASE profiles."""

import csv
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "fp32_mode2_v1/remote_selector/model"
SCORER = HERE.parent / "fp32_base_formula_v2/score_base_mte2.cpp"
DATA = (
    ("calibration_and_first_holdout", "records.json"),
    ("broad_holdout", "broad_holdout_records.json"),
    ("unaligned_development_and_holdout", "unaligned_records.json"),
    ("final_development", "final_holdout_records.json"),
    ("independent_alignment_holdout", "alignment_holdout_records.json"),
)


def main():
    observations = []
    for group, name in DATA:
        for record in json.loads((HERE / name).read_text()):
            # The source can emit a padded 256-column tile for N=64, but the
            # current PlanFinalizer does not represent that edge case.
            if record["tiling"]["baseN"] > 128:
                continue
            observations.append((group, record))
    with tempfile.TemporaryDirectory() as tmp:
        executable = Path(tmp) / "score_base_mte2"
        subprocess.run([
            "g++", "-std=c++17", "-O2", "-I", str(MODEL), str(SCORER),
            str(MODEL / "mip_solver.cpp"), str(MODEL / "plan_finalizer.cpp"),
            "-o", str(executable)], check=True)
        inputs = []
        for _, record in observations:
            tile = record["tiling"]
            inputs.append(" ".join(map(str, (*record["shape"],
                tile["estimated_nonidle_aics"], tile["singleCoreM"],
                tile["singleCoreN"], tile["singleCoreK"], tile["baseM"],
                tile["baseN"], tile["baseK"]))))
        outputs = []
        supported = []
        for observation, source_line in zip(observations, inputs):
            result = subprocess.run([str(executable)], input=source_line + "\n",
                                    text=True, capture_output=True)
            if result.returncode:
                print("skip unsupported source tile", observation[1]["shape"],
                      result.stderr.strip())
                continue
            supported.append(observation)
            outputs.append(result.stdout.splitlines()[1])
    rows = []
    for (group, record), line in zip(supported, outputs):
        m, n, k, baseline, transfer = map(float, line.split(","))
        actual = record["profile"]["median_us"]
        row = {
            "group": group, "set": record["set"], "m": int(m), "n": int(n),
            "k": int(k), "base_m": record["tiling"]["baseM"],
            "base_n": record["tiling"]["baseN"],
            "base_k": record["tiling"]["baseK"],
            "a_pitch_bytes": int(4 * k),
            "a_footprint_mib": round(4 * m * k / 2**20, 3),
            "measured_us": round(actual, 4), "baseline_us": round(baseline, 4),
            "transfer_us": round(transfer, 4),
            "baseline_ape_pct": round(100 * abs(baseline / actual - 1), 3),
            "transfer_ape_pct": round(100 * abs(transfer / actual - 1), 3),
            "mte2_busy_pct": round(100 * record["pipe"]["aic_mte2_ratio"], 3),
            "changed": transfer > baseline + 1e-3,
        }
        rows.append(row)
    with (HERE / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for group in sorted(set(row["group"] for row in rows)):
        subset = [row for row in rows if row["group"] == group]
        changed = [row for row in subset if row["changed"]]
        groups[group] = {
            "cases": len(subset), "changed": len(changed),
            "baseline_mape_pct": round(sum(r["baseline_ape_pct"] for r in subset) / len(subset), 3),
            "transfer_mape_pct": round(sum(r["transfer_ape_pct"] for r in subset) / len(subset), 3),
            "changed_baseline_mape_pct": round(sum(r["baseline_ape_pct"] for r in changed) / len(changed), 3) if changed else None,
            "changed_transfer_mape_pct": round(sum(r["transfer_ape_pct"] for r in changed) / len(changed), 3) if changed else None,
        }
    (HERE / "comparison_summary.json").write_text(json.dumps(groups, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
