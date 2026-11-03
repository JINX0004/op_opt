#!/usr/bin/env python3
"""Summarize incremental mode-6 A/B packet service from 16->64 copies."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = [json.loads(s) for s in (ROOT / "copy_raw.jsonl").read_text().splitlines()]
above_rows = [json.loads(s) for s in
              (ROOT / "copy_above_l2_288_raw.jsonl").read_text().splitlines()]
lookup = {(r["round"], r["regime"], r["operand"], r["k_packet"],
           r["copies"]): r["median_us"] for r in rows}
result = {"points": len(rows), "paths": {}}
for regime in ("l2", "above_l2"):
    for operand in ("A", "B"):
        path = f"{'near_l2' if regime == 'above_l2' else regime}_{operand}"
        by_size = {}
        for k_packet in (384, 512):
            slopes = []
            for round_index in (0, 1, 2):
                before = lookup.get((round_index, regime, operand, k_packet, 16))
                after = lookup.get((round_index, regime, operand, k_packet, 64))
                if before is not None and after is not None:
                    slopes.append((after - before) / 48.0)
            by_size[str(k_packet)] = {"median_us_per_packet": statistics.median(slopes),
                                      "round_slopes": slopes,
                                      "packet_bytes": 2 * 128 * k_packet}
        small = by_size["384"]["median_us_per_packet"]
        large = by_size["512"]["median_us_per_packet"]
        beta = (large - small) / (2 * 128 * (512 - 384))
        fixed = small - beta * 2 * 128 * 384
        result["paths"][path] = {"by_size": by_size,
                                  "linear_extra_us_per_byte": beta,
                                  "linear_fixed_us_per_packet": fixed,
                                  "linear_rate_bytes_per_us": 1 / beta if beta > 0 else None}
result["above_l2_288"] = {}
above_lookup = {(r["round"], r["operand"], r["k_packet"], r["copies"]):
                r["median_us"] for r in above_rows}
for operand in ("A", "B"):
    for k_packet in (384, 512):
        name = f"{operand}_{k_packet}"
        values = {str(copies): [above_lookup[(round_index, operand, k_packet, copies)]
                                for round_index in (0, 1)]
                  for copies in (128, 192, 256)}
        late_slopes = [(values["256"][i] - values["192"][i]) / 64.0
                       for i in (0, 1)]
        result["above_l2_288"][name] = {
            "task_us_by_copies": values,
            "late_us_per_packet_by_round": late_slopes,
            "late_us_per_packet_median": statistics.median(late_slopes)}
result["above_l2_288_points"] = len(above_rows)
print(json.dumps(result, indent=2, sort_keys=True))
