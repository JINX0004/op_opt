#!/usr/bin/env python3
"""Paired source BASE K-tile probe for FP32 16x16x65408 on 910B3."""

import importlib.util
import json
import statistics
from pathlib import Path


ROOT = Path("/home/xianfei/solvebase_bench_20261025")
SOURCE = ROOT / "fp32_100_source_family_v1/run.py"
OUT = ROOT / "cases/fp32_basek512_probe_v1"
spec = importlib.util.spec_from_file_location("fp32_source_benchmark", SOURCE)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

bench.OUT = OUT
bench.PACKAGES["proposed"] = ROOT / "ops_nn_fp32_basek512_probe_package"
bench.SOURCES["proposed"] = str(ROOT / "fp32_basek512_probe_source/matmul/mat_mul_v3/")
original_environment = bench.environment


def environment(arm, model_mode, log_level):
    env = original_environment(arm, model_mode, log_level)
    if arm == "research_default":
        env["MMV3_RESEARCH_SPLITK_FAMILY"] = "base"
    if arm == "proposed":
        env["MMV3_FP32_BASEK512_PROBE"] = "1"
    return env


bench.environment = environment
shape = [16, 16, 65408]
device = 4
OUT.mkdir(parents=True, exist_ok=True)
audits = {arm: bench.audit(0, shape, device, arm, 0)
          for arm in ("master", "research_default", "proposed")}
for arm, audit in audits.items():
    t = audit["tiling"]
    if audit["event"]["max_relative_error"] > .02:
        raise RuntimeError(f"incorrect output: {arm}")
    if arm != "master" and t["tilingEnableSplitCore"] != 0:
        raise RuntimeError(f"not BASE: {arm}: {t}")
    print(arm, "mode", t["tilingEnableSplitCore"],
          "base", (t["baseM"], t["baseN"], t["baseK"]),
          "steps", (t["stepKa"], t["stepKb"]),
          "core", (t["singleCoreM"], t["singleCoreN"], t["singleCoreK"]),
          "error", audit["event"]["max_relative_error"], flush=True)

if audits["research_default"]["tiling"]["baseK"] != 32:
    raise RuntimeError("source BASE reference tile changed")
if audits["proposed"]["tiling"]["baseK"] != 512:
    raise RuntimeError("proposed BASE K tile was not emitted")

timings = {arm: [] for arm in audits}
for round_index, order in enumerate((("master", "research_default", "proposed"),
                                     ("proposed", "research_default", "master"))):
    for arm in order:
        result = bench.profile(0, shape, device, arm, 0, round_index, audits[arm])
        value = result["profile"]["median_us"]
        timings[arm].append(value)
        print("round", round_index, arm, value, "us", flush=True)

summary = {"shape": shape, "device": device,
           "audits": {arm: a["tiling"] for arm, a in audits.items()},
           "rounds_us": timings,
           "median_us": {arm: statistics.median(values)
                         for arm, values in timings.items()}}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("summary", summary["median_us"], flush=True)
