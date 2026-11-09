#!/usr/bin/env python3
"""Paired FP32 4096x64x27456 proposed-BASE-tile benchmark on 910B3."""

import importlib.util
import json
import statistics
from pathlib import Path


ROOT = Path("/home/xianfei/solvebase_bench_20261025")
SOURCE = ROOT / "fp32_100_source_family_v1/run.py"
OUT = ROOT / "cases/fp32_4096x64_tile_probe_v1"
spec = importlib.util.spec_from_file_location("fp32_source_benchmark", SOURCE)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

bench.OUT = OUT
bench.PACKAGES["proposed"] = ROOT / "ops_nn_fp32_4096x64_tile_probe_package"
bench.SOURCES["proposed"] = str(ROOT / "fp32_4096x64_tile_probe_source/matmul/mat_mul_v3/")
original_environment = bench.environment


def environment(arm, model_mode, log_level):
    env = original_environment(arm, model_mode, log_level)
    if arm == "research_default":
        env["MMV3_RESEARCH_SPLITK_FAMILY"] = "base"
    if arm == "proposed":
        env["MMV3_FP32_4096X64_TILE_PROBE"] = "1"
    return env


bench.environment = environment
shape = [4096, 64, 27456]
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

reference = audits["research_default"]["tiling"]
proposal = audits["proposed"]["tiling"]
if tuple(reference[k] for k in ("baseM", "baseN", "baseK")) != (128, 128, 64):
    raise RuntimeError("source BASE reference tile changed")
if tuple(proposal[k] for k in ("baseM", "baseN", "baseK")) != (256, 64, 32):
    raise RuntimeError("proposed BASE tile was not emitted")
if tuple(proposal[k] for k in ("singleCoreM", "singleCoreN", "singleCoreK")) != (256, 64, 27456):
    raise RuntimeError("proposed core tile was not emitted")

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
           "max_relative_error": {arm: a["event"]["max_relative_error"]
                                  for arm, a in audits.items()},
           "rounds_us": timings,
           "median_us": {arm: statistics.median(values)
                         for arm, values in timings.items()}}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("summary", summary["median_us"], flush=True)
