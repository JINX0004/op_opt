#!/usr/bin/env python3
"""Collect AIC pipeline and memory metrics for the proposed FP32 tile."""

import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path


ROOT = Path("/home/xianfei/solvebase_bench_20261025")
SOURCE = ROOT / "fp32_100_source_family_v1/run.py"
OUT = ROOT / "cases/fp32_4096x64_tile_probe_v1/counters"
spec = importlib.util.spec_from_file_location("fp32_source_benchmark", SOURCE)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.PACKAGES["proposed"] = ROOT / "ops_nn_fp32_4096x64_tile_probe_package"
original_environment = bench.environment


def environment(arm, model_mode, log_level):
    env = original_environment(arm, model_mode, log_level)
    if arm == "proposed":
        env["MMV3_FP32_4096X64_TILE_PROBE"] = "1"
    return env


def run(arm, metric):
    dest = OUT / f"{arm}_{metric}"
    dest.mkdir(parents=True, exist_ok=True)
    result_path = dest / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    bench.require_idle(4, dest, "before")
    env = environment(arm, 0, 3)
    cmd = ["msprof", "--output=" + str(dest / "msprof"),
           "--task-time=on", "--ai-core=on", "--aic-metrics=" + metric,
           "env", str(ROOT / "fp32_bench"), "--api", "v3cached",
           "--device", "4", "--m", "4096", "--n", "64", "--k", "27456",
           "--warmup", "3", "--batches", "1", "--repetitions", "3"]
    with (dest / "stdout.log").open("w") as stdout, \
         (dest / "stderr.log").open("w") as stderr:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                              stderr=stderr, timeout=1200)
    bench.require_idle(4, dest, "after")
    if proc.returncode:
        raise RuntimeError((dest / "stderr.log").read_text()[-2000:])
    event = bench.harness_event(dest / "stdout.log")
    if event["dtype"] != "fp32" or event["max_relative_error"] > .02:
        raise RuntimeError(event)
    files = glob.glob(str(dest / "msprof/**/ai_core_op_summary.db"), recursive=True)
    if len(files) != 1:
        raise RuntimeError((dest, files))
    connection = sqlite3.connect(files[0])
    columns = [row[1] for row in connection.execute("pragma table_info(ai_core_metrics)")]
    values = connection.execute("select * from ai_core_metrics order by task_id").fetchall()
    connection.close()
    if len(values) < 3:
        raise RuntimeError((dest, len(values)))
    recent = values[-3:]
    metrics = {column: statistics.median(float(row[index]) for row in recent)
               for index, column in enumerate(columns)
               if column not in ("task_id", "stream_id", "subtask_id", "batch_id")
               and all(row[index] is not None for row in recent)}
    result = {"arm": arm, "metric": metric, "metrics": metrics,
              "event": event, "msprof_path": str(dest / "msprof")}
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    for arm, metric in (("proposed", "PipeUtilization"),
                        ("proposed", "MemoryAccess"),
                        ("proposed", "L2Cache"),
                        ("master", "PipeUtilization")):
        result = run(arm, metric)
        print(arm, metric, result["metrics"], flush=True)
