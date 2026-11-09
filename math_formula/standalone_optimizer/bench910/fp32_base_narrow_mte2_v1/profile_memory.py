#!/usr/bin/env python3
"""Compare source-BASE memory counters for narrow FP32 output widths."""

import concurrent.futures
import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path

ROOT = Path("/home/xianfei/solvebase_bench_20261025")
OUT = ROOT / "cases/fp32_base_narrow_mte2_v1/memory"
SOURCE = ROOT / "fp32_100_source_family_v1/run.py"
CASES = ((4096, 64, 32640), (4096, 96, 32640), (4096, 128, 32640))


def run_case(item):
    index, shape = item
    device = 4 + index
    spec = importlib.util.spec_from_file_location("fp32_source_benchmark", SOURCE)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.OUT = OUT
    runner.PACKAGES["proposed"] = ROOT / "ops_nn_fp32_forced_family_package"
    runner.SOURCES["proposed"] = str(ROOT / "ops_nn_source_build/matmul/mat_mul_v3/")
    results = []
    for metric in ("MemoryAccess", "L2Cache"):
        dest = OUT / f"device{device}" / f"{index:03d}_{runner.label(shape)}" / metric
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        runner.require_idle(device, dest, "before")
        env = runner.environment("proposed", 0, 3)
        cmd = ["msprof", "--output=" + str(dest / "msprof"),
               "--task-time=on", "--ai-core=on", "--aic-metrics=" + metric,
               "env", str(ROOT / "fp32_bench"), "--api", "v3cached",
               "--device", str(device), "--m", str(shape[0]), "--n", str(shape[1]),
               "--k", str(shape[2]), "--warmup", "3", "--batches", "1",
               "--repetitions", "3"]
        with (dest / "stdout.log").open("w") as stdout, (dest / "stderr.log").open("w") as stderr:
            proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                                  stderr=stderr, timeout=1200)
        runner.require_idle(device, dest, "after")
        if proc.returncode:
            raise RuntimeError((dest / "stderr.log").read_text()[-2000:])
        event = runner.harness_event(dest / "stdout.log")
        if event["dtype"] != "fp32" or event["max_relative_error"] > .02:
            raise RuntimeError(event)
        files = glob.glob(str(dest / "msprof/**/ai_core_op_summary.db"), recursive=True)
        if len(files) != 1:
            raise RuntimeError((dest, files))
        connection = sqlite3.connect(files[0])
        columns = [row[1] for row in connection.execute("pragma table_info(ai_core_metrics)")]
        values = connection.execute("select * from ai_core_metrics order by task_id").fetchall()
        connection.close()
        recent = values[-3:]
        metrics = {column: statistics.median(float(row[j]) for row in recent)
                   for j, column in enumerate(columns)
                   if column not in ("task_id", "stream_id", "subtask_id", "batch_id")
                   and all(row[j] is not None for row in recent)}
        result = {"shape": shape, "device": device, "metric": metric,
                  "event": event, "metrics": metrics}
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
        print("done", shape, metric, flush=True)
    return results


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        groups = list(pool.map(run_case, enumerate(CASES)))
    (OUT / "records.json").write_text(json.dumps([x for g in groups for x in g], indent=2) + "\n")
