#!/usr/bin/env python3
"""Source-BASE FP32 narrow-N calibration/holdout grid on 910B3."""

import concurrent.futures
import glob
import importlib.util
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path


ROOT = Path("/home/xianfei/solvebase_bench_20261025")
OUT = ROOT / "cases/fp32_base_narrow_mte2_v1"
SOURCE = ROOT / "fp32_100_source_family_v1/run.py"

# The first four shapes are calibration observations. The last four are
# holdouts; do not use their runtime or counters to choose service rates.
SHAPES = (
    (2048, 64, 27328),
    (4096, 64, 27328),
    (4096, 64, 27456),
    (2048, 128, 27328),
    (3072, 64, 27392),
    (6144, 64, 27392),
    (4096, 128, 27456),
    (3072, 128, 27456),
)
CALIBRATION_COUNT = 4


def load_benchmark():
    spec = importlib.util.spec_from_file_location("fp32_source_benchmark", SOURCE)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.OUT = OUT
    runner.PACKAGES["proposed"] = ROOT / "ops_nn_fp32_forced_family_package"
    runner.SOURCES["proposed"] = str(ROOT / "ops_nn_source_build/matmul/mat_mul_v3/")
    return runner


def process_device(device):
    runner = load_benchmark()
    records = []
    for index, shape in enumerate(SHAPES):
        if 4 + index % 4 != device:
            continue
        audit = runner.audit(index, list(shape), device, "proposed", 0)
        tiling = audit["tiling"]
        if tiling["tilingEnableSplitCore"] != 0:
            raise RuntimeError(f"not BASE: {shape}: {tiling}")
        dest = OUT / f"device{device}" / f"{index:03d}_{runner.label(shape)}" / "pipe"
        dest.mkdir(parents=True, exist_ok=True)
        result_path = dest / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            records.append(result)
            continue
        runner.require_idle(device, dest, "before")
        env = runner.environment("proposed", 0, 3)
        kind = "KERNEL_AICORE" if not tiling["nd2nzA"] and not tiling["nd2nzB"] else "MIX_AIC"
        cmd = ["msprof", "--output=" + str(dest / "msprof"),
               "--task-time=on", "--ai-core=on", "--aic-metrics=PipeUtilization",
               "env", str(ROOT / "fp32_bench"), "--api", "v3cached",
               "--device", str(device), "--m", str(shape[0]), "--n", str(shape[1]),
               "--k", str(shape[2]), "--warmup", "3", "--batches", "1",
               "--repetitions", "3"]
        with (dest / "stdout.log").open("w") as stdout, \
             (dest / "stderr.log").open("w") as stderr:
            proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=stdout,
                                  stderr=stderr, timeout=1200)
        runner.require_idle(device, dest, "after")
        if proc.returncode:
            raise RuntimeError((dest / "stderr.log").read_text()[-2000:])
        event = runner.harness_event(dest / "stdout.log")
        if event["dtype"] != "fp32" or event["max_relative_error"] > .02:
            raise RuntimeError(event)
        timing = subprocess.run(
            ["python3", str(ROOT / "bf16_online_vs_master/parse_profile.py"),
             str(dest / "msprof"), "--last", "3", "--kernel-type", kind],
            capture_output=True, text=True, check=True)
        profile = json.loads(timing.stdout)
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
        metrics = {column: statistics.median(float(row[j]) for row in recent)
                   for j, column in enumerate(columns)
                   if column not in ("task_id", "stream_id", "subtask_id", "batch_id")
                   and all(row[j] is not None for row in recent)}
        result = {"index": index,
                  "set": "calibration" if index < CALIBRATION_COUNT else "holdout",
                  "shape": shape, "device": device, "tiling": tiling,
                  "event": event, "profile": profile, "pipe": metrics}
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        records.append(result)
        print("done", shape, "device", device, "us", profile["median_us"], flush=True)
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        groups = list(executor.map(process_device, (4, 5, 6, 7)))
    records = sorted((x for group in groups for x in group), key=lambda x: x["index"])
    (OUT / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print("saved", len(records), "records", flush=True)


if __name__ == "__main__":
    main()
