#!/usr/bin/env python3
"""Keep the established FP16 family, tile and five-decimal objective fixed."""

import os
import subprocess
import sys

binary, fixture = sys.argv[1:3]
rows = [line.strip() for line in open(fixture, encoding="utf-8") if line.strip()]
env = dict(os.environ, MMV3_ONLINE_ROW_READ_PROFILE="1")
inputs = "".join(" ".join(row.split(",")[:3]) + " fp16\n" for row in rows)
run = subprocess.run([binary], input=inputs, text=True, capture_output=True, env=env)
if run.returncode:
    raise SystemExit(run.stderr or f"selector exited {run.returncode}")
actual = run.stdout.splitlines()[1:]
if actual != rows:
    for index, (expected, found) in enumerate(zip(rows, actual)):
        if expected != found:
            raise SystemExit(f"FP16 regression at row {index}:\n{expected}\n{found}")
    raise SystemExit(f"FP16 row count changed: {len(rows)} -> {len(actual)}")
