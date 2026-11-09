#!/usr/bin/env python3
"""New independent direct-copy FP32 BASE shapes for service-model validation."""

import sys
from pathlib import Path

ROOT = Path("/home/xianfei/solvebase_bench_20261025")
sys.path.insert(0, str(ROOT))
import fp32_base_narrow_mte2_measure_grid as grid

grid.OUT = ROOT / "cases/fp32_base_narrow_mte2_final_holdout_v1"
grid.CALIBRATION_COUNT = 0
grid.SHAPES = (
    (3584, 64, 20032),
    (5120, 64, 20480),
    (5120, 128, 20480),
    (2560, 128, 28672),
    (3584, 128, 20032),
    (4608, 64, 24512),
    (4608, 128, 24512),
    (3584, 64, 24512),
)

if __name__ == "__main__":
    grid.main()
