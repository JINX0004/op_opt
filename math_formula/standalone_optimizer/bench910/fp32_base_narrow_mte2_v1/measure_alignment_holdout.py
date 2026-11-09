#!/usr/bin/env python3
"""Independent 16-KiB-pitch FP32 BASE holdout after adding pitch class."""

import sys
from pathlib import Path

ROOT = Path("/home/xianfei/solvebase_bench_20261025")
sys.path.insert(0, str(ROOT))
import fp32_base_narrow_mte2_measure_grid as grid

grid.OUT = ROOT / "cases/fp32_base_narrow_mte2_alignment_holdout_v1"
grid.CALIBRATION_COUNT = 0
grid.SHAPES = (
    (6144, 64, 12288),
    (8192, 128, 12288),
    (3072, 64, 36864),
    (3072, 128, 36864),
    (4096, 64, 36864),
    (2048, 128, 45056),
    (4096, 64, 20160),
    (4096, 128, 20160),
)

if __name__ == "__main__":
    grid.main()
