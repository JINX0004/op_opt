#!/usr/bin/env python3
"""Independent narrow-N holdout at smaller/larger K and N=96."""

import sys
from pathlib import Path


ROOT = Path("/home/xianfei/solvebase_bench_20261025")
sys.path.insert(0, str(ROOT))
import fp32_base_narrow_mte2_measure_grid as grid
grid.OUT = ROOT / "cases/fp32_base_narrow_mte2_broad_holdout_v1"
grid.CALIBRATION_COUNT = 0
grid.SHAPES = (
    (3072, 64, 16256),
    (4096, 64, 16256),
    (6144, 64, 16256),
    (4096, 128, 16256),
    (3072, 96, 32640),
    (4096, 96, 32640),
    (4096, 64, 32640),
    (8192, 64, 8160),
)

if __name__ == "__main__":
    grid.main()
