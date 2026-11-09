#!/usr/bin/env python3
"""Development and held-out narrow-N row-width grid on 910B3."""

import sys
from pathlib import Path


ROOT = Path("/home/xianfei/solvebase_bench_20261025")
sys.path.insert(0, str(ROOT))
import fp32_base_narrow_mte2_measure_grid as grid

grid.OUT = ROOT / "cases/fp32_base_narrow_mte2_unaligned_v1"
grid.CALIBRATION_COUNT = 4
grid.SHAPES = (
    (2048, 96, 32640),
    (3072, 80, 32640),
    (3072, 112, 32640),
    (3072, 96, 16256),
    (4096, 80, 32640),
    (4096, 112, 32640),
    (4096, 96, 16256),
    (3072, 96, 24448),
)

if __name__ == "__main__":
    grid.main()
