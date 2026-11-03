#!/usr/bin/env python3
"""Add AIC phase-cycle markers to an isolated MatMulV3 source copy.

Run only on a fresh copy of the selector checkout. The diagnostic build
prints one line from AIC block zero after measuring the waits and Cube work.
"""

import sys
from pathlib import Path


def replace_once(path: Path, before: str, after: str) -> None:
    source = path.read_text()
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"expected one anchor in {path}, found {count}")
    path.write_text(source.replace(before, after, 1))


root = Path(sys.argv[1])
kernel = root / "matmul/mat_mul_v3/op_kernel"
base = kernel / "mat_mul_unaligned_base_kernel.h"
split = kernel / "mat_mul_unaligned_deterministic_splitk_kernel.h"

replace_once(
    base,
    """{
    ProcessNDtoNZ();
    if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
        mmb_.Process(index, enAtomic);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
        mma_.Process(index, enAtomic);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
        mmab_.Process(index, enAtomic);
    }
}""",
    """{
    // Diagnostic copy only: AIC block zero measures the complete AIV
    // conversion barrier and its own Cube work. Print after both intervals.
    uint64_t phaseBegin = 0;
    uint64_t phaseAfterWait = 0;
    if ASCEND_IS_AIC {
        if (GetBlockIdx() == 0) phaseBegin = static_cast<uint64_t>(GetSystemCycle());
    }
    ProcessNDtoNZ();
    if ASCEND_IS_AIC {
        if (GetBlockIdx() == 0) phaseAfterWait = static_cast<uint64_t>(GetSystemCycle());
    }
    if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
        mmb_.Process(index, enAtomic);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
        mma_.Process(index, enAtomic);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
        mmab_.Process(index, enAtomic);
    }
    if ASCEND_IS_AIC {
        if (GetBlockIdx() == 0) {
            const uint64_t phaseAfterCube = static_cast<uint64_t>(GetSystemCycle());
            AscendC::printf("MMV3_PHASE_BASE k=%lu wait_cycles=%lu cube_cycles=%lu\\n",
                            static_cast<uint64_t>(matmulTilingData_->matmulTiling.Ka),
                            phaseAfterWait - phaseBegin, phaseAfterCube - phaseAfterWait);
        }
    }
}""",
)

replace_once(
    split,
    """    GM_ADDR alignedworkspaceGM = reinterpret_cast<GM_ADDR>(mmGM +
                                 tiling.usedCoreNum * singleSize * NUM_TWO * sizeof(float)); // NUM_TWO for DB
    if ASCEND_IS_AIV {""",
    """    GM_ADDR alignedworkspaceGM = reinterpret_cast<GM_ADDR>(mmGM +
                                 tiling.usedCoreNum * singleSize * NUM_TWO * sizeof(float)); // NUM_TWO for DB
    // Diagnostic copy only: measure from AIC entry to the AIV handoff.
    uint64_t phaseBegin = 0;
    uint64_t phaseAfterWait = 0;
    if ASCEND_IS_AIC {
        if (GetBlockIdx() == 0) phaseBegin = static_cast<uint64_t>(GetSystemCycle());
    }
    if ASCEND_IS_AIV {""",
)
replace_once(
    split,
    """    if ASCEND_IS_AIC {
        WaitEvent(ND2NZ_AIV_SYNC_AIC_FLAG);
        if (GetBlockIdx() >= tiling.usedCoreNum) {""",
    """    if ASCEND_IS_AIC {
        WaitEvent(ND2NZ_AIV_SYNC_AIC_FLAG);
        if (GetBlockIdx() == 0) phaseAfterWait = static_cast<uint64_t>(GetSystemCycle());
        if (GetBlockIdx() >= tiling.usedCoreNum) {""",
)
replace_once(
    split,
    """        }
        return;
    }
}
#endif // __OP_KERNEL_MATMUL_V3_H__""",
    """        }
        if (GetBlockIdx() == 0) {
            const uint64_t phaseAfterCube = static_cast<uint64_t>(GetSystemCycle());
            AscendC::printf("MMV3_PHASE_SPLITK k=%lu wait_cycles=%lu cube_cycles=%lu\\n",
                            static_cast<uint64_t>(tiling.Ka),
                            phaseAfterWait - phaseBegin, phaseAfterCube - phaseAfterWait);
        }
        return;
    }
}
#endif // __OP_KERNEL_MATMUL_V3_H__""",
)

print(f"patched {base} and {split}")
