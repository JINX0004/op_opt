#!/usr/bin/env python3
"""Opt-in, shape-specific FP32 BASE 256x64x32 experiment."""

import sys
from pathlib import Path


def main() -> None:
    source = Path(sys.argv[1]) / "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp"
    original = source.read_text()
    before = "    DoBasicTiling();\n    OptimizeBasicKernelStepK();\n    SetNd2NzInfo();"
    after = """    DoBasicTiling();
    OptimizeBasicKernelStepK();
    // Research-only: execute the selector's BASE tuple for this FP32 shape.
    const char* fp32TileProbe = std::getenv("MMV3_FP32_4096X64_TILE_PROBE");
    if (fp32TileProbe != nullptr && fp32TileProbe[0] == '1' &&
        args_.aType == ge::DT_FLOAT && args_.bType == ge::DT_FLOAT &&
        args_.cType == ge::DT_FLOAT && !args_.isHf32 &&
        args_.mValue == 4096 && args_.nValue == 64 && args_.kValue == 27456 &&
        tilingSelect_ == TilingCalcSelect::BASE) {
        runInfo_.usedCoreNum = 16;
        runInfo_.baseM = 256;
        runInfo_.baseN = 64;
        runInfo_.baseK = 32;
        runInfo_.stepM = 1;
        runInfo_.stepN = 1;
        CalL1Tiling();
        DoL2CacheTiling();
        OP_LOGI(args_.opName,
            "FP32 BASE tile probe applied: cores=%lu baseM=%lu baseN=%lu baseK=%lu stepKa=%lu stepKb=%lu",
            runInfo_.usedCoreNum, runInfo_.baseM, runInfo_.baseN, runInfo_.baseK,
            runInfo_.stepKa, runInfo_.stepKb);
    }
    SetNd2NzInfo();"""
    if original.count(before) != 1:
        raise RuntimeError(f"expected one patch point in {source}")
    source.write_text(original.replace(before, after))


if __name__ == "__main__":
    main()
