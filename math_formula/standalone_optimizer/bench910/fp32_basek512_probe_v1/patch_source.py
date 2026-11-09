#!/usr/bin/env python3
"""Opt-in, shape-specific FP32 BASE baseK=512 experiment.

Apply only to an isolated copy of the source-matched research tiler.
The benchmark must confirm the emitted tile and output correctness.
"""

import sys
from pathlib import Path


def main() -> None:
    source = Path(sys.argv[1]) / "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp"
    original = source.read_text()
    before = "    DoBasicTiling();\n    OptimizeBasicKernelStepK();\n    SetNd2NzInfo();"
    after = """    DoBasicTiling();
    OptimizeBasicKernelStepK();
    // Research-only: test the model's proposed K tile for one FP32 shape.
    const char* fp32BaseK512Probe = std::getenv("MMV3_FP32_BASEK512_PROBE");
    if (fp32BaseK512Probe != nullptr && fp32BaseK512Probe[0] == '1' &&
        args_.aType == ge::DT_FLOAT && args_.bType == ge::DT_FLOAT &&
        args_.cType == ge::DT_FLOAT && !args_.isHf32 &&
        args_.mValue == 16 && args_.nValue == 16 && args_.kValue == 65408 &&
        tilingSelect_ == TilingCalcSelect::BASE) {
        runInfo_.baseK = 512;
        CalL1Tiling();
        OP_LOGI(args_.opName,
            "FP32 BASE K probe applied: baseM=%lu baseN=%lu baseK=%lu stepKa=%lu stepKb=%lu",
            runInfo_.baseM, runInfo_.baseN, runInfo_.baseK,
            runInfo_.stepKa, runInfo_.stepKb);
    }
    SetNd2NzInfo();"""
    if original.count(before) != 1:
        raise RuntimeError(f"expected one patch point in {source}")
    source.write_text(original.replace(before, after))


if __name__ == "__main__":
    main()
