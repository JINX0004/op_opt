#!/usr/bin/env python3
"""Isolated research patch: separate A physical pitch from logical K."""

import sys
from pathlib import Path

root = Path(sys.argv[1])
host = root / 'matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp'
block = root / 'matmul/mat_mul_v3/op_kernel/mat_mul_sc_splitk_block.h'

text = host.read_text()
include = '#include <cinttypes>\n'
assert text.count(include) == 1
text = text.replace(include, include + '#include <cstdlib>\n', 1)
anchor = '''    SetNd2NzInfo();
    DoTilingKey();'''
assert text.count(anchor) == 1
override = '''    // Research-only: A is physically padded while logical K and B stay fixed.
    // The paired device block uses Kb for loop count and Ka for A row pitch.
    const char* pitchText = std::getenv("MMV3_RESEARCH_A_PITCH");
    if (pitchText != nullptr && args_.aType == ge::DT_FLOAT &&
        args_.mValue >= 2048 && args_.kValue == 48384) {
        char* end = nullptr;
        const unsigned long pitch = std::strtoul(pitchText, &end, 10);
        if (end == pitchText || *end != '\\0' ||
            (pitch != 48896 && pitch != 49152 && pitch != 49408) ||
            tilingEnable_.tilingEnableSplitCore != TilingEnableSplitCore::SINGLE_CORE_SPLIT_K ||
            tilingData_.matmulTiling.Ka != 48384 ||
            tilingData_.matmulTiling.Kb != 48384) {
            OP_LOGE(args_.opName, "invalid fixed-work A-pitch experiment");
            return ge::GRAPH_FAILED;
        }
        tilingData_.matmulTiling.Ka = static_cast<uint32_t>(pitch);
        OP_LOGI(args_.opName, "fixed-work A pitch: logicalK=%lu Ka=%lu Kb=%lu",
                args_.kValue, static_cast<uint64_t>(tilingData_.matmulTiling.Ka),
                static_cast<uint64_t>(tilingData_.matmulTiling.Kb));
    }
    SetNd2NzInfo();
    DoTilingKey();'''
host.write_text(text.replace(anchor, override, 1))

text = block.read_text()
changes = (
    ('params_.kCnt = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka,',
     'params_.kCnt = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb,'),
    ('params_.kCoreTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.Ka) -',
     'params_.kCoreTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.Kb) -'),
    ('params_.loopK = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka,',
     'params_.loopK = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb,'),
)
for before, after in changes:
    assert text.count(before) == 1, before
    text = text.replace(before, after, 1)
block.write_text(text)
