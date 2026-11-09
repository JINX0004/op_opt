#!/usr/bin/env python3
"""Add a shape-guarded, opt-in model-tile override to the research host tiler."""

import sys
from pathlib import Path

source = Path(sys.argv[1])
text = source.read_text()
anchor = '''    tmpRunInfo.singleCoreK = tmpRunInfo.stepKa * tmpRunInfo.baseK;
    tmpRunInfo.dbL0c = DB_SIZE;
    tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::SINGLE_CORE_SPLIT_K;'''
assert text.count(anchor) == 1, f'expected one mode-2 anchor, found {text.count(anchor)}'
override = '''    // Isolated FP32 experiment: execute the exact model outer tile. The
    // original source file and production packages are never changed.
    const char* exactTile = std::getenv("MMV3_RESEARCH_MODE2_EXACT_TILE");
    if (exactTile != nullptr && std::strcmp(exactTile, "1") == 0 &&
        args_.aType == ge::DT_FLOAT && args_.mValue == 2560 &&
        args_.kValue == 32768 &&
        (args_.nValue == 768 || args_.nValue == 1024)) {
        if (tmpRunInfo.baseM != 128 || tmpRunInfo.baseN != 128 ||
            tmpRunInfo.baseK != 64 || tmpRunInfo.stepM != 2 ||
            tmpRunInfo.stepN != 1 || tmpRunInfo.stepKa != 4 ||
            tmpRunInfo.stepKb != 4 || compileInfo_.aicNum != 20) {
            OP_LOGE(args_.opName, "exact mode-2 experiment block mismatch");
            return false;
        }
        tmpRunInfo.singleCoreM = 128;
        tmpRunInfo.singleCoreN = args_.nValue;
        tmpRunInfo.usedCoreNum = 20;
        OP_LOGI(args_.opName,
            "exact mode-2 model tile: core=(%lu,%lu) base=(%lu,%lu,%lu) steps=(%lu,%lu,%lu,%lu)",
            tmpRunInfo.singleCoreM, tmpRunInfo.singleCoreN,
            tmpRunInfo.baseM, tmpRunInfo.baseN, tmpRunInfo.baseK,
            tmpRunInfo.stepM, tmpRunInfo.stepN, tmpRunInfo.stepKa, tmpRunInfo.stepKb);
    }
'''
if len(sys.argv) == 3 and sys.argv[2] == 'neighbors':
    override = override.replace(
        'args_.kValue == 32768',
        '(args_.kValue == 32512 || args_.kValue == 32768 || args_.kValue == 33024)')
elif len(sys.argv) != 2:
    raise SystemExit('usage: enable_exact_mode2_tile.py SOURCE [neighbors]')
source.write_text(text.replace(anchor, override + anchor, 1))
