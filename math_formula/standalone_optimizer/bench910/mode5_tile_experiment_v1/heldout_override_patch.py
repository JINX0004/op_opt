"""Temporarily expose a 128x128 mode-5 tile for eligible FP16 NKM shapes."""

from pathlib import Path

path = Path('/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp')
backup = Path('/tmp/mmv3_mode5_override_original.cpp')
text = path.read_text()
anchor = '''    tmpRunInfo.singleCoreK = tmpRunInfo.stepKa * tmpRunInfo.baseK;
    tmpRunInfo.dbL0c = DB_SIZE;
    tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::SINGLE_CORE_SPLIT_K;'''
replacement = '''    // Device-only comparison on the FP16 K=1536, N=384 NKM channel.
    // The unmodified source tile is used when this variable is unset.
    const char* mode5Tile = std::getenv("MMV3_MODE5_TILE_EXPERIMENT");
    if (isNKMsmallK && args_.aType == ge::DT_FLOAT16 &&
        args_.nValue == 384 && args_.kValue == 1536 &&
        mode5Tile != nullptr && mode5Tile[0] == 's') {
        tmpRunInfo.singleCoreM = 128;
        tmpRunInfo.singleCoreN = 128;
        tmpRunInfo.usedCoreNum = 20;
        OP_LOGI(args_.opName, "Mode-5 held-out tile: singleCoreM=%lu singleCoreN=%lu",
                tmpRunInfo.singleCoreM, tmpRunInfo.singleCoreN);
    }
    tmpRunInfo.singleCoreK = tmpRunInfo.stepKa * tmpRunInfo.baseK;
    tmpRunInfo.dbL0c = DB_SIZE;
    tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::SINGLE_CORE_SPLIT_K;'''
if text.count(anchor) != 1:
    raise SystemExit(f'expected one anchor, found {text.count(anchor)}')
if backup.exists():
    if backup.read_text() != text:
        raise SystemExit(f'backup differs from current source: {backup}')
else:
    backup.write_text(text)
path.write_text(text.replace(anchor, replacement, 1))
print(f'patched {path}; backup={backup}')
