from pathlib import Path
import sys
path = Path('/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp')
backup = Path('/tmp/mm_bf16_mode5_original.cpp')
anchor = '''    tmpRunInfo.singleCoreK = tmpRunInfo.stepKa * tmpRunInfo.baseK;
    tmpRunInfo.dbL0c = DB_SIZE;
    tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::SINGLE_CORE_SPLIT_K;'''
addition = '''    // Benchmark-only BF16 mode-5 geometry probe.
    const char* bf16Tile = std::getenv("MMV3_BF16_MODE5_TILE_PROBE");
    if (isNKMsmallK && args_.aType == ge::DT_BF16 &&
        args_.nValue == 384 && args_.kValue == 1536 &&
        bf16Tile != nullptr && bf16Tile[0] == 's') {
        tmpRunInfo.singleCoreM = 128;
        tmpRunInfo.singleCoreN = 128;
        tmpRunInfo.usedCoreNum = 20;
    }
'''
if sys.argv[1] == 'apply':
    original = path.read_text()
    if backup.exists():
        raise SystemExit('backup already exists')
    if original.count(anchor) != 1:
        raise SystemExit(f'anchor count={original.count(anchor)}')
    backup.write_text(original)
    path.write_text(original.replace(anchor, addition + anchor, 1))
    print('patched')
elif sys.argv[1] == 'restore':
    if not backup.exists():
        raise SystemExit('backup missing')
    path.write_text(backup.read_text())
    backup.unlink()
    print('restored')
else:
    raise SystemExit('use apply or restore')
