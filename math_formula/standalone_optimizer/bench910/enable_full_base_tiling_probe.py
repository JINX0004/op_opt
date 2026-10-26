#!/usr/bin/env python3
"""Add a research-only source-compatible BASE override to an isolated clone.

MMV3_RESEARCH_BASE_TILE=singleCoreM,singleCoreN,baseM,baseN,baseK,cores
is applied after MatMulV3 chooses the BASE family. Plain BASE key 0x10000
is restricted to singleCoreM=baseM and singleCoreN=baseN, as verified by
full-output device checks. CalL1Tiling() and DoL2CacheTiling() derive buffer
depths and L2 partitions for the new tile. The benchmark
must still verify the emitted tuple and every output element before timing.
"""

import argparse
from pathlib import Path


MARKER = "MMV3_RESEARCH_FULL_BASE_TILE"
L2_MARKER = "MMV3_RESEARCH_SOURCE_L2"
ANCHOR = """    if (runInfo_.needUpdate) {
        if (CheckMMTilingDataIsVaild()) {
"""
L2_ANCHOR = """        runInfo_.usedCoreNum = cores;
        runInfo_.stepM = 1;
"""
L2_INSERT = """        // MMV3_RESEARCH_SOURCE_L2: recalculate source L2 partitions
        // after changing the tile that one BASE Iterate call writes.
        DoL2CacheTiling();
"""
INSERT = """    // MMV3_RESEARCH_FULL_BASE_TILE: isolated, opt-in evaluation only.
    if (const char* requested = std::getenv("MMV3_RESEARCH_BASE_TILE")) {
        if (tilingEnable_.tilingEnableSplitCore != TilingEnableSplitCore::BASE ||
            !runInfo_.needUpdate) {
            OP_LOGE(args_.opName, "research BASE tile requested outside BASE path");
            return ge::FAILED;
        }
        unsigned long long cm = 0, cn = 0, bm = 0, bn = 0, bk = 0, cores = 0;
        char trailing = 0;
        if (std::sscanf(requested, "%llu,%llu,%llu,%llu,%llu,%llu%c",
                        &cm, &cn, &bm, &bn, &bk, &cores, &trailing) != 6 ||
            !cm || !cn || !bm || !bn || !bk || !cores ||
            cm % 16 || cn % 16 || bm % 16 || bn % 16 || bk % 16 ||
            bm != cm || bn != cn || cm > args_.mValue || cn > args_.nValue ||
            bk > args_.kValue || cores > compileInfo_.aicNum ||
            cores > static_cast<uint64_t>(MathUtil::CeilDivision(args_.mValue, cm) *
                                           MathUtil::CeilDivision(args_.nValue, cn)) ||
            2ULL * bm * bk * aDtypeSize_ > compileInfo_.l0ASize ||
            2ULL * bn * bk * bDtypeSize_ > compileInfo_.l0BSize ||
            bm * bn * sizeof(float) > compileInfo_.l0CSize) {
            OP_LOGE(args_.opName, "invalid MMV3_RESEARCH_BASE_TILE=%s", requested);
            return ge::FAILED;
        }
        runInfo_.baseM = bm;
        runInfo_.baseN = bn;
        runInfo_.baseK = bk;
        CalL1Tiling(); // source routine recomputes legal depthA1/B1 and stepKa/Kb
        runInfo_.singleCoreM = cm;
        runInfo_.singleCoreN = cn;
        runInfo_.singleCoreK = args_.kValue;
        runInfo_.usedCoreNum = cores;
        runInfo_.stepM = 1;
        runInfo_.stepN = 1;
        if (!runInfo_.depthA1 || !runInfo_.depthB1 ||
            !runInfo_.stepKa || !runInfo_.stepKb ||
            (runInfo_.depthA1 * bm + runInfo_.depthB1 * bn) * bk *
                aDtypeSize_ > compileInfo_.l1Size + 256) {
            OP_LOGE(args_.opName, "research BASE tile exceeds L1");
            return ge::FAILED;
        }
        OP_LOGI(args_.opName,
            "MMV3_RESEARCH_BASE_TILE=%s, source finalized depthAB=(%lu,%lu) stepK=(%lu,%lu)",
            requested, runInfo_.depthA1, runInfo_.depthB1,
            runInfo_.stepKa, runInfo_.stepKb);
    }
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    path = args.source.resolve()
    if "ops_nn_source_build" not in path.parts:
        raise SystemExit("refusing to patch outside ops_nn_source_build")
    source = path.read_text()
    if MARKER in source:
        if L2_MARKER in source:
            print("already patched", path)
            return
        if "MMV3_RESEARCH_NEUTRAL_L2" in source or \
                "MMV3_RESEARCH_REMAP_L2" in source:
            old_marker = ("MMV3_RESEARCH_NEUTRAL_L2" if
                          "MMV3_RESEARCH_NEUTRAL_L2" in source else
                          "MMV3_RESEARCH_REMAP_L2")
            start = source.index(f"        // {old_marker}:")
            end = source.index("        runInfo_.stepN = 1;", start)
            source = source[:start] + L2_INSERT + source[end:]
            source = source.replace("bm > cm || bn > cn", "bm != cm || bn != cn")
            source = source.replace(
                "2ULL * bm * bn * sizeof(float) > compileInfo_.l0CSize",
                "bm * bn * sizeof(float) > compileInfo_.l0CSize")
            path.write_text(source)
            print("upgraded to source-compatible BASE", path)
            return
        if source.count(L2_ANCHOR) != 1:
            raise SystemExit("existing research override differs")
        path.write_text(source.replace(L2_ANCHOR, L2_ANCHOR + L2_INSERT))
        print("upgraded L2 mapping", path)
        return
    if source.count(ANCHOR) != 1:
        raise SystemExit("source anchor mismatch")
    source = source.replace("#include <cinttypes>",
                            "#include <cinttypes>\n#include <cstdio>", 1)
    patch = INSERT.replace(L2_ANCHOR, L2_ANCHOR + L2_INSERT)
    path.write_text(source.replace(ANCHOR, patch + ANCHOR))
    print("patched", path)


if __name__ == "__main__":
    main()
