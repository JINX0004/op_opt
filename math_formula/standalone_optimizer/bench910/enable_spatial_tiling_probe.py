#!/usr/bin/env python3
"""Add an isolated, downward-only BASE M-tile experiment to an ops-nn clone.

This must never be applied to the user's working source tree. The override
changes the generated tiling tuple after the normal heuristic. Its exact
kernel path is validated by the output check in run_case.sh before timing.
"""
import argparse
from pathlib import Path


MARKER = "MMV3_RESEARCH_FORCE_BASE_M"
ANCHOR = """ge::graphStatus MatmulV3BaseTiling::DoLibApiTiling()
{
    SetRunInfo();
"""
REPLACEMENT = """ge::graphStatus MatmulV3BaseTiling::DoLibApiTiling()
{
    SetRunInfo();
    // MMV3_RESEARCH_FORCE_BASE_M: experiment only in an isolated clone.
    // The normal tiler has already chosen the family and a valid tuple.
    // Shrink both baseM and singleCoreM; original depth and double-buffer
    // allocations then remain conservative. run_case.sh checks the result.
    if (tilingEnable_.tilingEnableSplitCore == TilingEnableSplitCore::BASE) {
        const char *requested = std::getenv("MMV3_FORCE_BASE_M");
        if (requested != nullptr && requested[0] != '\\0') {
            char *end = nullptr;
            const uint64_t forced = std::strtoul(requested, &end, 10);
            if (end == requested || *end != '\\0' || forced < 16UL ||
                forced % 16UL != 0UL || forced > runInfo_.baseM ||
                forced > runInfo_.singleCoreM || !runInfo_.needUpdate) {
                OP_LOGE(args_.opName, "invalid MMV3_FORCE_BASE_M=%s", requested);
                return ge::FAILED;
            }
            runInfo_.baseM = forced;
            runInfo_.singleCoreM = forced;
            OP_LOGI(args_.opName,
                "MMV3_FORCE_BASE_M=%lu, BASE baseM=%lu singleCoreM=%lu",
                forced, runInfo_.baseM, runInfo_.singleCoreM);
        }
    }
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path,
                        help="matmul_v3_base_tiling.cpp inside isolated clone")
    args = parser.parse_args()
    path = args.source.resolve()
    if "ops_nn_source_build" not in path.parts:
        raise SystemExit("refusing to patch outside ops_nn_source_build")
    source = path.read_text()
    if MARKER in source:
        old = "            runInfo_.baseM = forced;\n"
        new = old + "            runInfo_.singleCoreM = forced;\n"
        if new in source:
            print("already patched", path)
            return
        if source.count(old) != 1:
            raise SystemExit("existing research override differs")
        path.write_text(source.replace(old, new))
        print("updated", path)
        return
    if source.count(ANCHOR) != 1:
        raise SystemExit("source anchor mismatch")
    path.write_text(source.replace(ANCHOR, REPLACEMENT))
    print("patched", path)


if __name__ == "__main__":
    main()
