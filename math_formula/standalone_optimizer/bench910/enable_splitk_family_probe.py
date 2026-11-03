#!/usr/bin/env python3
"""Add an opt-in family selector to an isolated MatMulV3 host checkout.

Set MMV3_RESEARCH_SPLITK_FAMILY=single or deterministic for paired device
measurements. With the variable unset, source selection is unchanged.
The source branch still decides all tile fields and may reject a family.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source
    body = source.read_text()
    marker = '    const char* researchFamily = std::getenv("MMV3_RESEARCH_SPLITK_FAMILY");'
    if marker not in body:
        old = '#include <cinttypes>\n'
        assert body.count(old) == 1
        body = body.replace(old, old + '#include <cstdlib>\n#include <cstring>\n', 1)
        old = 'void MatmulV3BaseTiling::DoSelectTiling()\n{\n    switch (tilingSelect_) {'
        assert body.count(old) == 1
        new = '''void MatmulV3BaseTiling::DoSelectTiling()
{
    // Research-only paired comparison: invoke an existing source branch.
    // No tile field or kernel implementation is modified by this selector.
    const char* researchFamily = std::getenv("MMV3_RESEARCH_SPLITK_FAMILY");
    if (researchFamily != nullptr) {
        if (std::strcmp(researchFamily, "single") == 0) {
            tilingSelect_ = TilingCalcSelect::SINGLE_CORE_SPLIT_K;
        } else if (std::strcmp(researchFamily, "deterministic") == 0) {
            tilingSelect_ = TilingCalcSelect::DETERMINISTIC_SPLIT_K;
        }
    }
    switch (tilingSelect_) {'''
        body = body.replace(old, new, 1)
    old = 'if (compileInfo_.supportL12BtBf16 || !SupportMultiSplitK()) {'
    new = '''const char* forced = std::getenv("MMV3_RESEARCH_SPLITK_FAMILY");
    const bool forceDeterministic = forced != nullptr &&
        std::strcmp(forced, "deterministic") == 0;
    // SupportMultiSplitK is a shape/performance selection heuristic here;
    // retain the hardware gate and verify every research run on device.
    if (compileInfo_.supportL12BtBf16 ||
        (!forceDeterministic && !SupportMultiSplitK())) {'''
    if new not in body:
        assert body.count(old) == 1
        body = body.replace(old, new, 1)
    old = '''if (CheckSingleCoreSplitKEdgeCases(tmpRunInfo, isNKM, isMKNsmallK, isNKMsmallK)) {'''
    new = '''const char* forcedSingle = std::getenv("MMV3_RESEARCH_SPLITK_FAMILY");
    const bool researchSingle = forcedSingle != nullptr &&
        std::strcmp(forcedSingle, "single") == 0;
    // This check contains spatial occupancy heuristics, not capacity checks.
    // Validate the emitted research kernel against the reference output.
    if (!researchSingle &&
        CheckSingleCoreSplitKEdgeCases(tmpRunInfo, isNKM, isMKNsmallK, isNKMsmallK)) {'''
    if new not in body:
        assert body.count(old) == 1
        body = body.replace(old, new, 1)
    source.write_text(body)
    print(f"patched: {source}")


if __name__ == "__main__":
    main()
