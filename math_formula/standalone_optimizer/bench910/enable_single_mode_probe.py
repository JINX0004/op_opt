#!/usr/bin/env python3
"""Add an opt-in mode-2 vs mode-6 source-tiler probe to an isolated build.

The existing single-core source branch computes all tile fields. Setting
MMV3_RESEARCH_SC_MODE=2 suppresses its final GM-to-L1 mode-6 conversion;
the normal mode-2 path and kernel are then used. Unset or 6 is unchanged.
The input source must already contain the isolated family-probe patch.
"""

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    path = args.source
    body = path.read_text()
    if 'MMV3_RESEARCH_SPLITK_FAMILY' not in body:
        raise SystemExit('requires the isolated split-K family probe')
    new = '''        // Research-only mode comparison: keep the source tile and select
        // the ordinary single-core kernel instead of its GM-to-L1 variant.
        const char* researchMode = std::getenv("MMV3_RESEARCH_SC_MODE");
        if (researchMode == nullptr || std::strcmp(researchMode, "2") != 0) {
            IsGmToL1ByShape();
        }'''
    if new in body:
        print(f'already patched: {path}')
        return
    pattern = r'^        IsGmToL1ByShape\(\);\s*// 走61：[^\n]*$'
    if len(re.findall(pattern, body, flags=re.MULTILINE)) != 1:
        raise SystemExit(f'expected exactly one mode-6 conversion in {path}')
    path.write_text(re.sub(pattern, new, body, count=1, flags=re.MULTILINE))
    print(f'patched: {path}')


if __name__ == '__main__':
    main()
