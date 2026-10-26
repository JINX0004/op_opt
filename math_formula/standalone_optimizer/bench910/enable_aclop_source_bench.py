#!/usr/bin/env python3
"""Add a benchmark-only escape from an unavailable CANN 9 legacy cache hook.

Apply only to an isolated copy of ops-nn-sribd, never the user's source tree.
The original code remains the default; MMV3_SKIP_LEGACY_OP_CACHE enables
the bypass for the aclop single-operator benchmark process.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    source = (args.source_root / "matmul/mat_mul_v3/op_host/op_tiling/"
              "matmul_v3_tiling.cpp")
    original = source.read_text()
    before_include = "#include <type_traits>"
    before_check = "if(!TilingPrepareForOpCache(context)) {"
    if original.count(before_include) != 1 or original.count(before_check) != 1:
        raise SystemExit("unexpected source; no edit made")
    updated = original.replace(before_include,
                               "#include <cstdlib>\n#include <type_traits>")
    updated = updated.replace(
        before_check,
        '// aclop single-op in CANN 9.0.0 lacks the optional legacy cache hook.\n'
        '  // Normal operator use still runs the original preparation.\n'
        '  if (std::getenv("MMV3_SKIP_LEGACY_OP_CACHE") == nullptr &&\n'
        '      !TilingPrepareForOpCache(context)) {')
    source.write_text(updated)
    print(source)


if __name__ == "__main__":
    main()
