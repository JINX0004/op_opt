#!/usr/bin/env python3
"""Extract the realized MatMulV3 tiling from CANN 9.0 INFO plog lines."""

import argparse
import glob
import json
import re
from pathlib import Path


SPATIAL_FIELDS = (
    "mValue", "nValue", "kValue", "singleCoreM", "singleCoreN",
    "singleCoreK", "baseM", "baseN", "baseK", "stepM", "stepN",
    "stepKa", "stepKb", "depthA1", "depthB1",
)
MODE_FIELDS = ("tilingEnableSplitCore", "tilingEnableFullLoad",
               "tilingEnableFixOpti")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, help="benchmark process ID")
    parser.add_argument("--plog", type=Path, help="explicit CANN plog file")
    args = parser.parse_args()
    if args.plog:
        path = args.plog
    elif args.pid:
        files = sorted(glob.glob(str(
            Path.home() / "ascend/log/debug/plog" / f"plog-{args.pid}_*.log")))
        if not files:
            raise SystemExit("no INFO plog for PID; set ASCEND_GLOBAL_LOG_LEVEL=1")
        path = Path(files[-1])
    else:
        parser.error("provide --pid or --plog")
    result: dict[str, object] = {"kind": "matmul_v3_tiling", "plog": str(path)}
    with path.open(errors="replace") as source:
        for line in source:
            if "SetL2CacheFlag" in line and "singleCoreM:" in line:
                source_match = re.search(
                    r"\[(/[^\]\s]*matmul_v3_l2_cache\.cpp):\d+\]", line)
                if source_match:
                    result["tiling_source"] = source_match.group(1)
                for field in SPATIAL_FIELDS:
                    match = re.search(rf"\b{field}:\s*(\d+)", line)
                    if match:
                        result[field] = int(match.group(1))
                for field in MODE_FIELDS:
                    match = re.search(rf"\b{field}\s+(\d+)", line)
                    if match:
                        result[field] = int(match.group(1))
            elif "Tiling Key is" in line:
                match = re.search(r"Tiling Key is\s+(0x[0-9a-fA-F]+)", line)
                if match:
                    result["tiling_key"] = int(match.group(1), 16)
            elif "Tiling key is" in line and "block_dim is" in line:
                match = re.search(r"block_dim is\s+(\d+)", line)
                if match:
                    result["block_dim"] = int(match.group(1))
            elif "args_.nd2nzA =" in line:
                match = re.search(r"args_\.nd2nzA = (\d+), args_\.nd2nzB = (\d+)", line)
                if match:
                    result["nd2nzA"], result["nd2nzB"] = map(int, match.groups())
            elif "final workspace size is" in line:
                match = re.search(r"final workspace size is\s+(\d+)", line)
                if match:
                    result["workspace_bytes"] = int(match.group(1))
    if not all(field in result for field in SPATIAL_FIELDS + MODE_FIELDS):
        raise SystemExit("complete MatMulV3 tiling line not found in " + str(path))
    result["family"] = ("BASE" if result["tilingEnableSplitCore"] == 0
                        else "OTHER")
    if "block_dim" in result:
        m_tiles = (result["mValue"] + result["singleCoreM"] - 1) // result["singleCoreM"]
        n_tiles = (result["nValue"] + result["singleCoreN"] - 1) // result["singleCoreN"]
        result["output_tiles"] = m_tiles * n_tiles
        result["estimated_nonidle_aics"] = min(result["block_dim"], m_tiles * n_tiles)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
