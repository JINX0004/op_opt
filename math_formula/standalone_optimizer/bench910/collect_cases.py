#!/usr/bin/env python3
"""Make a portable, compact dataset from successful run_case.sh directories."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    cases = []
    for directory in args.case_dirs:
        case = json.loads((directory / "case.json").read_text())
        timing = json.loads((directory / "kernel_time.json").read_text())
        tiling = json.loads((directory / "tiling.json").read_text())
        case.pop("profile_csv", None)
        case.pop("tiling_log", None)
        case["case_id"] = directory.name
        case["kernel_durations_us"] = timing["durations_us"]
        case["tilingEnableFullLoad"] = tiling["tilingEnableFullLoad"]
        case["tilingEnableFixOpti"] = tiling["tilingEnableFixOpti"]
        case["nd2nzA"] = tiling["nd2nzA"]
        case["nd2nzB"] = tiling["nd2nzB"]
        cases.append(case)
    print(json.dumps({"schema": "solvebase_pilot_cases_v1", "cases": cases},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
