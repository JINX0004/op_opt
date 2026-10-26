#!/usr/bin/env python3
"""Join verified MatMulV3 tiling and device task-time results."""

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    lines = path.read_text().splitlines()
    return json.loads(lines[-1])


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def axis_packets(shape: int, core: int, base: int) -> int:
    full, tail = divmod(shape, core)
    return full * ceil_div(core, base) + ceil_div(tail, base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, type=Path)
    parser.add_argument("--tiling", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--operator-variant", default="unspecified")
    parser.add_argument("--forced-used-cores", type=int)
    parser.add_argument("--expected-split-core", type=int, default=0)
    parser.add_argument("--expected-nd2nz-a", type=int, default=0)
    parser.add_argument("--expected-nd2nz-b", type=int, default=0)
    args = parser.parse_args()
    event, tiling, profile = (read_json(path) for path in
                              (args.event, args.tiling, args.profile))
    shape = (event["m"], event["n"], event["k"])
    realized = (tiling["mValue"], tiling["nValue"], tiling["kValue"])
    if shape != realized:
        raise SystemExit(f"shape mismatch: event {shape}, tiling {realized}")
    if (tiling.get("tilingEnableSplitCore") != args.expected_split_core or
            tiling.get("nd2nzA") != args.expected_nd2nz_a or
            tiling.get("nd2nzB") != args.expected_nd2nz_b):
        raise SystemExit("source split/conversion mode differs from expected mode")
    if not all("MatMulV3" in name for name in profile["kernel_names"]):
        raise SystemExit("profile did not execute MatMulV3")
    m, n, k = shape
    nm = axis_packets(m, tiling["singleCoreM"], tiling["baseM"])
    nn = axis_packets(n, tiling["singleCoreN"], tiling["baseN"])
    k_slices = ceil_div(k, tiling["singleCoreK"])
    # Deterministic Split-K distributes K slices across AICs even when
    # there is only one spatial output tile.
    nonidle_aics = (min(tiling["block_dim"], k_slices)
                    if args.expected_split_core == 3 else
                    tiling.get("estimated_nonidle_aics"))
    result = {
        "kind": ("solvebase_validation_case" if args.expected_split_core == 0
                 else "matmul_v3_split_validation_case"),
        "api": event["kind"],
        "operator_variant": args.operator_variant,
        "forced_used_cores": args.forced_used_cores,
        "shape": list(shape),
        "device": event["device"],
        "dtype": event["dtype"],
        "input_pattern": event.get("input_pattern", "unspecified"),
        "run_id": event["run_id"],
        "max_abs_error": event["max_abs_error"],
        "max_relative_error": event.get("max_relative_error"),
        "tiling_probe_event_us": event["median_us"],
        "kernel_median_us": profile["median_us"],
        "kernel_min_us": profile["min_us"],
        "kernel_max_us": profile["max_us"],
        "kernel_count": profile["kernel_count"],
        "kernel_names": profile["kernel_names"],
        "model_F_flops": 2 * m * n * k,
        "model_D_read_bytes": 2 * k * (m * nn + n * nm),
        "model_output_bytes": 2 * m * n,
        "model_NM_base_packets": nm,
        "model_NN_base_packets": nn,
        "model_S_spatial_packets": nm * nn,
        "model_P_k_packets": ceil_div(k, tiling["baseK"]),
        "split_core_mode": tiling["tilingEnableSplitCore"],
        "nd2nz_a": tiling.get("nd2nzA"),
        "nd2nz_b": tiling.get("nd2nzB"),
        "k_slices": k_slices,
        "tiling_key": tiling.get("tiling_key"),
        "tiling_source": tiling.get("tiling_source"),
        "block_dim": tiling.get("block_dim"),
        "output_tiles": tiling["output_tiles"],
        "estimated_nonidle_aics": nonidle_aics,
        "singleCoreM": tiling["singleCoreM"],
        "singleCoreN": tiling["singleCoreN"],
        "singleCoreK": tiling["singleCoreK"],
        "baseM": tiling["baseM"],
        "baseN": tiling["baseN"],
        "baseK": tiling["baseK"],
        "stepM": tiling["stepM"],
        "stepN": tiling["stepN"],
        "stepKa": tiling["stepKa"],
        "stepKb": tiling["stepKb"],
        "depthA1": tiling["depthA1"],
        "depthB1": tiling["depthB1"],
        "workspace_bytes": tiling.get("workspace_bytes"),
        "tiling_log": tiling["plog"],
        "profile_csv": profile["profile_csv"],
    }
    # For Split-K, packet counts above describe tile geometry; they are not
    # a full byte-traffic model for the partial-result and vector-reduce path.
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
