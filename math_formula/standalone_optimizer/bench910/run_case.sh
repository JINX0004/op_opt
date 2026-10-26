#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
    printf 'Usage: CANN_ROOT=/path/to/cann bash run_case.sh M N K OUT_DIR [DEVICE] [WARMUP] [REPETITIONS]\n' >&2
    exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CANN_ROOT:?Set CANN_ROOT to the installed CANN root}"
set +u
source "$CANN_ROOT/set_env.sh"
set -u
api="${BENCH_API:-v3cached}"
preload="${LD_PRELOAD:-}"
operator_variant="cann_builtin"
if [[ -n "${BENCH_CUSTOM_OPP_PATH:-}" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="$BENCH_CUSTOM_OPP_PATH"
    operator_variant="$(basename "$BENCH_CUSTOM_OPP_PATH")"
    opapi="$BENCH_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so"
    if [[ ! -f "$opapi" ]]; then
        printf 'Custom operator API not found: %s\n' "$opapi" >&2
        exit 2
    fi
    export LD_LIBRARY_PATH="$(dirname "$opapi"):${LD_LIBRARY_PATH:-}"
    # Direct aclop dispatch needs the custom OPP metadata, but not the
    # aclnn API library. Preloading that library into Python profiler
    # subprocesses can perturb their signal setup.
    if [[ "$api" == "mm" || "$api" == "aclnn" ]]; then
        preload="$opapi${preload:+ $preload}"
    fi
fi

m="$1"; n="$2"; k="$3"; output_dir="$4"
device="${5:-0}"; warmup="${6:-20}"; repetitions="${7:-5}"
mkdir -p "$output_dir"

# INFO logging is confined to this single correctness/tiling run. It is
# intentionally not used for the timing profile, where its cost would bias
# the measurement. The CANN 9.0.0 single-op compile path needs the explicit
# dtype-inference bypass, and the runner checks every output element.
ASCEND_GLOBAL_LOG_LEVEL=1 IGNORE_INFER_ERROR=1 LD_PRELOAD="$preload" \
    "$script_dir/base_aclnn_bench" --api "$api" --device "$device" \
    --m "$m" --n "$n" --k "$k" --warmup 0 --batches 1 --repetitions 1 \
    > "$output_dir/tiling_event.jsonl"
pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' \
    "$output_dir/tiling_event.jsonl")"
python3 "$script_dir/parse_tiling_log.py" --pid "$pid" > "$output_dir/tiling.json"
python3 - "$output_dir/tiling.json" "${BENCH_EXPECT_SOURCE:-}" \
    "${BENCH_EXPECT_SPLIT_CORE_MODE:-0}" "${BENCH_EXPECT_ND2NZ_A:-0}" \
    "${BENCH_EXPECT_ND2NZ_B:-0}" <<'PY'
import json
import sys

tiling = json.load(open(sys.argv[1]))
expected_mode, expected_a, expected_b = map(int, sys.argv[3:6])
if (tiling.get("tilingEnableSplitCore") != expected_mode or
        tiling.get("nd2nzA") != expected_a or
        tiling.get("nd2nzB") != expected_b):
    raise SystemExit("unexpected split/conversion mode; "
                     f"expected={(expected_mode, expected_a, expected_b)} "
                     f"saved tiling in {sys.argv[1]}")
if sys.argv[2] and not tiling.get("tiling_source", "").startswith(sys.argv[2]):
    raise SystemExit("tiling source does not match BENCH_EXPECT_SOURCE: "
                     f"{tiling.get('tiling_source')!r}")
PY

IGNORE_INFER_ERROR=1 msprof --output="$output_dir/profile" \
    --task-time=on --ai-core=on \
    env LD_PRELOAD="$preload" "$script_dir/base_aclnn_bench" \
    --api "$api" --device "$device" \
    --m "$m" --n "$n" --k "$k" --warmup "$warmup" \
    --batches 1 --repetitions "$repetitions" \
    > "$output_dir/msprof_stdout.log" 2> "$output_dir/msprof_stderr.log"
python3 "$script_dir/parse_profile.py" "$output_dir/profile" \
    --last "$repetitions" \
    --kernel-type "${BENCH_PROFILE_KERNEL_TYPE:-KERNEL_AICORE}" \
    > "$output_dir/kernel_time.json"
merge_args=(
    --event "$output_dir/tiling_event.jsonl"
    --tiling "$output_dir/tiling.json"
    --profile "$output_dir/kernel_time.json"
    --operator-variant "$operator_variant"
    --expected-split-core "${BENCH_EXPECT_SPLIT_CORE_MODE:-0}"
    --expected-nd2nz-a "${BENCH_EXPECT_ND2NZ_A:-0}"
    --expected-nd2nz-b "${BENCH_EXPECT_ND2NZ_B:-0}"
)
if [[ -n "${MMV3_FORCE_USED_CORES:-}" ]]; then
    merge_args+=(--forced-used-cores "$MMV3_FORCE_USED_CORES")
fi
python3 "$script_dir/merge_case.py" "${merge_args[@]}" > "$output_dir/case.json"
cat "$output_dir/case.json"
