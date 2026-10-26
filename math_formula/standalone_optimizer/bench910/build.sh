#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cann_root="${CANN_ROOT:-}"
if [[ -z "$cann_root" ]]; then
    atc_path="$(command -v atc || true)"
    if [[ -z "$atc_path" ]]; then
        printf 'Set CANN_ROOT to the installed CANN directory (atc is not on PATH).\n' >&2
        exit 2
    fi
    cann_root="$(cd "$(dirname "$atc_path")/.." && pwd)"
fi
output="${1:-$script_dir/base_aclnn_bench}"
driver_lib="${ASCEND_DRIVER_LIB:-/usr/local/Ascend/driver/lib64/driver}"

g++ -std=c++17 -O2 -Wall -Wextra \
    -I"$cann_root/include" \
    "$script_dir/base_aclnn_bench.cpp" \
    -L"$cann_root/lib64" -Wl,-rpath,"$cann_root/lib64" \
    -Wl,-rpath-link,"$driver_lib" \
    -lascendcl -lopapi -lnnopbase -lacl_op_compiler \
    -o "$output"
printf 'Built %s against %s\n' "$output" "$cann_root"

g++ -std=c++17 -O2 -Wall -Wextra \
    -I"$cann_root/include" \
    -I"$cann_root/aarch64-linux/asc/include/tiling" \
    "$script_dir/hardware_probe.cpp" \
    -L"$cann_root/lib64" -Wl,-rpath,"$cann_root/lib64" \
    -Wl,-rpath-link,"$driver_lib" \
    -lascendcl -ltiling_api -lplatform -lunified_dlog \
    -o "$script_dir/hardware_probe"
printf 'Built %s against %s\n' "$script_dir/hardware_probe" "$cann_root"
