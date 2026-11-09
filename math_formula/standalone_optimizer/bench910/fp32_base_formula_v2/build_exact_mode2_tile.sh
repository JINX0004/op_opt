#!/usr/bin/env bash
set -euo pipefail

root=/home/xianfei/solvebase_bench_20261025
experiment="${MMV3_EXACT_EXPERIMENT_DIR:-$root/fp32_base_formula_v2/exact_mode2_tile}"
source_file="$root/ops_nn_source_build/matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp"
backup="$experiment/matmul_v3_base_tiling.original.cpp"
baseline="$root/ops_nn_fp32_forced_family_package"
package="${MMV3_EXACT_PACKAGE_PATH:-$root/ops_nn_fp32_mode2_exact_tile_package}"
library=op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so

mkdir -p "$experiment"
test ! -e "$backup"
test ! -e "$package"
cp "$source_file" "$backup"
restore_source() { cp "$backup" "$source_file"; }
trap restore_source EXIT

if [[ "${MMV3_EXACT_PATCH_MODE:-}" == neighbors ]]; then
    python3 "$experiment/enable_exact_mode2_tile.py" "$source_file" neighbors
else
    python3 "$experiment/enable_exact_mode2_tile.py" "$source_file"
fi
set +u
source /home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0/set_env.sh
set -u
cmake --build "$root/ops_nn_source_build/build" \
    --target cann-ops-nn-bench910-linux.aarch64_ascendc_cust_opmaster -j8 \
    > "$experiment/build_host.log" 2>&1
cp -a "$baseline" "$package"
cp "$root/ops_nn_source_build/build/libcust_opmaster_rt2.0.so" "$package/$library"
sha256sum "$backup" "$source_file" "$baseline/$library" "$package/$library" \
    > "$experiment/build_hashes.txt"
tail -n 8 "$experiment/build_host.log"
cat "$experiment/build_hashes.txt"
