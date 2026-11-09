#!/usr/bin/env bash
# Assemble a research-only package with the prior fixed-work host tiler.
set -euo pipefail
root=/home/xianfei/solvebase_bench_20261025
experiment="$root/fp32_mode2_cache_policy_v1"
prior="$root/fp32_mode2_in_context_pitch_v1/ops_nn_pitch_overlay_package"
package="$experiment/ops_nn_cache_policy_package"
kernel=MatMulV3_ND_ND_ND_ND_FP32_FP32_FP32_FP32_high_performance
kernel_dir=op_impl/ai_core/tbe/kernel/ascend910b/mat_mul_v3
host_lib=op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so

test -f "$experiment/binary_fast/$kernel.o"
test -f "$experiment/binary_fast/$kernel.json"
test ! -e "$package"
cp -a "$prior" "$package"
cp "$experiment/binary_fast/$kernel.o" "$package/$kernel_dir/$kernel.o"
cp "$experiment/binary_fast/$kernel.json" "$package/$kernel_dir/$kernel.json"
sha256sum "$package/$host_lib" "$package/$kernel_dir/$kernel.o" \
    "$package/$kernel_dir/$kernel.json" "$experiment/fp32_cache_bench" \
    > "$experiment/package_hashes.txt"
cat "$experiment/package_hashes.txt"
