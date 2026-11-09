#!/usr/bin/env bash
# Assemble the isolated experiment as soon as the patched FP32 kernel is ready.
set -euo pipefail

root=/home/xianfei/solvebase_bench_20261025
experiment="$root/fp32_mode2_in_context_pitch_v1"
source_root="$experiment/ops_nn_pitch_source"
baseline="$root/ops_nn_fp32_forced_family_package"
package="$experiment/ops_nn_pitch_overlay_package"
library=op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
kernel=MatMulV3_ND_ND_ND_ND_FP32_FP32_FP32_FP32_high_performance
source_kernel="$source_root/build/binary/ascend910b/bin/ascend910b/mat_mul_v3"
package_kernel=op_impl/ai_core/tbe/kernel/ascend910b/mat_mul_v3

test -f "$source_root/build/libcust_opmaster_rt2.0.so"
test -f "$source_kernel/$kernel.o"
test -f "$source_kernel/$kernel.json"
test ! -e "$package"
cp -a "$baseline" "$package"
cp "$source_root/build/libcust_opmaster_rt2.0.so" "$package/$library"
cp "$source_kernel/$kernel.o" "$package/$package_kernel/$kernel.o"
cp "$source_kernel/$kernel.json" "$package/$package_kernel/$kernel.json"
sha256sum "$package/$library" "$package/$package_kernel/$kernel.o" \
    "$package/$package_kernel/$kernel.json" > "$experiment/overlay_hashes.txt"
cat "$experiment/overlay_hashes.txt"
