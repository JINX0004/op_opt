#!/usr/bin/env bash
# Control build in which policy 0 makes no explicit L2 cache-hint call.
set -euo pipefail
root=/home/xianfei/solvebase_bench_20261025/fp32_mode2_cache_policy_v1
source_package="$root/ops_nn_cache_policy_package"
package="$root/no_normal/ops_nn_cache_policy_package"
kernel=MatMulV3_ND_ND_ND_ND_FP32_FP32_FP32_FP32_high_performance
kernel_dir=op_impl/ai_core/tbe/kernel/ascend910b/mat_mul_v3
test -f "$root/no_normal/binary/$kernel.o"
test -f "$root/no_normal/binary/$kernel.json"
test ! -e "$package"
cp -a "$source_package" "$package"
cp "$root/no_normal/binary/$kernel.o" "$package/$kernel_dir/$kernel.o"
cp "$root/no_normal/binary/$kernel.json" "$package/$kernel_dir/$kernel.json"
sha256sum "$package/$kernel_dir/$kernel.o" "$package/$kernel_dir/$kernel.json" \
    > "$root/no_normal/package_hashes.txt"
