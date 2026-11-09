#!/usr/bin/env bash
# Reproduce the isolated FP32 cache-policy binary on the 910 machine.
set -euo pipefail
root=/home/xianfei/solvebase_bench_20261025
experiment="$root/fp32_mode2_cache_policy_v1"
prior="$root/fp32_mode2_in_context_pitch_v1/ops_nn_pitch_source/build/tbe"
cann=/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0
config=mat_mul_v3_binary_high_performance.json_4

test ! -e "$experiment/tbe_slim"
mkdir -p "$experiment/tbe_slim/dynamic" "$experiment/tbe_slim/ascendc" \
    "$experiment/tbe_slim/config/ascend910b/mat_mul_v3" "$experiment/binary_fast"
cp -a "$prior/ascendc/mat_mul_v3" "$experiment/tbe_slim/ascendc/"
cp "$prior/dynamic/mat_mul_v3.py" "$experiment/tbe_slim/dynamic/"
cp "$prior/config/ascend910b/mat_mul_v3/$config" \
    "$experiment/tbe_slim/config/ascend910b/mat_mul_v3/"
python3 "$experiment/patch_kernel.py" \
    "$experiment/tbe_slim/ascendc/mat_mul_v3/mat_mul_sc_splitk_kernel.h"
python3 "$experiment/slim_tiling_key.py" \
    "$experiment/tbe_slim/ascendc/mat_mul_v3/mat_mul_v3_tiling_key.h"

source "$cann/set_env.sh"
g++ -std=c++17 -O2 -I"$cann/include" -I"$cann/aarch64-linux/include" \
    "$experiment/fp32_cache_bench.cpp" -L"$cann/lib64" \
    -Wl,-rpath,"$cann/lib64" -lascendcl -lopapi -lnnopbase \
    -lacl_op_compiler -o "$experiment/fp32_cache_bench"
cd "$experiment"
asc_opc "$experiment/tbe_slim/dynamic/mat_mul_v3.py" \
    --main_func=mat_mul_v3 \
    --input_param="$experiment/tbe_slim/config/ascend910b/mat_mul_v3/$config" \
    --soc_version=Ascend910B1 --output="$experiment/binary_fast/" \
    --impl_mode=high_performance,optional --op_mode=dynamic
