#!/usr/bin/env bash
# Rebuild the source variant that omits explicit normal-cache calls.
set -euo pipefail
root=/home/xianfei/solvebase_bench_20261025/fp32_mode2_cache_policy_v1
cann=/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0
variant="$root/no_normal"
config=mat_mul_v3_binary_high_performance.json_4
test ! -e "$variant/tbe"
mkdir -p "$variant/binary"
cp -a "$root/tbe_slim" "$variant/tbe"
python3 "$root/patch_no_normal.py" \
    "$variant/tbe/ascendc/mat_mul_v3/mat_mul_sc_splitk_kernel.h"
source "$cann/set_env.sh"
cd "$variant"
asc_opc "$variant/tbe/dynamic/mat_mul_v3.py" \
    --main_func=mat_mul_v3 \
    --input_param="$variant/tbe/config/ascend910b/mat_mul_v3/$config" \
    --soc_version=Ascend910B1 --output="$variant/binary/" \
    --impl_mode=high_performance,optional --op_mode=dynamic
bash "$root/make_no_normal_overlay.sh"
