#!/usr/bin/env bash
set -euo pipefail
root=/home/xianfei/solvebase_bench_20261025/fp32_mode2_stage_ladder_v1
cann=/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0
source "$cann/set_env.sh"
cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release \
    -DSOC_VERSION=Ascend910B3 -DASCEND_CANN_PACKAGE_PATH="$cann"
cmake --build "$root/build" -j4
