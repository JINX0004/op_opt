#!/usr/bin/env bash
set -euo pipefail
repo=/home/xianfei/dev/ops-nn-sribd-mip-selector
src="$repo/matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp"
backup=$(mktemp /tmp/matmul_v3_base_tiling.XXXXXX.cpp)
cp -p "$src" "$backup"
restore() {
  cp "$backup" "$src"
  touch "$src"
  rm -f "$backup"
}
trap restore EXIT
python3 - "$src" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text()
needle='bool MatmulV3BaseTiling::CheckSingleTilingOk(MatmulV3RunInfo &tmpRunInfo)\n{\n'
assert s.count(needle)==1
injection='''bool MatmulV3BaseTiling::CheckSingleTilingOk(MatmulV3RunInfo &tmpRunInfo)
{
    // Temporary device experiment: apply the selector's mode-2 geometry to
    // exactly one FP32 shape. The production branch is restored after build.
    if (std::getenv("MMV3_FORCE_MODE2_MODEL_TILE") != nullptr && mipMode_ == 2 &&
        args_.aType == ge::DT_FLOAT && args_.mValue == 2560 &&
        args_.nValue == 768 && args_.kValue == 32768) {
        tmpRunInfo.singleCoreM = 128;
        tmpRunInfo.singleCoreN = 768;
        tmpRunInfo.usedCoreNum = 20;
        OP_LOGI(args_.opName, "Forced model mode-2 tile: singleCoreM=128 singleCoreN=768 singleCoreK=%lu",
                tmpRunInfo.stepKa * tmpRunInfo.baseK);
    }
'''
p.write_text(s.replace(needle,injection))
PY
source /home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0/set_env.sh
cmake --build "$repo/build_mip" --target cann-ops-nn-bench910-linux.aarch64_ascendc_cust_opmaster -j4
base=/home/xianfei/solvebase_bench_20261025
package="$base/ops_nn_fp32_mode2_model_tile_package"
if [[ -e "$package" ]]; then
  echo "package already exists: $package" >&2
  exit 1
fi
cp -a "$base/ops_nn_fp32_source_base_fix_package" "$package"
cp "$repo/build_mip/libcust_opmaster_rt2.0.so" "$package/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
sha256sum "$package/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
restore
trap - EXIT
cmake --build "$repo/build_mip" --target cann-ops-nn-bench910-linux.aarch64_ascendc_cust_opmaster -j4
cd "$repo"
git status --short matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp
