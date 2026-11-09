#!/usr/bin/env python3
"""Build a BF16-only narrow-B diagnostic host and restore the source tree."""

import hashlib
import shutil
import subprocess
from pathlib import Path

REPO = Path('/home/xianfei/dev/ops-nn-sribd-mip-selector')
BENCH = Path('/home/xianfei/solvebase_bench_20261025')
HEADER = REPO / 'experimental/mip_selector/model/hardware_profile.h'
BUILD = REPO / 'build_mip'
LIB = BUILD / 'libcust_opmaster_rt2.0.so'
BASE_PACKAGE = BENCH / 'ops_nn_bf16_mode5_revised_package'
NEW_PACKAGE = BENCH / 'ops_nn_bf16_narrow_rows_diagnostic_package'
RELATIVE_LIB = Path('op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so')
BASE_SHA = '3e0d3a91ad4efcf50ce5af3d05ccbd452c45b8d105fc58f9edf2883d7e053471'
ANCHOR = '    hardware.primitives["sc_mode5_iterate"] = {"count", 1.76, 1.0};\n'
ADDITION = (
    '    // Paired BF16/FP16 910B3 packet probes found equivalent 16-wide B reads.\n'
    '    // Diagnostic only: use the established FP16 narrow-row coefficients.\n'
    '    hardware.primitives["base_mte2_b_narrow_rows_one"] = '
    '{"rows", 0.0, 124.25456613790051};\n'
    '    hardware.primitives["base_mte2_b_narrow_rows_full"] = '
    '{"rows", 0.0, 85.48932166157344};\n'
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(label):
    log = BENCH / f'bf16_narrow_rows_{label}_build.log'
    with log.open('w') as output:
        proc = subprocess.run(['cmake', '--build', str(BUILD), '--target',
                               'cann-ops-nn-bench910-linux.aarch64_ascendc_cust_opmaster',
                               '-j8'], stdout=output,
                              stderr=subprocess.STDOUT, cwd=REPO)
    if proc.returncode:
        raise RuntimeError(f'{label} host build failed: {log.read_text()[-3000:]}')
    print(label, sha(LIB), log, flush=True)


def main():
    original = HEADER.read_text()
    if original.count(ANCHOR) != 1 or ADDITION in original:
        raise RuntimeError('unexpected hardware profile source')
    if sha(LIB) != BASE_SHA or sha(BASE_PACKAGE / RELATIVE_LIB) != BASE_SHA:
        raise RuntimeError('current build/package is not the measured BF16 baseline')
    patched = original.replace(ANCHOR, ANCHOR + ADDITION, 1)
    try:
        HEADER.write_text(patched)
        build('diagnostic')
        shutil.copytree(BASE_PACKAGE, NEW_PACKAGE, dirs_exist_ok=True)
        shutil.copy2(LIB, NEW_PACKAGE / RELATIVE_LIB)
        print('diagnostic_package', NEW_PACKAGE, sha(NEW_PACKAGE / RELATIVE_LIB), flush=True)
    finally:
        HEADER.write_text(original)
        build('restored')
        if sha(LIB) != BASE_SHA:
            raise RuntimeError('restored host library hash does not match baseline')


if __name__ == '__main__':
    main()
