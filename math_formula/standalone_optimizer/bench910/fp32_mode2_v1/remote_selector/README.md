# Experimental MatMulV3 MIP family selector

This experiment chooses the **implementation family and single-core Split-K
path online** in the CANN host tiler. It evaluates the bounded legal choices
with the calibrated reduced objective and takes their minimum. The previous
HiGHS model had only one-hot selection rows after evaluating those choices,
so no external solver is needed. The host still calculates the actual block
and buffer tiling by default. `MMV3_ONLINE_TILE=1` additionally tests the
selected BASE K tile when the source emits the same spatial tile. The source
recalculates L1 depths and K steps; an incompatible tuple falls back to the
source tile. This switch does not inject Split-K tiles.

The shared measured 910B3 primitive profile is in
`model/hardware_profile.h`.

`MMV3_ONLINE_ROW_READ_PROFILE=1` selects a conservative experimental FP16
BASE read profile. It retains the established generic objective for wider
tiles and adds a calibrated B-row service only when the ND row is shorter
than 64 bytes. The narrow-row contention term is quadratic in occupied-core
fraction; device-4 probes measured one, four, ten, and 20 cores.
`MMV3_ONLINE_ROW_READ_PROFILE=full` selects the broader source-step packet
fit for diagnostics. Paired source-backed trials found two large-shape
family regressions with that full profile, so it must remain experimental.
BF16 ignores this FP16-only environment switch. Its experimental profile
now includes the measured 16-wide B-row coefficient independently. Set the
variable before the process starts; online decisions are cached by shape and
type signature.

## Data types

The reduced model resolves a type signature `(A, B, output, accumulator)`
for every workload. Legacy `fp16` and `bf16` CLI rows map to
`(dtype, dtype, dtype, fp32)`. Transfer volumes and local/workspace footprints
use the relevant element widths. The source-backed feasibility domain
admits same-type FP16, BF16 and experimental FP32 with FP32 accumulation.
Mixed-output signatures remain outside this domain. The host continues to
use the source's emitted block and buffer tiles.

The FP16 profile and family objective are unchanged. The BF16 profile is
**experimental**. It starts with the shared 910B3 byte-service baseline and
now has BF16 source-in-context mode-5 Fixpipe and `IterateAll` measurements:
108000 bytes/µs and 1.76 µs per call on the `K=1536`, `N=384`, large-M
channel. Paired BF16/FP16 packet probes also measured equivalent 16-wide
B ND-to-L1 service on 910B3, so BF16 now uses the established narrow-row
coefficients for BASE when each B source row is shorter than 64 bytes.
For BF16 family selection, the source's prepared BASE tile is passed to
the reduced objective. Plain BASE uses one L0C buffer, so that candidate
gets a one-buffer preflight. On the measured narrow-B path, read startup
uses source `stepKa` and `stepKb` L1 packets rather than each `baseK`
slice. The balanced core-work assignment and overlap remain approximate.
Other BF16 Cube, reduction, and output rates remain provisional.
The online host keeps the source family decision for BF16 by default. Set
`MMV3_ONLINE_BF16_EXPERIMENT=1` together with `MMV3_ONLINE_SELECTOR=1` for
BF16 source-backed experiments. The diagnostic CLI always evaluates BF16.

The [BF16 mode-5 calibration and held-out device report](../../bf16_mode5_calibration_v1/README.md)
records the primitive derivation and paired comparison with clean master.
For large-M `N=384`, `K=1536`, the revised selector chooses mode 5 and
source-sized tiles. Five paired shapes emitted the same mode-5 tiling as
master and ran within about 1% of master's kernel time. The BF16
MatMulV3 benchmark uses the same benchmark-only dtype-inference proto
patch in both packages; that patch is not part of the product branch.
The [earlier BF16 comparison](../../bf16_online_vs_master_20261103/README.md)
records the generic-model BASE regressions that motivated calibration.
The [BF16 narrow-B boundary experiment](../../bf16_narrow_b_calibration_v1/README.md)
measured packet slopes, scored the source-emitted BASE tile, and validated
the corrected family on two narrow long-K shapes.
The [source-BASE BF16 follow-up](../../bf16_source_base_v1/README.md)
compares the source-prepared BASE tile and reports a fresh 100-shape 910B3
run: 18 family changes, all faster online in both timing orders, and a
1.0532× geometric mean across the selected dataset. Two cases did not
activate the online selection gate; BF16 remains opt-in.

Run `ctest --test-dir /tmp/mmv3-mip-build --output-on-failure` after building
to check dtype resolution, preflight and family coverage. The FP16 regression
also compares 30 saved selector rows exactly during development.

In paired device-4 FP16 source-backed checks with the source tile still in
control, the narrow profile changed 16×16×102400 from BASE (about 934 µs)
to deterministic Split-K (about 43 µs). Seven nearby N=16 or N=32 shapes
also selected the faster family (2.11–18.86×). The broad `full` profile
regressed on 2560×1024×36864 and 1024×2560×36864, so keep it diagnostic.
For the N=32 checks, the model scored a hypothetical BASE tile with
`baseN=16` while forced source BASE emitted `baseN=32`; the observed family
improvement is real but does not establish that the row cost accurately
describes the emitted BASE tile. The BF16 BASE follow-up now scores its
source-prepared tile. FP16 and the other implementation families still have
the source-versus-model geometry limitation described above.

The model compares these implementation codes:

| Code | Source implementation |
| --- | --- |
| 0 | BASE |
| 2 | Single-core Split-K, ordinary path |
| 3 | Deterministic multi-core Split-K |
| 5 | Single-core Split-K, NK33 path |
| 6 | Single-core Split-K, GM-to-L1 path |

## Build the optional diagnostic CLI

The CLI uses the same direct search and profile as the host and prints a CSV
for inspection. The host does not read that CSV. No HiGHS dependency remains.

```bash
cmake -S experimental/mip_selector -B /tmp/mmv3-mip-build
cmake --build /tmp/mmv3-mip-build -j4
printf '2560 1024 36864 fp16\n' | /tmp/mmv3-mip-build/mmopt_selector
```

`fp16`, `bf16`, and experimental `fp32` are accepted. FP32 rows use
`ExperimentalFp32BaseMte2910B3()`, including the narrow-N MTE2 service.
The model preflight rejects unsupported shapes. The CLI exits nonzero
for a malformed row or model failure.

## Build and use the host tiler

Build the host target against the installed CANN package, with `mat_mul_v3`
enabled. The 910B3 checkout was built with:

```bash
source /home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0/set_env.sh
cmake -S . -B build_mip -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0 \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DASCEND_COMPILE_OPS='mat_mul_v3;batch_mat_mul_v3;fused_mat_mul;matmul.common' \
  -DASCEND_OP_NAME=mat_mul_v3 -DVENDOR_NAME=bench910 \
  -DENABLE_CUSTOM=ON -DENABLE_PACKAGE=ON -DENABLE_BINARY=ON \
  -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON \
  -DASCEND_CHECK_OPTYPE_DUPLICATE=OFF
cmake --build build_mip \
  --target cann-ops-nn-bench910-linux.aarch64_ascendc_cust_opmaster -j4
```

Install/package the resulting `build_mip/libcust_opmaster_rt2.0.so` using the
same CANN package layout as the baseline host library. For the isolated
`aclop` benchmark package on this machine, the library is at
`op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/` within the package. The
experiment package used for the smoke test is
`/home/xianfei/solvebase_bench_20261025/ops_nn_mip_branch_package`.

Set `MMV3_ONLINE_SELECTOR=1` only in the experimental arm. Leave it unset for
the original source decision. The online choice applies only to ND,
nontransposed, no-bias, same-type FP16/BF16/FP32 A, B, and output on a
20-AIC 910B3 with the expected capabilities. FP32 uses the experimental
BASE MTE2 profile only with `MMV3_ONLINE_SELECTOR=1`; the default FP32
path remains source selection. FP32 mode 5 additionally requires
`MMV3_FP32_MODE5_EXPERIMENT`, and FP32 mode 6 is excluded. Invalid or
unsupported model inputs fall back to the source selector. Repeated
`(M,N,K,A,B,output,accumulator)` choices are cached in the process.

For the FP16/BF16 BASE K-tile experiment, set both
`MMV3_ONLINE_SELECTOR=1` and `MMV3_ONLINE_TILE=1`. The host keeps the source's spatial tile and launch
count, and replaces `baseK` only when the model chose the same
`singleCoreM/N/K` and `baseM/N` and the tile passes buffer checks. It then
recalculates `depthA1/B1` and `stepKa/Kb` before serialization. For
16×16×102400 FP16, the model proposes `baseK=1024` while the unmodified
source uses `baseK=64`. Validate correctness and device latency before
extending this path to other geometries or families. FP32 online selection
does not inject a model K tile, even if `MMV3_ONLINE_TILE=1` is set.

For the isolated `aclop` benchmark, also set
`MMV3_SKIP_LEGACY_OP_CACHE=1`. Its package does not provide the optional
legacy cache hook. Normal operator use leaves this variable unset.

To compare kernel times, run the same shapes and package with the online
switch unset and set, respectively, on the same idle device. Record the actual
`tilingEnableSplitCore`, tiling geometry, kernel key, correctness error, and
per-kernel latency in both arms. The logged `estimated_us` is a model value,
not a measured result. Without `MMV3_ONLINE_TILE=1`, this host integration
selects only family and path.
The first tiling request for each shape runs the bounded search; subsequent
requests use the cache. Measure compile/tiling overhead separately from
kernel latency when evaluating end-to-end performance.

## 100-shape FP16 device comparison

The [paired 910B3 report](../../online_100_vs_master_fp16_v1/README.md)
compares this branch with clean MatMulV3 master on 100 FP16 shapes.
The updated geometric mean master/online kernel-time ratio is 1.0189×:
1.2318× for ten changed-family cases and 0.9977× for 90 cases with
identical emitted tiling. The report preserves raw device summaries,
eight-round reruns of two large-M cases, and the analysis script.

## Experimental plain FP32 model

The [first FP32 source audit and timing report](../../fp32_plain_v1/README.md)
extends the reduced model and CLI to same-type FP32, NN, ND/ND/ND,
no bias, and HF32 disabled. The experimental search now compares plain
BASE, deterministic multi-core Split-K, and FP32 single-core Split-K
mode 2. FP32 mode 2 uses `baseK=64` and direct FP32 AIC output, so it has
its own geometry and output timing path. The [source domain description](FP32_SOURCE_DOMAIN.md)
gives the constraints and current validation. The FP32 Cube rate was
fitted to one source BASE tile; the mode-2 output service reuses a measured
FP32 atomic-output rate as a provisional proxy. FP32 converted-A
(`K>65535`) is still outside this domain. The default FP32 source path
remains unchanged; the experimental online path requires
`MMV3_ONLINE_SELECTOR=1`.
