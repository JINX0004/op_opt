# 910B BASE timing harness

`base_aclnn_bench.cpp` is the first device-side validation harness for the
[`SolveBase` measurement plan](../SOLVE_BASE_910B_MEASUREMENT_PLAN.md). It
builds against the machine's installed CANN and calls `MatMulV3` directly
through `aclopCompileAndExecuteV2` with preallocated FP16 ND tensors on one device.
It uses a nonuniform, exactly representable FP16 input pattern and checks
every output element. It records repeated
`aclrtEventElapsedTime` measurements in JSONL. Tensor creation, H2D copy,
and first-time compilation occur before timing. For direct MatMulV3, CANN
manages the reported workspace internally; only the comparison aclnn paths
use the harness's preallocated workspace buffer.

```bash
CANN_ROOT=/path/to/cann-9.0.0 bash build.sh
CANN_ROOT=/path/to/cann-9.0.0 bash run_hardware_probe.sh 0
source /path/to/cann-9.0.0/set_env.sh
export IGNORE_INFER_ERROR=1
./base_aclnn_bench --device 0 --m 256 --n 256 --k 1024 \
    --api v3cached --warmup 20 --batches 5 --repetitions 20
```

The installed CANN 9.0.0 package did not register MatMulV3's dtype-inference
function for the single-operator compile API. `IGNORE_INFER_ERROR=1` was
needed for this API; the harness provides explicit tensor shapes and dtypes
and checks all output elements against a uniform-input reference before
timing. This bypass needs revalidation if CANN or the operator package
changes. The optional `--api aclnn` and `--api mm` run `aclnnMatmul` and
`aclnnMm` for comparison. Both dispatched `MatMulV2` on the tested CANN
installation and are **not** valid SolveBase validation runs.
The source bundled in this CANN installation differs from this repository's
`mat_mul_v3` checkout, as recorded in the [pilot report](PILOT_910B3_CANN9.md).
The user's separate remote `ops-nn-sribd` source has now also been exercised:
see the [source-matched report](SOURCE_MATCHED_910B3_CANN9.md) and
[case records](source_matched_results.json). Set `BENCH_CUSTOM_OPP_PATH` to
the extracted vendor package and `BENCH_API=v3cached` to select it;
`MMV3_FORCE_USED_CORES` enables the source's core-count control. Direct V3
dispatch does not preload the custom aclnn API library. Set
`BENCH_EXPECT_SOURCE` to the expected source-path prefix to reject a case
if dispatch silently selects a different tiler.

The first V3 call compiles the operator. `--api v3cached` uses
`aclopExecuteV2` for later calls in the same process. The runner alone
reports `tiling_verified=false`: it does not control or read the installed
operator's tile fields. Its event interval can include a gap between
recording the start event and host submission of the operator. Use `msprof`
task time for device-kernel duration. `parse_profile.py` reads the exported
task-time CSV. A complete first-round case is:

```bash
CANN_ROOT=/path/to/cann-9.0.0 bash run_case.sh 256 256 1024 results/256x256x1024
```

`run_case.sh` makes an INFO-log correctness/tiling run and a separate
low-logging profiler run. `parse_tiling_log.py` extracts the source tiling
tuple, key, block dimension, conversion flags, and workspace from the
installed operator's INFO log. `merge_case.py` checks that the shape and
family agree, then writes `case.json`. Non-BASE or conversion-dependent
cases stop after the tiling run. `tiling_probe_event_us` is from the
INFO-log run and must **not** be used as a timing result; use
`kernel_median_us` from the profiler. The [pilot record](PILOT_910B3_CANN9.md)
and [machine-readable cases](pilot_results.json) show the tested CANN path.
The 256×256×1024 pilot selected
`singleCoreM/N=32/128`, `baseM/N/K=32/128/128`, 20 launch blocks, and
16 spatial output tiles. The model's *active* Cube-core count for this
shape is at most 16, even though the source launched 20 blocks. Do not
equate block dimension with useful work on every AIC.

Do not fit primitive coefficients from these whole-matmul timings. The
[primitive probe report](PRIMITIVE_910B3_CANN9.md) contains the Ascend C
Cube/MTE/Fixpipe probes, raw profiler task times, fitted exploratory slopes,
and source-checked controlled BASE tile comparisons. The source-valid
research override lives in an isolated remote clone; it never edits the
user's working source. Its first `baseM`-only variant failed the full output
check and was discarded. Shrinking `baseM` and `singleCoreM` together passed.
The current reduced equation does not accept the measured packet latencies,
so the illustrative solver profile has not been replaced.
The [five-round controlled tiling repeats](spatial_repeat_analysis.json)
show the normal tile wins every 256 and 1024 round and four of five 512
rounds. [`repeat_spatial.py`](repeat_spatial.py) reruns those comparisons on
the prepared isolated 910B3 checkout; its output is
[`spatial_repeat_raw.jsonl`](spatial_repeat_raw.jsonl).
The [optimizer-versus-default experiment](OPTIMIZER_VS_DEFAULT_910B3.md)
predeclares 12 BASE workloads and audits both a source-executable restricted
M-tile selector and the full reduced solver. The restricted selector chose
the source default on all 12 shapes; the full solver's different tuples
still need a source-backed finalizer and cannot support a speedup claim yet.
The [large-shape extension](LARGE_SHAPE_OPTIMIZER_VS_DEFAULT_910B3.md)
repeats this audit on seven workloads up to 8192×8192×4096 and adds six
randomized paired rounds for three large shapes. The restricted optimizer
again selects the default; its checked `baseM=64` alternative is 1.34–1.60×
slower than default on those three shapes, with paired 95% intervals above 1.
The later [source-compatible full BASE evaluation](FULL_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
uses the revised per-AIC objective, verifies the full emitted tuple and all
output elements, and runs four randomized paired rounds on the same seven
large shapes. The model ties the source's 128×256 default with its 256×128
choice; the latter measures 1.197× the default time on the declared suite.
It also records why outer `singleCoreM/N` values larger than `baseM/N` are
not executable by this tested plain BASE kernel path.
The subsequent [operand-shaped GM→L1 probe](directional_read/README.md)
measures A1 and B1 ND→NZ packets separately, with 48 randomized profiler
points. Its effective rates explain the tile orientation sign but overstate
the checked whole-kernel gap. The rates remain diagnostic and are not used
as a calibrated default profile.
The [Huawei memory-path diagram comparison](HUAWEI_DIAGRAM_RATE_COMPARISON.md)
converts the supplied bytes-per-cycle labels at 1.8 GHz and 20 AICs and
checks them against the measured primitives, including the read-cache and
Fixpipe path distinctions.
The [packet-aware SolveBase rerun](PACKET_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
uses the source-step MTE2 primitive profile on 12 new shapes. It matches
the default on 11; its only different tile is 15.94% slower over six
source-verified paired rounds. The reduced model's round-robin core load
does not reproduce the source BASE scheduler on that tail-heavy shape.
The [larger-shape follow-up](PACKET_BASE_XLARGE_910B3.md) checked eight
8192–16384-scale workloads. The optimizer and source default selected the
same 128×256 tile on all eight; the model underpredicted measured kernel
time by 3.7–20.5%, so this suite provides an accuracy check but no
distinct-tiling speedup comparison.
The [unrestricted-core diagnostic](UNRESTRICTED_BASE_XLARGE_DIAGNOSTIC.md)
turns off `base_core_equals_base` under the older byte-only objective. It
chooses larger outer tiles on five of those eight cases, but the predicted
gain against the same objective's constrained solve is at most 0.26%, and
the tested source BASE kernel does not compute those tiles correctly.
This applies to the plain BASE load mode. The
[B-in-L1 full-load path](FULL_LOAD_CORE_TILING_SUPPORT_910B3.md) is a
different BASE kernel: two source-backed 910B3 cases used larger core tiles
and passed full-output checks.
The [additional five-shape source-backed comparison](PACKET_BASE_EXPLORATORY_MORE_SHAPES_910B3.md)
tested four new different optimizer/default choices over randomized paired
rounds. Three swapped-tile choices were 5.1–13.0% slower; a 19-core choice
was effectively tied. The fifth shape selected the same tile.
The [abstract balanced-class rerun](abstract_balance_three_v1/README.md)
retested the three prior swapped-tile regressions after replacing the
reduced model's row-major assignment. The optimizer still chose 256×128
and was 6.1–13.9% slower than the source default over four new paired
rounds per shape. All 24 timed runs passed exact-output and tiling checks.
The [above-L2 working-set comparison](above_l2_v1/README.md) screened 12
further shapes with combined FP16 A+B+C footprints of 207–408 MiB. It
matched the source's 128×256 tile on nine shapes. Its three 256×128 choices
were slower by paired geometric means of 3.8%, 4.1%, and 12.3%; the first
shape's four-round 95% interval includes a tie. The matched M/N pair with
double K exposed a growing absolute model underprediction.
The [K>65535 Split-K experiment](k_gt_65535_v1/README.md) audits the
source's A ND→NZ conversion boundary. Three deterministic Split-K shapes
retain the same Cube tiling across K=65408/65536, but the converted path
takes 1.37–1.64× as long in four paired rounds. A larger shape selects
single-core Split-K instead. The current reduced solver chooses BASE for
all three boundary shapes and cannot represent one verified oversized
Split-K core tile.

To rebuild the primitive probe on the 910B3, use the CANN 9.0.0 installation
and its bundled `llvm-objdump` on `PATH`:

```bash
source /path/to/cann-9.0.0/set_env.sh
export PATH=/path/to/cann-9.0.0/tools/bisheng_compiler/bin:$PATH
cmake -S primitives -B primitives/build \
  -DASCEND_CANN_PACKAGE_PATH=/path/to/cann-9.0.0 \
  -DSOC_VERSION=Ascend910B3 -DCMAKE_BUILD_TYPE=Release
cmake --build primitives/build -j4
python3 primitives/sweep.py --group nd2nz_read_corrected
python3 primitives/sweep.py --group cube_write
python3 primitives/sweep.py --group launch_read_synced
python3 primitives/sweep.py --group hbm_read
python3 primitives/repeat_selected.py
python3 primitives/repeat_hbm.py
python3 primitives/analyze.py > primitives/analysis.json
```
