# Source-matched 910B3 MatMulV3 validation

These [case records](source_matched_results.json) use the operator from
`/home/xianfei/dev/ops-nn-sribd/matmul/mat_mul_v3` on the 910B3 machine,
not CANN's bundled MatMulV3. The repository is at commit
`b2cdb471de09d93eb79b34812c1ebe4e938ae10c`; its
`matmul_v3_base_tiling.cpp` has user changes, including
`MMV3_FORCE_USED_CORES`, and SHA-256
`5bb53987aa5ef6d46611eb5d577aeecdeb077913bab9e617be272fdbee48f4d1`.
The original remote repository was not edited by this experiment.

The operator package already present in that remote repository was extracted to
`/home/xianfei/solvebase_bench_20261025/ops_nn_package`. For isolated
`aclopCompileAndExecuteV2` runs, CANN 9.0.0 reported that its legacy
`TilingPrepareForOpCache` hook was unavailable. We cloned the source into
`/home/xianfei/solvebase_bench_20261025/ops_nn_source_build`, copied the
user-modified base tiler into it, and applied the benchmark-only
[`enable_aclop_source_bench.py`](enable_aclop_source_bench.py) change to
`matmul_v3_tiling.cpp`. With `MMV3_SKIP_LEGACY_OP_CACHE=1`, that optional
preparation is skipped; without it, the original code path remains. The
host tiling library was built against installed CANN 9.0.0 and placed into
a separate package at
`/home/xianfei/solvebase_bench_20261025/ops_nn_source_package`.
All kernel binaries and other package files came unchanged from the existing
package. The rebuilt `libcust_opmaster_rt2.0.so` has SHA-256
`2b7771345604073001b55293e10b8d21af3dc7328b934ec55f01c1b8ebbbabae`.
The host tiler is rebuilt from the identified source. The existing kernel
binaries were reused; their exact source revision has not been independently
attested by a fresh complete kernel rebuild.

The direct MatMulV3 runner used `IGNORE_INFER_ERROR=1`, explicit FP16 tensor
descriptors, and the benchmark-only cache switch. The CANN INFO log names
the isolated source path, so operator selection was checked independently
of the environment variables. All cases emitted BASE key 65536 with
`nd2nzA=nd2nzB=0`, and every output element matched the patterned FP16
reference exactly. The profiler reports actual MatMulV3 task times:

| M×N×K | Launch blocks | Spatial tiles | Median kernel time | Range, five tasks |
| --- | ---: | ---: | ---: | ---: |
| 256×256×1024 | 20 | 16 | 8.520 µs | 8.500–8.544 µs |
| 512×512×1024 | 20 | 20 | 10.040 µs | 9.964–10.140 µs |
| 1024×1024×4096 | 20 | 32 | 45.104 µs | 45.024–45.324 µs |
| 256×256×1024, `MMV3_FORCE_USED_CORES=8` | 8 | 16 | 8.880 µs | 8.700–9.460 µs |

For the forced-core case, the source log explicitly says it clamped
`usedCoreNum` to 8, and the launch block dimension is 8; its spatial and
base tiles are unchanged. This is one controlled core-count comparison,
but one short batch per setting is insufficient to establish a reliable
performance ranking. It does show that the source control can be exercised
by the harness without corrupting the output.
A later three-task source-verified repeat of the 20-block 256 case had an
8.780 µs median. The repeat passed `BENCH_EXPECT_SOURCE`, which checks the
source path parsed from the emitted tiling log before profiling.

The raw profiler exports and tiling logs are retained under
`/home/xianfei/solvebase_bench_20261025/cases/source_matched_clean_*`.
The compact checked-in JSON contains every selected kernel duration and
tiling feature. These whole-matmul runs still cannot identify separate
Cube, read, output, and startup parameters. A tile override and isolated
primitive probes remain necessary before replacing the illustrative
`SolveBase` timing profile.

To rerun against the prepared isolated package:

```bash
MMV3_SKIP_LEGACY_OP_CACHE=1 \
BENCH_API=v3cached \
BENCH_CUSTOM_OPP_PATH=/home/xianfei/solvebase_bench_20261025/ops_nn_source_package \
BENCH_EXPECT_SOURCE=/home/xianfei/solvebase_bench_20261025/ops_nn_source_build/matmul/mat_mul_v3/ \
CANN_ROOT=/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0 \
bash /home/xianfei/solvebase_bench_20261025/run_case.sh \
  256 256 1024 /home/xianfei/solvebase_bench_20261025/cases/recheck 0 10 10
```
