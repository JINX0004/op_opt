# MatMulV3 converted-A measurement on 910B3

The follow-up [in-context AIC phase measurement](PHASE_MEASUREMENT.md)
separates the conversion handoff wait from each family's AIC matmul work.

This experiment isolates the FP16 `MatrixAtoNZV2` vector routine used by
MatMulV3 BASE and deterministic Split-K when `M=2560`, `N=1024`, and
`K>65535`. It measures the source-emitted conversion tiles at K=65536 and
73728, plus the other tile at each K. The operator still uses its own mixed
Cube/vector kernel and synchronization, so these are measurements of the
conversion routine in a standalone 40-AIV kernel, **not direct timestamps
inside the complete operator**.

## Source correspondence and procedure

Both kernel families call `MatrixAtoNZV2` in the remote selector branch
`codex/matmul-v3-mip-selector` (`bc40c260`). The default package used for
the full Split-K sweep has a byte-identical `mat_mul_nd2nz.h` header. The probe in
[`probe_kernel.cpp`](probe_kernel.cpp) calls that same header function with
`M`, `K`, `baseAN`, `baseAD`, FP16, and no transpose. Its `usedCoreNum` is set
so that `MatrixAtoNZV2` partitions the conversion into 40 AIV workloads,
matching the source's 20 AIC / 40 AIV launch. For these shapes the function
chooses its `MatrixtoNZ` / `CopyPadNd2Nz` path. The output buffer has the
same `2MK` byte converted-A region. The runner checks that every byte was
written for a constant input and that all 40 AIVs reached the completion
token; this checks coverage and completion but does not independently prove
the ND-to-NZ permutation.

The probe was built against CANN 9.0.0 for Ascend910B3, in Release mode.
`sweep.py` launched `msprof --task-time=on --ai-core=on` on devices 4–7
concurrently, after checking each was idle. Each profile used 10 warmups and
8 profiled repetitions. The table below reports the median of the 8
`AI_VECTOR_CORE` task times per profile, then the median across profiles.
Each source tile has two profiles per device; each alternative has one.
Output checks and device-idle checks passed in all 28 profiles. The
[machine-readable summary](sweep_summary.json) retains every profile's
samples and the remote raw-profile path. The remote executable and profiles
are under `/home/xianfei/solvebase_bench_20261025/converted_a_probe_v1`.

| K | `baseAN × baseAD` | Source tile? | Profiles | Task time, median µs | Profile-median range µs | Approx. converted bytes/s* |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 65536 | 19 × 2048 | yes | 8 | 1251.315 | 1246.186–1255.040 | 536 GB/s |
| 65536 | 47 × 1024 | no | 4 | 781.226 | 780.356–786.850 | 859 GB/s |
| 73728 | 19 × 2048 | no | 4 | 1503.480 | 1494.470–1511.940 | 502 GB/s |
| 73728 | 47 × 1024 | yes | 8 | 963.275 | 954.332–970.498 | 784 GB/s |

\*Read plus write bytes (`4MK`) divided by task time. This is an effective
end-to-end throughput, including conversion work; it is not a GM link rate.
The empty vector-kernel baseline was 5.105 µs median across the four
devices. Subtracting that baseline from the source-tile measurements gives
about **1246.21 µs** at K=65536 and **958.17 µs** at K=73728.

The source changes `baseAN × baseAD` from 19 × 2048 to 47 × 1024 as K
increases. The latter lowers the isolated conversion task by 470.09 µs at
K=65536 and 540.21 µs at K=73728. Thus the source's converted-A routine is
about 288.04 µs faster at K=73728 than at K=65536 despite processing 12.5%
more elements. `MatrixtoNZ` partitions
`ceil(M/baseAN) × ceil(K/baseAD)` work units over 40 AIVs; this yields
4320 units for the source K=65536 tile and 3960 for the source K=73728
tile. The size and cost of each unit also change, so the unit count alone
does not predict runtime.

## Relation to the selector error

The online model's `ConversionUs` uses `2MK` bytes of read, `MK` ND-to-NZ
elements, `2MK` bytes of write, and one sync. With its current exploratory
rates, it predicts **322.75 µs** at K=65536 and **363.04 µs** at K=73728;
`baseAN` and `baseAD` are absent from that term. Those figures are much
lower than the isolated source-routine measurements and predict the wrong
K trend. The difference is diagnostic, not a calibrated correction to add
to full-kernel time: vector work, memory traffic, and synchronization can
overlap differently inside the full operator.

Using the [full-kernel BASE](../online_base_read_calibration_v1/long_k_base_sweep_summary.json)
and [full-kernel Split-K](../online_base_read_calibration_v1/long_k_default_sweep_summary.json)
measurements, a simple *inferred remainder* is full time minus the isolated
conversion time after the empty-kernel baseline:

| K | BASE full µs | Split-K full µs | Conversion proxy µs | BASE remainder µs | Split-K remainder µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65536 | 3280.26 | 2795.05 | 1246.21 | 2034.05 | 1548.84 |
| 73728 | 3276.23 | 2653.41 | 958.17 | 2318.06 | 1695.24 |

These remainders are **not measured Cube-phase times**. The standalone
probe omits the AIV-to-AIC handoff, changes cache context, and may miss
overlap or contention present in the source mixed kernel. The two families
use the same A conversion routine and tile for each K, while their full
times differ by 485.21 µs at K=65536 and 622.82 µs at K=73728. A common
conversion correction therefore cannot fix the wrong BASE-versus-Split-K
ranking. The [completed in-context phase sweep](PHASE_MEASUREMENT.md)
measures the AIC handoff wait and main matmul interval for each family.
It also evaluates the subsequent L1-step read-startup revision. A
source-tiler `baseAN/baseAD` sweep in the complete kernels remains useful
for modeling the conversion tile's effect on absolute time.

## Reproduction

On the 910 machine, with this directory copied to the remote path above:

```bash
source /home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0/set_env.sh
export PATH=/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0/tools/bisheng_compiler/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0 \
  -DMMV3_KERNEL_SOURCE=/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/op_kernel
cmake --build build -j4
python3 sweep.py
```
