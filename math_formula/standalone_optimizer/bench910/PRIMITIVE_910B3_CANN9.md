# SolveBase primitive measurements on 910B3: exploratory result

The primitive probes and a controlled BASE tiling comparison ran on the
910B3. They **do not yet justify a production `SolveBase` timing profile**.
The original reduced equation's serial global read-packet latency term was
incompatible with the measured per-core latency: inserting the probe
coefficients predicted 35–485 µs for source-matched kernels measured at
8.5–45 µs. A simple per-core wave count also failed absolute-time
validation. The current solver uses a more specific busiest-AIC assignment
model (see [the later large-shape audit](LARGE_SHAPE_OPTIMIZER_VS_DEFAULT_910B3.md#full-solver-status)),
but the probe values still require held-out calibration and validation.

## Provenance and method

- Device 0 is a 910B3 with 20 AICs, CANN 9.0.0, driver 25.3.rc1. The
  [hardware query](hardware_910b3.json) records the usable buffer sizes.
- The remote source is `/home/xianfei/dev/ops-nn-sribd/matmul/mat_mul_v3`
  at commit `b2cdb471de09d93eb79b34812c1ebe4e938ae10c`. The user's
  modified `matmul_v3_base_tiling.cpp` had SHA-256
  `5bb53987aa5ef6d46611eb5d577aeecdeb077913bab9e617be272fdbee48f4d1`
  and was not edited. The controlled-tiling host library was built from an
  isolated source clone; its package library SHA-256 is
  `c214f7e83f542482041a43dfcac79601ff6aa8f258c5ccafec0259151366e811`.
  The device kernel binaries in that isolated package were reused from the
  pre-existing custom package, not recompiled during this session.
- [`primitives/probe_kernel.cpp`](primitives/probe_kernel.cpp) and
  [`primitives/probe_runner.cpp`](primitives/probe_runner.cpp) build against
  installed CANN 9.0.0. Each task-time point uses 10 warmup launches and
  eight measured AIC tasks under `msprof --task-time=on --ai-core=on`.
  AscendCL event brackets are saved separately and are not used for rates.
  Five randomized independent profiler runs of selected points are saved in
  `primitives/repeat_selected_corrected_raw.jsonl`.
- Mode 2 keeps GM-to-L1 and L1-to-L0 setup fixed, repeats FP16 128³ Cube
  accumulation in L0C, and checks the FP16 output. Mode 3 computes one
  resident Cube tile, varies the number and size of FP16 L0C-to-GM Fixpipe
  packets, and checks its output. Mode 4 varies GM ND-to-L1 NZ operand
  packets. The read probe explicitly waits for MTE2 completion; its output
  token checks that every AIC finished, though it does not numerically
  verify the copied L1 payload. Mode 1 is a flat-copy comparison, not the
  preferred BASE read estimate. The main ND→NZ sweep uses approximately
  1.25–10 MiB working sets, well below the device's 192 MiB L2 cache.
  A separate 160–320 MiB sweep crosses that capacity. Early read records
  with an undersized source allocation were discarded; their raw files are
  quarantined in [`primitives/discarded`](primitives/discarded/README.md).

The fit uses profiler kernel medians and the linear probe equation
`time_us = intercept_us + beta * bytes_per_core_MiB + packet_us * packets`
for read/write, and the analogous MFLOP equation for Cube. Aggregate
read/write rate is `20 * 2^20 / beta` bytes/µs at 20 AICs. The intercept
contains the probe's setup and launch work; it is **not** a separately
identified `SolveBase` launch latency. The raw profiler task times, not
only the fit, are in the JSONL files next to
[`primitives/analyze.py`](primitives/analyze.py).
Some early mode-2 raw rows labeled `bytes_per_core` with the nominal
`packets * packet_bytes` work axis even though Cube mode keeps GM traffic
fixed. The analyzer uses Cube repeat count and ignores that raw field for
mode 2; current collection code records zero for that field.

| 20-AIC exploratory coefficient | Estimate | Scope |
| --- | ---: | --- |
| FP16 Cube $R_C$ | 14.72 million FLOP/µs per AIC | Repeated 128³ L0C accumulation |
| GM ND→L1 NZ $R_R$ | 3.71 million bytes/µs aggregate | 4–32 KiB packets, L2-hot |
| Operand-read $\lambda_R$ | 0.110 µs per packet on one AIC | Serialized MTE2 packet loop |
| FP16 Fixpipe $R_W$ | 5.23 million bytes/µs aggregate | 16–128 rows × 128 columns, L2-hot |
| Output $\lambda_W$ | 0.163 µs per packet on one AIC | Serialized Fixpipe packet loop |
| Above-L2 ND→L1 NZ differential rate | 1.06 million bytes/µs aggregate | 240–320 MiB working set; five-run range 1.03–1.16 million |

The independently repeated, randomized 20-AIC measurements give these
paired-round slopes. Each range covers five runs, not a confidence interval;
the byte rates are effective aggregate rates for these particular packet
shapes and memory states. The Cube result is per AIC.

| Paired probe quantity | Five-run median | Five-run range |
| --- | ---: | ---: |
| Repeated 128³ FP16 Cube | 14.89 million FLOP/µs | 14.44–14.92 million |
| ND→NZ read, fixed 16 packets, varying bytes | 3.56 million bytes/µs | 3.43–4.15 million |
| ND→NZ read, fixed total bytes, varying packets | 0.111 µs/packet | 0.107–0.111 |
| FP16 Fixpipe, fixed 16 packets, varying bytes | 5.37 million bytes/µs | 4.88–6.51 million |
| FP16 Fixpipe, fixed total bytes, varying packets | 0.174 µs/packet | 0.168–0.178 |
| Minimal 20-AIC token kernel | 2.68 µs | 2.62–2.91 µs |

The one-AIC read and write rates do not scale into the 20-AIC results by a
constant factor. The minimal token kernel varied with AIC count and is not
identical to BASE dispatch, so no standalone `lambda_launch` or
occupancy-independent memory rate has been accepted. Above-L2 packet
latency could not be separated reliably: equal 240 MiB transfers with 48
versus 64 packets gave an unstable 0.11–0.66 µs/packet timing difference
across five runs. The repeated above-L2 differential throughput was 1.06
million bytes/µs, with a 1.03–1.16 million range.

## Source-matched tiling comparison

The isolated research override
[`enable_spatial_tiling_probe.py`](enable_spatial_tiling_probe.py) only
shrinks both `baseM` and `singleCoreM` after the normal tiler selects the
BASE path. It keeps the other emitted fields and uses the existing kernel
binary. The first attempt changed `baseM` alone, failed the full output
check, and was discarded. Every admitted row below passed a check of every
FP16 output element, retained tiling key 65536, and emitted
`nd2nzA=nd2nzB=0`. Task times are per-kernel medians from eight profiler
tasks; [all emitted fields and raw task times](spatial_results.json) are
preserved.

| Shape $(M,N,K)$ | `baseM=singleCoreM` | Useful AICs | Median kernel time |
| --- | ---: | ---: | ---: |
| (256,256,1024) | 32 (normal tiler) | 16 | 8.61 µs |
| (256,256,1024) | 16 | 20 | 10.38 µs |
| (512,512,1024) | 112 (normal tiler) | 20 | 10.37 µs |
| (512,512,1024) | 96 | 20 | 11.31 µs |
| (512,512,1024) | 64 | 20 | 11.52 µs |
| (512,512,1024) | 32 | 20 | 14.62 µs |
| (1024,1024,4096) | 128 (normal tiler) | 20 | 45.26 µs |
| (1024,1024,4096) | 64 | 20 | 59.01 µs |

The eight plans were then measured in five independent randomized rounds,
with full-output and tiling checks on every run. [Raw repeated cases](spatial_repeat_raw.jsonl)
and their [summary](spatial_repeat_analysis.json) preserve each result.
The normal tile won all five rounds for 256 and 1024. For 512 it won four
rounds; `baseM=96` won one by 0.038 µs. Median times across independent
runs were 8.79 versus 10.85 µs for the 256 pair, 10.57 versus 10.78 µs
for the closest 512 pair, and 45.16 versus 59.20 µs for the 1024 pair.
The paired 512 difference (`baseM=96` minus normal) ranged from -0.038 to
1.030 µs, so that ranking is not stable in every run.

Both the current global-packet equation and a trial per-core-wave equation
select the lowest five-run median tile among **these eight** candidates.
That is a narrow ranking check, not an acceptance test for arbitrary shapes
or tile choices.
With the measured coefficients, the current absolute predictions on the
original source-matched shape cases are approximately 35.0, 43.9, and
484.6 µs, versus 8.5, 10.0, and 45.1 µs observed. Replacing global packet
counts by per-core waves yields approximately 6.3, 7.5, and 58.8 µs,
which still misses in both directions. The full numbers, residuals, and
ranking choices are in [`primitives/analysis.json`](primitives/analysis.json).

## Consequence for `SolveBase`

The solver's `2 S ceil(K/baseK) lambda_R` and `N_o lambda_W` terms count
packet costs over all output tiles as if they were serial. The probe
latencies are measured **per AIC**, while source BASE runs those packets
across occupied AICs. Dividing by AIC count is not sufficient: output-tile
waves, tail imbalance, intra-tile overlap, and reuse affect the critical
path. The reduced model also treats `R_R` and `R_W` as fixed aggregate
device rates, whereas measured rates change with occupied cores and packet
shape. The model's `D(g)` counts requested operand traffic but may differ
from physical L2/HBM bytes when tiles are reused or cached.

Before enabling a measured profile, revise the BASE service term around
**per-core tile waves**, explicitly model the overlap of Cube, MTE2, and
Fixpipe for a wave, then test the new equation on held-out shapes and
controlled tilings. Keep L2-hot and HBM-working-set rates separate, and
repeat controlled cases in randomized order to establish ranking
uncertainty. The illustrative profile in the solver remains unchanged.
