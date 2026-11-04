# 100-shape BF16 MatMulV3 online selector versus clean master

**Follow-up:** The [BF16 narrow-B calibration](../bf16_narrow_b_calibration_v1/README.md)
identified and corrected the `16×16×65408` family decision in an isolated
source-backed test. The results below are the historical generic BF16
selector baseline; the corrected branch has not yet had a fresh 100-shape
device timing run.

This experiment runs the calibrated BF16 online selector on the same
[100 distinct shapes](dataset.json) used for the earlier
[FP16 comparison](../online_100_vs_master_fp16_v1/README.md). The workload is
same-type BF16 A, B, and output; NN; ND/ND/ND; no bias. These shapes were
chosen to exercise implementation boundaries. They are not a random or
representative workload distribution.

## Protocol

- Ascend 910B3 devices 4, 6, and 7, with each shape's master/online pair
  on the same device. Index modulo three assigns shapes to devices in that
  order. Device 5 was occupied and was not used.
- CANN 9.0.0. The clean master package has host-library SHA-256
  `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`.
  The calibrated online package has host-library SHA-256
  `3e0d3a91ad4efcf50ce5af3d05ccbd452c45b8d105fc58f9edf2883d7e053471`.
  Both include the same benchmark-only BF16 dtype-inference proto patch
  described in the [initial BF16 report](../bf16_online_vs_master_20261103/README.md).
- Online enables `MMV3_ONLINE_SELECTOR=1` and
  `MMV3_ONLINE_BF16_EXPERIMENT=1`; `MMV3_ONLINE_TILE` is unset. Each package
  uses the source tiler for its selected family's actual block and buffer
  tiling.
- Every arm gets an independent full-output and emitted-tiling audit,
  five warmups, and five profiled MatMulV3 tasks. An emitted family or tile
  difference receives two rounds with reversed arm order; mode-5 controls
  also receive two rounds. Device idleness is checked before and after
  each audit and timing run.
- The [runner](run.py) records every shape independently under
  `/home/xianfei/solvebase_bench_20261025/cases/bf16_100_vs_master_v1`
  on 910. The [portable records](records.json) preserve the 100 audit and
  timing summaries; the complete profiler traces remain at the path above.
  The [analysis](analyze.py) verifies coverage, correctness, decisions, and
  required repeats before computing descriptive speedups. Reproduce the
  tables locally with `python3 analyze.py --records records.json`.

## Results

The [analysis](analysis.json) validates all 100 distinct shapes with no
failed cases. Maximum relative output error was 0.003115, below the
benchmark's 0.02 limit. The [per-shape CSV](comparison.csv) gives emitted
family, core tile, timing, and speedup. Speedup is master time divided by
online time; a value greater than one favors online. The times below are
MatMulV3 kernel times and exclude host selector overhead.

| Family selection | Cases | Geometric mean speedup | Median shape speedup | Online faster / slower | Emitted tiling identical |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Changed](family_changed.csv) | 11 | 0.9112× | 1.1716× | 10 / 1 | 0 |
| [Unchanged](family_unchanged.csv) | 89 | 0.9997× | 1.0004× | 46 / 42, one tie | 89 |
| **All** | **100** | **0.9895×** | **1.0007×** | **56 / 43, one tie** | **89** |

Ten changed-family choices were faster online in **both** reversed-order
rounds. The remaining choice, `16×16×65408`, was about **20.1× slower**:
master chose deterministic Split-K and measured 29.55 µs; online chose BASE
and measured 594.08 µs. Both rounds reproduced the regression (master/online
ratios 0.049 and 0.051). The online BASE core tile spans the full
`16×16×65408` workload, whereas master uses a `384×16×384` deterministic
Split-K tile. The FP16 run of this same shape selected deterministic Split-K
in both packages. This is evidence that the BF16 BASE estimate/decision near
the `K=65535` boundary needs correction; the measurements alone do not
isolate which primitive cost is missing.

| Shape M×N×K | Master → online mode | Master µs | Online µs | Speedup |
| --- | :---: | ---: | ---: | ---: |
| 16×16×65408 | 3 → 0 | 29.55 | 594.08 | **0.050×** |
| 256×512×16384 | 3 → 0 | 50.61 | 49.75 | 1.017× |
| 2048×768×27392 | 3 → 0 | 452.04 | 389.74 | 1.160× |
| 2304×768×32768 | 3 → 6 | 902.67 | 771.51 | 1.170× |
| 2560×768×32768 | 3 → 6 | 1070.55 | 879.76 | 1.217× |
| 2560×1024×32768 | 3 → 6 | 1314.13 | 869.08 | 1.512× |
| 2560×1024×49152 | 3 → 6 | 1419.32 | 1196.62 | 1.186× |
| 3072×1024×73728 | 2 → 3 | 4122.26 | 3710.40 | 1.111× |
| 768×2048×32768 | 3 → 0 | 632.62 | 536.22 | 1.180× |
| 1664×768×32768 | 3 → 0 | 500.80 | 320.65 | 1.562× |
| 2432×768×32768 | 3 → 6 | 989.68 | 844.76 | 1.172× |

Modes 0, 2, 3, and 6 mean BASE, single-core Split-K mode 2,
deterministic Split-K, and single-core Split-K mode 6, respectively.
Removing the severe `16×16×65408` case changes the 99-shape geometric mean
to 1.0199×; this is an outlier sensitivity calculation, **not** a corrected
headline result. The 20,000-resample shape-bootstrap interval for the full
100-shape geometric mean is 0.9227–1.0320×. Because the shapes deliberately
cover boundaries, it is not a population confidence interval for arbitrary
MatMul workloads.

| Dataset group | Cases | Geometric mean speedup | Family changes |
| --- | ---: | ---: | ---: |
| Previous general | 64 | 0.9746× | 9 |
| Previous mode-5 boundary | 8 | 1.0010× | 0 |
| New narrow, long K | 7 | 1.0016× | 0 |
| New family transition | 7 | 1.0878× | 2 |
| New mode-5 boundary | 7 | 0.9984× | 0 |
| New large M/N | 7 | 0.9999× | 0 |

The 15 mode-5 boundary controls all selected the same family and tile in
both packages, including mode 5 within its source feasibility window and
BASE on either side. The 89 unchanged-family controls also emitted identical
tiling; their small timing differences cannot be attributed to a different
online tiling decision. Most of these controls have one timing round. The
two-round mode-5 control at `83968×384×1536` measured 439.78 versus
417.94 µs despite identical tiling, illustrating residual timing/package
variation.

**Conclusion:** the calibrated BF16 selector finds ten useful family changes
on this dataset, but the severe narrow, just-below-65535 K regression means
this validation does not support enabling BF16 online selection by default.
The next model check is the BF16 BASE cost for narrow M/N and long K, followed
by a fresh paired rerun of the affected boundary and full suite after a fix.
