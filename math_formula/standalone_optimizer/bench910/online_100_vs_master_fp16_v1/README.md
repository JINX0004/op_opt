# 100-shape FP16 online selector versus clean MatMulV3 master

This is a fresh, paired device comparison of the pushed L2-fixed online
selector (`codex/matmul-v3-mip-selector` at `de888261`) against clean
MatMulV3 master. The [dataset](dataset.json) consists of the previous
64 general shapes, eight mode-5 boundary shapes, and 28 new shapes.
The [generator](make_dataset.py) checks that all 100 are distinct and
have positive dimensions aligned to 16.

## Protocol

- Ascend 910B3 devices 4–7, 25 shapes per device, CANN 9.0.0.
  Each master/online pair runs on the same device. The harness checks
  device idleness around independent full-output and emitted-tiling
  audits and before and after every timed run.
- FP16 input/output, ND/ND/ND, NN, no bias. Master is the clean
  `ops_nn_master_fp16_package` with host-library SHA-256
  `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`.
  Online is `ops_nn_mip_l2fallback_package` with host-library SHA-256
  `74a2ce828e903469c7ce4e53e5951256423735bac6e3d8c0fcfdb94ee608e24b`.
- Online enables `MMV3_ONLINE_SELECTOR=1` and
  `MMV3_ONLINE_ROW_READ_PROFILE=1`; BASE tile injection is disabled.
  Timed arms use ten warmups and eight `msprof` kernel calls. Changed
  selections and mode-5 controls receive two reversed-order rounds;
  identical-tiling controls originally received one round. The two
  large-M cases below were later rerun with eight reversed-order rounds
  on their original devices and supersede those two one-round entries.
- The [runner](../online_expanded_vs_master_fp16_v1/run.py) uses the
  fixed dataset and separate package paths. The [analysis](analyze.py)
  validates completeness, output error, decisions, tiling, and rounds,
  then computes descriptive speedups and a shape-bootstrap interval.

Raw profiles and device idle snapshots are under
`/home/xianfei/solvebase_bench_20261025/cases/online_100_vs_master_fp16_v1`
on 910. The four [device summaries](device4.json) are preserved here;
`device4.json` retains the initial case-92 harness failure and
[`retry92b.json`](retry92b.json) is the successful replacement record.
The failure came from hard-coded B-conversion and profiler-kernel
assumptions in the measurement runner. Both master and online emitted
`nd2nzB=1` for 65536×448×1536; the corrected runner uses each arm's
audited conversion flags and profiles the conversion kernel.
The two later eight-round reruns and their idle snapshots are under
`/home/xianfei/solvebase_bench_20261025/cases/online_100_two_shape_rerun_v2`
on 910. Their raw summaries are [65536×256×1536](rerun_device7.json)
and [47104×384×1536](rerun_device6.json). The
[initial aggregate](analysis_before_rerun.json) and
[initial per-case CSV](comparison_before_rerun.csv) are preserved.

## Results

The [analysis](analysis.json) validates exactly 100 distinct, correct
results. Maximum relative output error was 0.000445. The
[per-case comparison](comparison.csv) includes both timing rounds where
the implementation family changed.

### By implementation-family decision

The speedup in this table is the geometric mean of each shape's
master/online kernel-time ratio. A value above 1 means online is faster.

| Family selection | Cases | Geometric mean speedup | Median shape speedup | Online faster / slower | Identical tiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Unchanged](family_unchanged.csv) | 90 | 0.9977× | 1.0002× | 46 / 44 | 90 |
| [Changed](family_changed.csv) | 10 | 1.2318× | 1.1727× | 10 / 0 | 0 |

All ten changed-family cases were faster online in **both** of their
reversed-order timing rounds; nine exceeded 5% speedup. The unchanged
family cases also have identical emitted tiling. Their near-parity
results include fresh-process timing variation, as the
[diagnosis](REGRESSION_DIAGNOSIS.md) demonstrates. The full shape-level
records for each category are in the linked CSV files.

| Shape M×N×K | Master → online mode | Master µs | Online µs | Speedup |
| --- | :---: | ---: | ---: | ---: |
| 256×512×16384 | 3 → 0 | 49.92 | 48.62 | 1.027× |
| 2048×768×27392 | 3 → 0 | 454.62 | 395.98 | 1.148× |
| 2304×768×32768 | 3 → 6 | 898.69 | 775.29 | 1.159× |
| 2560×768×32768 | 3 → 6 | 1090.39 | 979.25 | 1.113× |
| 2560×1024×32768 | 3 → 6 | 1334.02 | 870.18 | 1.533× |
| 2560×1024×49152 | 3 → 6 | 1421.27 | 1198.20 | 1.186× |
| 3072×1024×73728 | 2 → 3 | 4119.20 | 3699.57 | 1.113× |
| 768×2048×32768 | 3 → 0 | 636.58 | 482.71 | 1.319× |
| 1664×768×32768 | 3 → 0 | 502.00 | 313.39 | 1.602× |
| 2432×768×32768 | 3 → 6 | 986.92 | 798.26 | 1.236× |

### By dataset group

| Dataset group | Cases | Geometric mean master/online | Family changes |
| --- | ---: | ---: | ---: |
| Previous general | 64 | 1.0155× | 8 |
| Previous mode-5 boundary | 8 | 1.0017× | 0 |
| New narrow, long K | 7 | 1.0173× | 0 |
| New family transition | 7 | 1.1045× | 2 |
| New mode-5 boundary | 7 | 1.0070× | 0 |
| New large M/N | 7 | 1.0017× | 0 |
| **All** | **100** | **1.0189×** | **10** |

Most unchanged-family cases still have only one timing round, which
can vary by several percent despite identical kernel choices.

| Superseded shape | Initial master / online | Eight-round master µs | Eight-round online µs | Updated ratio | Online faster rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 65536×256×1536 | 0.916× | 242.151 | 236.518 | 1.024× | 7/8 |
| 47104×384×1536 | 0.955× | 243.081 | 240.629 | 1.010× | 4/8 |

The eight-round results replace only these two rows in the updated
100-case aggregate. The geometric mean increased from 1.0172× to
1.0189×. Both reruns passed full-output audits and matched their
original emitted tilings.

Six controls with large apparent differences were rerun with two
reversed-order rounds. Their raw records are
[`repeat_controls_device5.json`](repeat_controls_device5.json) and
[`repeat_controls_device6.json`](repeat_controls_device6.json).
All six again emitted identical master/online tilings.

| Shape | Initial ratio | Two-round diagnostic ratio | Observation |
| --- | ---: | ---: | --- |
| 64×16×65536 | 0.937× | 1.065× | Difference reversed |
| 512×128×73728 | 0.923× | 1.023× | Difference reversed |
| 2048×1536×32768 | 1.085× | 1.054× | Online repeat rounds varied widely |
| 2048×2048×27392 | 0.939× | 0.999× | Near parity on repeat |
| 65536×256×1536 | 0.916× | 0.955× | Online slower in both repeat rounds |
| 47104×384×1536 | 0.955× | 0.961× | Online slower in both repeat rounds |

The last two large-M BASE controls were slower in the two-round repeat despite
matching emitted tilings. A subsequent [path and measurement
investigation](REGRESSION_DIAGNOSIS.md) compared selector on/off within
the same online binary, master against itself, and master against the
online binary with its selector disabled. It found no reproducible
selector-caused slowdown; fresh-process timing variation is large
enough to account for the observed gaps.

The 20,000-resample shape-bootstrap interval for the updated all-case geometric
mean is 1.0048–1.0362×. It describes sensitivity to the particular 100
shapes; these shapes were selected deliberately, so this is not a
population confidence interval for arbitrary MatMul workloads. Timings
are kernel runtimes and exclude host selector overhead. The six
two-round diagnostics remain separate; only the two eight-round reruns
replace their initial measurements in the updated aggregate.
