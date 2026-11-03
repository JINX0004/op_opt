# Revised online selector versus clean MatMulV3 master: 22 FP16 shapes

This is a fresh device comparison with **clean master as the baseline**.
It reruns all 20 shapes from the previous FP16 suite and adds the two
held-out mode-5 shapes, `65536×384×1536` and `81920×384×1536`.
The [runner](run.py), four device summaries
([4](device4.json), [5](device5.json), [6](device6.json),
[7](device7.json)), [merged CSV](comparison.csv), and
[aggregate analysis](analysis.json) preserve the decisions and times.
Raw profiles are under
`/home/xianfei/solvebase_bench_20261025/cases/online_mode5_vs_master_fp16_v1`.

## Protocol

- Master package: `ops_nn_master_fp16_package`, host library SHA-256
  `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`.
- Revised package: `ops_nn_mip_mode5_package`, host library SHA-256
  `bd4a1faa7670d7400f8f67cf4c033013004771a2bec1f471205363cea134d368`.
- Ascend 910B3 devices 4–7, CANN 9.0.0, FP16 ND/ND/ND, NN, no bias.
  Each master/revised pair ran on the **same device**. Devices were idle
  at each before/after check.
- Both arms received independent full-output and emitted-tiling audits.
  Maximum relative error across the 44 audits was 0.000445. All 20
  same-family cases emitted identical tile fields and keys.
- Timed runs used ten warmups and eight `msprof` kernel calls. Changed-
  family cases and all three mode-5 shapes had two reversed-order rounds;
  other equal-family cases had one round. Each table entry is the median
  of its round medians.

Modes: 0=BASE, 3=deterministic Split-K, 5=single-core NK33, and
6=single-core GM-to-L1. Speedup is master time divided by revised time.

| # | M×N×K | Master→revised mode | Master µs | Revised µs | Speedup |
| ---: | --- | :---: | ---: | ---: | ---: |
| 0 | 16×16×102400 | 3→3 | 41.33 | 42.39 | 0.975× |
| 1 | 32×16×102400 | 3→3 | 48.90 | 49.12 | 0.995× |
| 2 | 64×16×102400 | 3→3 | 63.22 | 60.90 | 1.038× |
| 3 | 128×16×65536 | 3→3 | 50.36 | 50.60 | 0.995× |
| 4 | 256×16×32768 | 3→3 | 33.24 | 32.95 | 1.009× |
| 5 | 16×32×102400 | 3→3 | 41.10 | 43.09 | 0.954× |
| 6 | 32×32×102400 | 3→3 | 48.76 | 49.53 | 0.984× |
| 7 | 64×32×65536 | 3→3 | 45.33 | 44.51 | 1.018× |
| 8 | 64×64×65536 | 3→3 | 45.45 | 45.07 | 1.008× |
| 9 | 128×128×32768 | 3→3 | 30.46 | 30.51 | 0.998× |
| 10 | 1792×768×27392 | 3→3 | 387.03 | 385.60 | 1.004× |
| 11 | 2048×768×32768 | 3→0 | 734.62 | 520.57 | 1.411× |
| 12 | 2560×1024×36864 | 3→6 | 1026.53 | 840.27 | 1.222× |
| 13 | 1024×2560×36864 | 6→6 | 845.69 | 843.45 | 1.003× |
| 14 | 2048×2048×4096 | 0→0 | 139.52 | 139.99 | 0.997× |
| 15 | 4096×4096×8192 | 0→0 | 995.22 | 995.64 | 1.000× |
| 16 | 4096×4096×32768 | 6→6 | 5614.83 | 5622.64 | 0.999× |
| 17 | 49152×384×1536 | 5→5 | 247.94 | 247.37 | 1.002× |
| 18 | 2560×1024×73728 | 3→3 | 2670.12 | 2736.92 | 0.976× |
| 19 | 8192×4096×16384 | 0→0 | 4183.75 | 4155.39 | 1.007× |
| 20 | 65536×384×1536 | 5→5 | 315.43 | 316.05 | 0.998× |
| 21 | 81920×384×1536 | 5→5 | 398.71 | 396.84 | 1.005× |

## Interpretation

The revised selector differs from master on **two** shapes. It wins
both changed-family comparisons in each of the two paired rounds:
`2048×768×32768` selects BASE rather than deterministic Split-K
(1.411× median speedup), and `2560×1024×36864` selects mode 6 rather
than deterministic Split-K (1.222×). On the three mode-5 shapes, the
revised estimator now selects mode 5 and the source-sized geometry;
master emits the same tiling, so their runtimes agree within run drift.

The descriptive geometric mean speedup over these 22 deliberately
chosen shapes is **1.0233×**. A shape-bootstrap 95% interval is
**0.9950–1.0642×**, which includes a tie; this is not evidence of a
population-wide speedup. Equal-tile cases generally had one round,
and their small differences should not be attributed to the selector.
For example, `2560×1024×73728` has identical master/revised tiling
but a 2.4% timing difference in its single round.

The earlier [mode-5 model validation](../mode5_tile_experiment_v1/MODEL_VALIDATION.md)
compared the revised package with the **previous online package** to
diagnose the decision change. This fresh 22-case run is the performance
comparison against master.
