# Converted-A read model versus master: FP16 on 910B3

The local converted-A Split-K read-startup change was copied as a
seven-line patch to the experimental 910 selector branch
`codex/matmul-v3-mip-selector` at base commit `bc40c260`. Its rebuilt
host tiler was installed in a **new** package; the clean master and
earlier online packages were preserved. This experiment measures the
executed MatMulV3 kernel with the online family selector enabled and
`MMV3_ONLINE_ROW_READ_PROFILE=1`. `MMV3_ONLINE_TILE` is unset, so the
source tiler chooses the actual block and buffer tiles.

## Protocol and provenance

- Hardware: Ascend 910B3 devices 4–7, installed CANN 9.0.0; FP16
  ND/ND/ND, NN, no bias. Devices 0–3 had other work and were unused.
- The 20 shapes are fixed in [the runner](run.py). Each shape compared
  clean master and the revised online package on the **same device**.
  Cases with different families had two reversed-order rounds; equal-family
  cases had one round. Each timed run used 10 warmups and eight `msprof`
  kernel calls. The timing reported per case is the median of round medians.
- Both arms had independent source-tiling and full-output correctness
  audits. The maximum relative error across all 40 main audits was
  0.000445. All 17 equal-family cases emitted identical source tiles and
  keys. No case or device-occupancy check failed, and devices 4–7 were
  idle after the run.
- Master host library SHA-256:
  `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`.
  Revised online host library SHA-256:
  `7ef0e1688e95a2085f5c504b17158651c03aea7ba5dfb8b72d95823b6c73e68e`.
- The remote raw records live under
  `/home/xianfei/solvebase_bench_20261025/cases/online_grouped_vs_master_fp16_v1`.
  The four compact device summaries are [device 4](device4.json),
  [device 5](device5.json), [device 6](device6.json), and
  [device 7](device7.json). [The analyzer](analyze.py) produces the
  [merged CSV](comparison.csv) and [aggregate JSON](analysis.json).

## Results

Speedup is master kernel time divided by revised-online kernel time;
values above one favor online. Modes are 0=BASE, 3=deterministic Split-K,
5=single-core NK33, and 6=single-core GM-to-L1.

| M×N×K | Master → online mode | Master µs | Online µs | Speedup |
| --- | :---: | ---: | ---: | ---: |
| 16×16×102400 | 3 → 3 | 39.48 | 40.87 | 0.966× |
| 32×16×102400 | 3 → 3 | 50.55 | 46.32 | 1.091× |
| 64×16×102400 | 3 → 3 | 66.35 | 64.96 | 1.021× |
| 128×16×65536 | 3 → 3 | 48.98 | 50.45 | 0.971× |
| 256×16×32768 | 3 → 3 | 33.67 | 33.02 | 1.020× |
| 16×32×102400 | 3 → 3 | 44.17 | 40.18 | 1.099× |
| 32×32×102400 | 3 → 3 | 48.48 | 47.75 | 1.015× |
| 64×32×65536 | 3 → 3 | 44.65 | 45.39 | 0.984× |
| 64×64×65536 | 3 → 3 | 47.67 | 46.68 | 1.021× |
| 128×128×32768 | 3 → 3 | 30.35 | 28.53 | 1.064× |
| 1792×768×27392 | 3 → 3 | 383.52 | 386.06 | 0.993× |
| 2048×768×32768 | 3 → 0 | 749.74 | 486.12 | 1.542× |
| 2560×1024×36864 | 3 → 6 | 1028.69 | 839.25 | 1.226× |
| 1024×2560×36864 | 6 → 6 | 841.73 | 840.64 | 1.001× |
| 2048×2048×4096 | 0 → 0 | 140.13 | 139.60 | 1.004× |
| 4096×4096×8192 | 0 → 0 | 1010.96 | 996.08 | 1.015× |
| 4096×4096×32768 | 6 → 6 | 5640.71 | 5605.88 | 1.006× |
| 49152×384×1536 | 5 → 0 | 246.89 | 258.82 | 0.954× |
| 2560×1024×73728 | 3 → 3 | 2662.04 | 2652.89 | 1.003× |
| 8192×4096×16384 | 0 → 0 | 4150.54 | 4171.64 | 0.995× |

The online family differs from master on **three** shapes, with **two
wins and one loss**. Compared with the earlier narrow-profile experiment,
the only family decision that changed is `2560×1024×73728`: it now
selects deterministic Split-K. Its online and master audits both emitted
`singleCoreM×N=2560×384`, `baseM×N×K=128×128×128`, stepK=3, and
555,745,280 bytes of workspace, with zero relative output error.
An independent device-4 smoke pair measured 2652.34 µs master and
2689.65 µs online; the full-suite device-6 pair measured 2662.04 and
2652.89 µs. The small differences with identical tiles show no
demonstrated kernel speed change. The prior **23.3% BASE family
regression** is removed because the revised selector chooses the same
source family as master.

Across this deliberately selected 20-shape set, geometric mean
`T_master/T_online` is **1.0433×**. The sums of one case median per shape
are 17.309 ms master and 16.821 ms online. These are descriptive suite
figures, not a population-wide deployment estimate. Equal-tile cases
had one timing round, so small differences there should not be ascribed
to the selector. The remaining changed-family loss is
`49152×384×1536` (online **4.8% slower**); it is unrelated to the
converted-A correction.

For that remaining regression, the online estimator's candidate is
**BASE**, 20 AICs, `singleCoreM×N×K=128×128×1536`,
`baseM×N×K=128×128×128`, with a **416.10-µs** objective. It scores
deterministic Split-K at 516.65 µs (12 AICs, 128×128, stepK=1) and
single-core mode 5 at 785.42 µs (20 AICs, 128×128×384, stepN=stepK=3);
modes 2 and 6 are infeasible. The source tiler actually runs BASE
`singleCoreM×N=128×256`, `baseM×N×K=128×256×64`, stepKa/Kb=8/4
for the online arm. Master runs mode 5 with
`singleCoreM×N×K=2464×384×384`, `baseM×N×K=128×128×128`.
The candidate-tile mismatch and mode-5 cost overestimate remain; the
converted-A read change does not address this K=1536 case.

The standalone CTest suite passed 6/6. The 910 selector CLI rebuilt and
selected mode 3 on the target shape before device testing. The experimental
branch has one uncommitted model-file change; no master source or installed
CANN package was modified.
