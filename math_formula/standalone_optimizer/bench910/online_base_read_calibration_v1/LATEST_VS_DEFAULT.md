# Latest online selector versus default MatMulV3 (FP16, 910B3)

This is a direct source-backed comparison on device 4. The default arm uses
the clean MatMulV3 selection logic from `master` at `b2cdb471`; the online
arm uses `codex/matmul-v3-mip-selector` at `bc40c260` with
`MMV3_ONLINE_SELECTOR=1` and `MMV3_ONLINE_ROW_READ_PROFILE=1`.
`MMV3_ONLINE_TILE` is unset in both arms, so the source host tiler emits
the block and buffer tiles for the chosen family. Both packages were built
against installed CANN 9.0.0. The master package has only the test-only
legacy-cache bypass required by this isolated `aclop` harness; its family
selection is unchanged.

The workload is FP16 ND/ND/ND, NN, no bias. Each arm first had a source
tiling and correctness audit. The device was checked idle before and after
audits and timed runs. Timing uses the `msprof` MatMulV3 kernel task, with
10 warmups and 8 measured calls per round. Shapes where the mode changed
had two rounds with reversed arm order; equal-mode controls had one round.
The table reports the median of available round medians. All 40 audits
passed; maximum relative output error was 0.000445. No device-4
interference flags were raised. The complete [per-case summary](online_narrow_vs_default_fp16_summary.json)
records emitted tilings and round medians; [the runner](run_latest_vs_default.py)
shows the exact environment and procedure. Raw profiles and CANN logs are
on 910 at
`/home/xianfei/solvebase_bench_20261025/cases/online_narrow_vs_default_fp16_v1`.

Speedup is **default time / online time**; values above 1 favor online.
Modes: 0 = BASE, 3 = deterministic Split-K, 5 = single-core NK33 Split-K,
6 = single-core GM-to-L1 Split-K.

| M×N×K | Default → online mode | Default µs | Online µs | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 16×16×102400 | 3 → 3 | 42.18 | 44.03 | 0.958× |
| 32×16×102400 | 3 → 3 | 48.50 | 49.19 | 0.986× |
| 64×16×102400 | 3 → 3 | 63.39 | 60.92 | 1.041× |
| 128×16×65536 | 3 → 3 | 49.58 | 48.42 | 1.024× |
| 256×16×32768 | 3 → 3 | 34.05 | 31.59 | 1.078× |
| 16×32×102400 | 3 → 3 | 41.90 | 39.94 | 1.049× |
| 32×32×102400 | 3 → 3 | 50.60 | 48.98 | 1.033× |
| 64×32×65536 | 3 → 3 | 44.72 | 44.53 | 1.004× |
| 64×64×65536 | 3 → 3 | 46.23 | 45.96 | 1.006× |
| 128×128×32768 | 3 → 3 | 30.34 | 30.71 | 0.988× |
| 1792×768×27392 | 3 → 3 | 386.19 | 386.82 | 0.998× |
| 2048×768×32768 | 3 → 0 | 741.38 | 508.93 | 1.457× |
| 2560×1024×36864 | 3 → 6 | 1020.13 | 843.51 | 1.209× |
| 1024×2560×36864 | 6 → 6 | 842.83 | 839.84 | 1.004× |
| 2048×2048×4096 | 0 → 0 | 139.73 | 140.49 | 0.995× |
| 4096×4096×8192 | 0 → 0 | 995.43 | 994.36 | 1.001× |
| 4096×4096×32768 | 6 → 6 | 5634.71 | 5640.84 | 0.999× |
| 49152×384×1536 | 5 → 0 | 247.04 | 262.43 | 0.941× |
| 2560×1024×73728 | 3 → 0 | 2656.76 | 3276.55 | 0.811× |
| 8192×4096×16384 | 0 → 0 | 4208.16 | 4184.97 | 1.006× |

The selector changed the family on **4 of 20** shapes: two wins and two
losses. On all eight narrow-N cases, it chose the same deterministic
Split-K family as default. All 16 equal-mode cases also matched the
source-emitted `singleCoreM/N/K`, `baseM/N/K`, steps, and tiling key; their
small timing differences do not show a different tiling benefit. Thus the earlier 2.11–21.96× gains versus the
*generic online model* must not be presented as gains versus default
MatMulV3. The narrow B-row correction mainly repairs a bad online choice.

Across this selected 20-shape set, the geometric mean of per-shape
speedups is 1.023×, while the sum of one kernel time per shape is 17.3238
ms default versus 17.5230 ms online (online 1.15% slower by that weighting).
Neither is an expected deployment speedup: the set is deliberately chosen
for coverage, and same-mode cases have only one round. In particular, the
23.3% long-K regression and 6.2% mode-5 regression remain. The online
objective still scores its optimized hypothetical BASE tile rather than
the source-emitted tile for every family; that mismatch is the next model
issue to resolve before treating the selector as a general replacement.

Package host-library SHA-256: default
`b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`;
online
`092693da5d9818375738c688e167e87b51f0f08cf84e4a9ad2233ecc89495cd5`.
