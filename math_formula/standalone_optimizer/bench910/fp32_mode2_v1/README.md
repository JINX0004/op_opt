# FP32 single-core Split-K mode 2 search on 910B3

This follow-up adds FP32 mode 2 to the experimental standalone family
selector on the 910 branch `codex/matmul-v3-mip-selector` at commit
`2f256c00`, pushed to `origin`. The FP32 product
online gate remains closed. The source snapshot in [remote_selector](remote_selector/)
contains the changed C++ model, preflight, tests, and documentation; the
working implementation is in
`/home/xianfei/dev/ops-nn-sribd-mip-selector/experimental/mip_selector` on 910.

## Model and feasibility

- Same-type FP32, NN, ND/ND/ND, no bias, HF32 disabled. Converted-A
  `K>65535` remains excluded.
- The mode-2 source tile uses `baseM=baseN=128`, `baseK=64`, and
  `singleCoreK=baseK×stepK`. It searches the source's 2×4 and 3×3 block
  patterns, output partitions, and occupied AICs. M core extents align to
  eight elements; N extents align to 64. The source's long-K, L1/L0,
  workspace, and spatial-load gates are checked before scoring.
- FP32 mode 2 calls `IterateAll` for each K packet, with atomic output
  updates after the first. Its AIC writes FP32 output directly; the BF16
  vector cast/output stage is absent. The objective uses the existing
  measured FP32 atomic-output service as a provisional proxy.
- FP32 modes 5/6 remain excluded from this NN/ND experiment. FP16/BF16
  behavior is unchanged.

## Validation

The 910 build passed `mmopt_dtype_tests` and the FP16 golden test. All
eleven master mode-2 source tuples in the earlier [100-case experiment](../fp32_100_source_family_v1/README.md)
pass the new preflight. Their source-tile objective has a **13.0% mean
absolute relative error**, consistently underestimating measured time;
see [mode2_scores.csv](mode2_scores.csv). Dedicated FP32 mode-2 output
calibration remains necessary before treating the score as accurate.

The revised [100 decisions](mode2_decisions.csv) choose mode 2 six times.
All six shapes also used mode 2 in clean master. Their forced-family runs
on devices 4–7 emitted identical tiling fields to master, passed FP32
correctness, and differed by less than 1% in kernel time. The five other
master mode-2 shapes selected a previously measured faster alternative;
one improvement was only 1.2% and is within observed timing variation.

| Shape | New family | Master µs | Proposed µs | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 2816×1024×36864 | 0 | 3929.93 | 3306.14 | 1.189× |
| 3072×1024×36864 | 0 | 4090.13 | 3301.07 | 1.239× |
| 1024×2304×36864 | 2 | 2970.88 | 2976.66 | 0.998× |
| 1024×2816×36864 | 2 | 3471.49 | 3473.29 | 0.999× |
| 2048×1536×32768 | 2 | 3274.14 | 3283.34 | 0.997× |
| 2048×2048×27392 | 0 | 3508.02 | 3465.26 | 1.012× |
| 2048×3072×32768 | 2 | 6117.36 | 6121.94 | 0.999× |
| 3072×2048×32768 | 2 | 6998.58 | 7054.64 | 0.992× |
| 8192×4096×32768 | 2 | 33839.26 | 33820.22 | 1.001× |
| 2816×1024×49152 | 3 | 5716.10 | 4476.07 | 1.277× |
| 1024×2432×32768 | 0 | 2661.01 | 2424.43 | 1.098× |

The combined [100-case analysis](analysis.json) has a **0.981×** geometric
mean speedup against master, versus **0.979×** before mode 2. Family changes
fall from 32 to 26. The 21 deterministic Split-K → BASE changes still
dominate the regressions, so the FP32 product gate stays closed.

The six paired [validation records](validation_records.json) contain tiling,
correctness, and `msprof` timings. The original 94 pairs are reused only
when the new family matches the earlier proposed family. [analyze.py](analyze.py)
verifies this join and rejects any unmeasured family change. Full raw
profiles are under `/home/xianfei/solvebase_bench_20261025/cases/fp32_mode2_v1`
on 910. The experiment forces the **family** and lets MatMulV3 emit the
tile; it does not benchmark the model's proposed mode-2 tile.
