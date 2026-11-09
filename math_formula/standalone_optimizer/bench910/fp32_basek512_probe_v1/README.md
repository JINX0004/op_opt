# FP32 16×16×65408: execute the estimator's BASE K tile

The FP32 selector proposes BASE with `singleCoreM=singleCoreN=16`,
`singleCoreK=65408`, `baseM=baseN=16`, and `baseK=512`, scoring it at
**33.16204 µs**. Earlier family comparisons executed source-generated
BASE with `baseK=32`; they did not execute this proposed tile. This
experiment tests the proposed tile directly on a 910B3.

## Method

- The [patch](patch_source.py) changes only the host tiler in an isolated
  copy of the source-matched research package. It applies only to same-type
  FP32, HF32 disabled, shape 16×16×65408, forced BASE, with
  `MMV3_FP32_BASEK512_PROBE=1`. It sets `baseK=512` and recomputes the L1
  K steps. The original FP32 device binary is used for every arm; its
  SHA-256 is `41bb84c6f7363b494ed632c786e96722f2eb4f518a02342f141e764021cf7024`.
- The isolated host library SHA-256 is
  `7fd7bc68fe986d6f5099df92443038e34d5c1aa5168295f8461b9de37d16f1ed`.
  The source BASE host library is
  `cd074481e4819bffc4e8d85d1d42e9c889d670dc3ecb213264cecb83ebbe5ad3`.
- The [runner](run.py) uses the existing `fp32_bench` and `msprof` protocol
  on idle device 4 with CANN 9.0.0. Each arm passes the output check
  (`max_relative_error=0.0`) and emitted-tiling audit. Each round takes
  the median of five profiled kernel tasks after five warmups. The two
  rounds reverse arm order. Times are device kernel times; they exclude
  host scoring.

| Arm | Mode | `baseM×baseN×baseK` | `stepKa/stepKb` | Round 0 µs | Round 1 µs | Median µs |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Clean master | deterministic Split-K | 128×128×64 | 3/3 | 40.976 | 38.860 | **39.918** |
| Source BASE | BASE | 16×16×32 | 64/64 | 142.780 | 142.480 | **142.630** |
| Proposed BASE | BASE | 16×16×512 | 4/4 | 43.760 | 44.360 | **44.060** |

The proposed tile is **3.24× faster** than source BASE, but **10.4% slower**
than clean-master deterministic Split-K. The 33.162 µs estimate
**underestimates measured time by 24.7%**, using measured time as the
denominator. Both paired rounds rank deterministic Split-K
ahead of proposed BASE. Two rounds establish this observed ordering but
are not a broad statistical performance claim.

The source and proposed BASE plans both have a 2048-element L1 K window
(`baseK×stepK`), while the proposed plan uses far fewer L0 K packets.
This likely removes much of the small-`baseK` overhead, although the
experiment does not isolate the exact device stage. The remaining
33.16-to-44.06 µs gap shows that scoring the proposed tile itself is
still optimistic. The [audited tiling and timings](summary.json) are
saved here; raw `msprof` records are under
`/home/xianfei/solvebase_bench_20261025/cases/fp32_basek512_probe_v1`
on 910.
