# Executed mode-5 tile comparison, FP16 49152×384×1536

The online estimator's mode-5 winner is `singleCoreM×N=128×128`; the
MatMulV3 source tiler emits `2464×384`. This experiment runs both in
the same mode-5 kernel on Ascend 910B3 device 5. The only host-tiling
difference is the single-core M/N geometry. Both use 20 AICs,
`singleCoreK=384`, `baseM×N×K=128×128×128`, and `stepM×N×K=1×3×3`.

## Reproduction and isolation

[`override_patch.py`](override_patch.py) adds a temporary environment
override to the experimental branch's `CheckSingleTilingOk`, restricted
to this FP16 shape. `MMV3_MODE5_TILE_EXPERIMENT=l` selects 2464×384;
`s` selects 128×128. The host-only library was built against CANN
9.0.0 and copied into a separate package at
`/home/xianfei/solvebase_bench_20261025/ops_nn_mode5_tile_experiment_package`.
The master and previous online packages were preserved. The experimental
package's host library SHA-256 is
`4d33b6b9977d87d2968398a7bd86182f27866720ddde5c2059288f5d34e9b55d`.

[`run.py`](run.py) checks device idleness, audits full-output correctness
and emitted tiling, then profiles eight MatMulV3 kernel calls per arm
after ten warmups. Two rounds reverse the arm order. Both audits passed
with zero reported maximum relative error. See the
[large](large_audit.json) and [small](small_audit.json) audit records.
Raw `msprof` directories are under
`/home/xianfei/solvebase_bench_20261025/cases/mode5_tile_experiment_v1`;
the [summary](summary.json) contains round medians and pipeline counters.
No process was running on device 5 in the before/after idle checks.

## Results

| Round | 2464×384 (µs) | 128×128 (µs) | Small / large |
| --- | ---: | ---: | ---: |
| 0, large then small | 248.016 | 672.194 | 2.710× |
| 1, small then large | 248.606 | 672.972 | 2.707× |
| Median of round medians | **248.311** | **672.583** | **2.709×** |

The measured difference is 424.27 µs, with paired round differences
424.18 and 424.37 µs. This is a two-round comparison of one shape,
not a general mode-5 tiling calibration.

| Median `msprof` counter | 2464×384 (µs) | 128×128 (µs) |
| --- | ---: | ---: |
| AIC MAC active | 198.994 | 205.540 |
| AIC MTE1 active | 175.665 | 172.355 |
| AIC MTE2 active | 123.741 | 501.137 |
| AIC Fixpipe active | 139.957 | 142.074 |
| AIC scalar active | 127.906 | 539.036 |
| AIC wall | 227.989 | 656.829 |

The estimator scores 128×128 at 785.42 µs and 2464×384 at 795.42 µs,
reversing their measured order. MAC and Fixpipe demands are similar;
the small tile has much higher MTE2 and scalar activity. It creates
1,152 spatial tiles versus 20 for the large geometry, or roughly 58
versus one tile per AIC. The current `ScSliceUs` accounting gives both
geometries similar read-packet demand and has no measured term for
repeated `IterateAll` setup. The counters establish the missing
geometry-dependent cost, though they do not isolate one instruction or
transfer as its sole cause. The small tile also has `singleCoreN=128`
while the NK33 block's configured N panel is `stepN×baseN=384`;
the resulting underfilled panel is a plausible contributor, not an
independently measured cause.

The [measured model and held-out validation](MODEL_VALIDATION.md) use
the Fixpipe and per-`IterateAll` observations from this run. The
separate test package retains the tile override for reproduction. The
experimental branch's host source was restored after the run; its
later online model changes are limited to the estimator and hardware
profile files.
