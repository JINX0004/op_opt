# Mode-5 tile diagnosis: FP16 49152×384×1536 on 910B3

This check uses the existing device-5 source run in
[`online_grouped_vs_master_v1`](online_grouped_vs_master_v1/README.md),
the [paired tile experiment](mode5_tile_experiment_v1/README.md),
the experimental online model on the 910 machine, and the source kernel
in `mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel.h`.

## Candidate coverage and model ranking

The source tiler emits mode 5, 20 AICs, `singleCoreM×N×K=2464×384×384`,
`baseM×N×K=128×128×128`, and `stepM×N×K=1×3×3`. The default
`TileValues` domain includes `2464`: it aligns `ceil(49152/20)` to 16;
it also includes the full N extent, `384`. `FinalizePlan` accepts this
tile with 20 output tiles and 96,468,992 bytes of workspace.

| Mode-5 tile, 20 AICs | Reduced objective (µs) |
| --- | ---: |
| Model-selected `128×128×384` | 785.42 |
| Source-emitted `2464×384×384` | 795.42 |

The source tile is present and feasible. The model ranks the small tile
10.00 µs (1.3%) faster, but the paired execution measures 248.31 µs
for the large tile and 672.58 µs for the small tile: the model's
ranking is reversed. The current
online host path applies a model-selected `baseK` only to BASE when
`MMV3_ONLINE_TILE=1`; mode-5 geometry remains the source tiler's choice.

## Why the source-tile objective is too high

The model's `ScSliceUs` adds GM input, FP32 Fixpipe store, and atomic
increment inside one serial memory term for each K packet. It applies
the `sc_fp32_store` rate of 34,190 bytes/µs, measured by a serial
distinct-address output probe, to all four mode-5 packets. It also
charges two generic `gm_read` startups per 128×128×128 base packet,
even though the source uses `stepKa=stepKb=3` and `MM_CFG_PRELOAD_NK`.
`ScSliceUs` then sums four such slice times. This cannot represent the
observed overlap of the source's MAC, MTE, and Fixpipe pipelines.

The paired `msprof` operator summaries are stable across the two
reversed-order rounds. Median active times are:

| Counter | Large 2464×384 (µs) | Small 128×128 (µs) |
| --- | ---: | ---: |
| AIC MAC | 198.99 | 205.54 |
| AIC MTE1 | 175.67 | 172.36 |
| AIC MTE2 | 123.74 | 501.14 |
| AIC Fixpipe | 139.96 | 142.07 |
| AIC scalar | 127.91 | 539.04 |
| AIC wall time | 227.99 | 656.83 |
| Whole kernel | 248.31 | 672.58 |

The small tile's MAC and Fixpipe activity barely change, while MTE2
activity grows by 377 µs and scalar activity by 411 µs. These active
times overlap and cannot be added to obtain wall time. For comparison,
the source tile writes about 15.1 MB of FP32
partials across four K packets; 140 µs of reported Fixpipe active time
corresponds to roughly 108 kB/µs of *in-context* demand. This is
different from the serial probe's 34.2 kB/µs. The rate is inferred from
one shape and should not be installed as a general mode-5 coefficient.

Sensitivity scores below recompute the same objective at both tiles;
they are **model-only counterfactuals**, not device results. The
`fast_output` case substitutes the separately measured mode-6
in-context 91 kB/µs total output rate with no extra atomic term.
`grouped_startup` divides the generic read startup by three to show
the possible effect of the 3×K step. Neither substitution has been
validated for mode 5.

| Model scenario | 128×128 (µs) | 2464×384 (µs) |
| --- | ---: | ---: |
| Current | 785.42 | 795.42 |
| Fast output only | 335.01 | 344.40 |
| Grouped startup only | 683.69 | 690.18 |
| Both | 233.28 | 239.16 |

The output coefficient is the largest isolated source of the *large
tile's absolute-time* overprediction. Even the optimistic `both`
scenario still ranks the small tile first, so changing output rate
alone does not repair tiling selection. The reduced objective has no
measured cost for roughly 58 separate
spatial-tile handoffs on each small-tile AIC, compared with one
source-sized output tile per AIC. Across four K packets this means
roughly 232 versus four `IterateAll` calls per AIC. Source code confirms
that mode 5 calls `SetSingleShape`, `SetTensorA/B`, and `IterateAll` for
each assigned tile and K packet. The paired profile confirms large
MTE2 and scalar demand on the small-tile path; the exact sub-operation
responsible has not been isolated.

## Subsequent model and validation

The [measured FP16 mode-5 objective](mode5_tile_experiment_v1/MODEL_VALIDATION.md)
now charges one per-`IterateAll` setup and treats MAC, MTE2, and
Fixpipe as concurrent service demands. Its primitive parameters came
from the paired run above. Two other eligible M sizes were measured
after the coefficients were frozen; both chose the faster source-sized
mode-5 tile. A subsequent [fresh 22-shape comparison with clean
master](online_mode5_vs_master_v1/README.md) found that revised online
and master both select the same mode-5 tile on all three shapes, with
nearly equal measured runtimes. Absolute time error and the
BASE-versus-mode-5 objective margin still need broader validation.

The saved source profile is under
`/home/xianfei/solvebase_bench_20261025/cases/online_grouped_vs_master_fp16_v1/device5/timing/r{0,1}/49152x384x1536/default/profile`.
