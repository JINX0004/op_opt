# K > 65535: source Split-K paths on 910B3

The source-built MatMulV3 switches A from on-the-fly ND reads to an A ND→NZ
workspace conversion when FP16 NN `K` crosses 65535. On three matched
deterministic Split-K shapes, moving from the nearest tested aligned value
below the boundary (`K=65408`) to the first aligned value above it
(`K=65536`) increased the source kernel's measured time by 37–64%. The
Cube tiling fields and 20-AIC launch stayed fixed within each pair. This is
a measurement of the **threshold path as a whole**: the kernel key and A
format change along with conversion, so the delta is not a pure ND→NZ
primitive latency.

## Source implementation and feasibility

The [source tiling audit](source_audit.jsonl) was recorded before repeated
timing. Its six boundary cases all use `tilingEnableSplitCore=3`
(`DETERMINISTIC_SPLIT_K`), `baseM=baseN=baseK=128`, `stepKa=stepKb=3`,
`singleCoreK=384`, and 20 launched AICs. Below the boundary, `nd2nzA=0`
and the key is 65584 (`0x10030`); above it, `nd2nzA=1` and the key is 48
(`0x30`). B stays unconverted. Source
[`MatMulUnAlignedKernelDeterministicSplitK`](../../../../mat_mul_v3/op_kernel/mat_mul_unaligned_deterministic_splitk_kernel.h)
performs A conversion on AIVs, synchronizes, then lets the AICs compute
their K partitions and AIVs reduce FP32 partial results.

| M=N | Source singleCoreM/N | Workspace at K=65408 / 65536 | Added A workspace |
| ---: | ---: | ---: | ---: |
| 256 | 384×256 | 35 / 67 MiB | 32 MiB |
| 512 | 384×512 | 50 / 114 MiB | 64 MiB |
| 1024 | 384×1024 | 80 / 208 MiB | 128 MiB |

The workspace increase equals `2 M × 65536` bytes in each pair. The
source-valid `singleCoreM=384 > M=256` shows that a Split-K core tile may
exceed the output shape; the kernel handles that tail. The original
standalone domain rejected this geometry. The revised
[`TileValues`](../../mip_solver.cpp) admits this source 384 extent for
long-K Split-K, and [`FinalizePlan`](../../plan_finalizer.cpp)
accepts it while still rejecting oversized BASE tiles.

## Device method and boundary timing

Device 0 was a Huawei 910B3 with CANN 9.0.0 and the isolated source-built
operator package. The [sweep driver](../long_k_split_sweep.py) used source
defaults, ten warmups, eight `MIX_AIC` profiler tasks per run, and four
randomized paired rounds for each M=N. Each run rechecked the emitted
tiling against the prior audit, source path, kernel key, and every FP16
output element. The largest observed relative output error was 0.028%,
below the runner's 2% tolerance. The six [smoke runs](smoke_raw.jsonl) and their
[48 task times](smoke_tasks.jsonl) were excluded. The [24 repeated runs](repeat_raw.jsonl),
[192 task times](repeat_tasks.jsonl), and [paired analysis](analysis.json)
are retained.

| M=N | K=65408 / 65536 device median µs | Paired high/low ratio, 95% interval | Mean high minus K-scaled low µs |
| ---: | ---: | ---: | ---: |
| 256 | 61.612 / 101.578 | 1.6396 [1.5432, 1.7421] | 39.774 |
| 512 | 190.463 / 303.468 | 1.5965 [1.5605, 1.6334] | 113.285 |
| 1024 | 744.411 / 1021.994 | 1.3718 [1.3640, 1.3798] | 275.600 |

Device columns are medians of four run medians. Ratios are geometric means
of four within-round high/low ratios, with two-sided Student-t intervals
over paired log ratios. The last column subtracts the low-K time scaled
by `65536/65408`, assuming time would grow linearly over those extra 128
K elements in the absence of the threshold change. This adjustment is
diagnostic, not an independently measured conversion time. The intervals
describe these repeated runs only, not unseen shapes.

## Consequences for the standalone optimizer

The [original local solver audit](solver_audit.csv) used a declared 512 MiB
workspace budget, the existing source-step BASE packet profile, the
`{64,128,256}` M/N grid, and the current generic Split-K rates. The
original audit helper's 32 MiB budget would mark all above-threshold
cases infeasible even though the device runs allocate 67–208 MiB. With
512 MiB, that reduced model selected BASE on all six cases, while the source
selected deterministic Split-K. These are **model diagnostics**, not
device-tested optimizer tiles: the current BASE research override cannot
force a new family. The model's provisional common conversion term predicts
about 32.6, 64.8, and 129.3 µs at M=256, 512, and 1024, respectively;
the measured K-adjusted threshold excesses are 39.8, 113.3, and 275.6 µs.
The difference also includes the changed kernel path and any cache effects.

The [revised solver audit](revised_solver_audit.csv) repeats the six
shapes with source 384-extent candidates and busiest-AIC K-slice timing.
The local command is:

```sh
printf '256 256 65408\n256 256 65536\n512 512 65408\n512 512 65536\n1024 1024 65408\n1024 1024 65536\n' |
  mmopt_full_solver_audit stdin source_base packet_profile 512 source_split_grid
```

| M=N | K | Revised BASE / best Split-K (µs) | Source-pattern estimate / measured time (µs) | Reduced-model winner |
| ---: | ---: | ---: | ---: | --- |
| 256 | 65408 | 160.044 / 85.885 | 85.885 / 61.612 | Split-K |
| 256 | 65536 | 192.921 / 118.565 | 118.565 / 101.578 | Split-K |
| 512 | 65408 | 241.098 / 327.362 | 330.075 / 190.463 | BASE |
| 512 | 65536 | 306.317 / 392.492 | 395.205 / 303.468 | BASE |
| 1024 | 65408 | 597.206 / 1293.270 | 1298.700 / 744.411 | BASE |
| 1024 | 65536 | 727.466 / 1423.720 | 1429.150 / 1021.994 | BASE |

The source-pattern column evaluates the observed 384×N, base-128, step-3,
20-AIC tuple, independently of the solver's selected geometry. The revised
model changes the 256² family choice but still chooses BASE at 512² and
1024². Its large-shape Split-K overestimate is visible even for the exact
source pattern, so more geometry search alone cannot resolve the family
error. No revised choice has been forced through and measured on device.
The next steps are independent A ND→NZ measurement, source-backed
Split-K tiling override, and a richer model of source MK/NK spatial work
and memory reuse.

One further source case, `4096×1024×73728`, selected
`SINGLE_CORE_SPLIT_K` (mode 2, key 32) with A conversion rather than
deterministic Split-K. Its [one source-verified run](single_core_case.json)
measured 4966.628 µs and passed the full-output check. The current
standalone solver has no mode-2 family, so this case is a separate future
implementation branch and is not included in the paired table.
