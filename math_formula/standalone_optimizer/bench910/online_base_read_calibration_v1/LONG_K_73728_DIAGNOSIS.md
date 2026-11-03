# Why the online selector chooses BASE for 2560×1024×73728 FP16

The committed online selector at branch `bc40c260` chose BASE for this shape, but
default MatMulV3 chooses deterministic Split-K. The paired device-4 result
was **3276.55 µs BASE versus 2656.76 µs Split-K** (two reversed-order
rounds, eight profiled kernel calls per arm per round, correctness passed).
See [the direct comparison](LATEST_VS_DEFAULT.md) and its
[raw summary](online_narrow_vs_default_fp16_summary.json).

## What the selector compares

The [candidate diagnostic](diagnose_long_k.cpp), compiled against the exact
remote selector branch, reports these reduced objectives:

| Family | Optimized model tile (`singleCoreM×N`, `baseM×N×K`) | Model µs |
| --- | --- | ---: |
| BASE | 128×128, 128×128×128 | 3006.17 |
| Deterministic Split-K | 128×128, 128×128×128, stepK=1 | 3035.62 |

The margin is only **29.45 µs**, so a small relative modeling error changes
the choice. The host uses only the model's family decision. With
`MMV3_ONLINE_TILE` unset, the source emits different tiles:

| Family | Source `singleCoreM×N`, `baseM×N×K` | Other emitted fields | Measured µs |
| --- | --- | --- | ---: |
| BASE | 128×256, 128×256×64 | 80 output tiles, 20 useful AICs; A ND→NZ | 3276.55 |
| Deterministic Split-K | 2560×384, 128×128×128 | stepK=3, 3 output panels, 20 K AICs; A ND→NZ | 2656.76 |

The standalone preflight initially rejects both exact source tuples for
unrelated reasons. BASE's 128×256 L0C tile needs the source's one-buffer
policy, while the model defaults to two L0C buffers. Source Split-K uses
555,745,280 bytes of workspace, above the model profile's 512 MiB cap.
MatMulV3 and the benchmark actually allocate it. These are **feasibility
representation mismatches**, not proof that the source tiles are illegal.
The generated domain also omits `singleCoreN=384` at N=1024, so the source
Split-K geometry would need an explicit domain value even after raising the
workspace cap. The online hook currently uses the optimized *family* to
choose a source tiling path; unless `MMV3_ONLINE_TILE=1` passes its guarded
BASE checks, it does not install the optimized M/N/K tile. This accounts
for part of the distance between the model's reported tile and the tile
that actually ran.

The difference is larger for deterministic Split-K than the grid omission
alone suggests. In this shape's NK-order source branch,
`DoDeterministicMultiCoreSplitKTiling()` sets `singleCoreM=M` and
`singleCoreN=stepN×baseN` (2560×384 here), with an optional L2-driven
M split. `SolveSplit` instead permits arbitrary independent outer M/N
extents and selects 128×128. Its modeled workspace is 401,080,320 bytes
(382.5 MiB), versus 555,745,280 bytes (530 MiB) for the source tile.
The 128×128 tuple is therefore an abstract reduced-model candidate, not
the tile the tested source path emitted. The reduced objective also treats
those very different panel structures as nearly tied: 3035.62 µs for
its selected Split-K tuple and 3070.85 µs for the source tuple under the
expanded diagnostic workspace cap. Source-path legality and panel costs
need to be represented before the model's tile can be used for deployment.

Under the original per-baseK-read startup objective, allowing one L0C buffer for BASE and raising only the diagnostic workspace
cap to 2 GiB lets `ReducedObjectiveUs` score the emitted tuples. It then
predicts **2598.55 µs BASE versus 3070.85 µs Split-K**, still choosing the
wrong family, now by 472.31 µs. Thus matching source geometry and relaxing
the cap alone does not fix this case. At K=73728, the source-geometry model
underestimates BASE by 677.68 µs and overestimates Split-K by 417.44 µs;
their errors reverse the measured family gap by 1095.13 µs. The Split-K
source-geometry score here is a diagnostic calculation under the expanded
workspace cap, not a legal result under the current online cap.

## Controlled K-boundary experiment

Two source-backed sweeps held each family's spatial and buffer tiling fixed
at `M=2560, N=1024` and changed only K. Below the threshold, the broader
diagnostic profile selected BASE to force that family; at and above K=65536,
the narrow production profile already selected BASE. Both runs used the
same latest package. The Split-K sweep used the clean default package.
Audited tilings, correctness, and two eight-sample round medians are in the
[BASE summary](long_k_base_sweep_summary.json) and
[Split-K summary](long_k_default_sweep_summary.json). The exact runners are
[BASE](run_long_k_base_sweep.py) and
[Split-K](run_long_k_default_sweep.py). Remote raw profiles live in
`cases/online_long_k_base_sweep_v1` and
`cases/online_long_k_default_sweep_v1` under
`/home/xianfei/solvebase_bench_20261025`.

| K | A conversion | Measured BASE µs | Measured Split-K µs | Model BASE µs on source tile | Model Split-K µs on source tile |
| ---: | --- | ---: | ---: | ---: | ---: |
| 65408 | no | 2315.70 | 1623.99 | 1983.72 | 2418.87 |
| 65536 | yes | 3280.26 | 2795.05 | 2310.34 | 2744.45 |
| 73728 | yes | 3276.23 | 2653.41 | 2598.55 | 3070.85 |

The BASE tile is 128×256, baseK=64, depthA1/B1=16/8 and stepKa/Kb=8/4
at all three K values. The Split-K tile is 2560×384, baseK=128 and
stepKa/Kb=3/3 throughout. Crossing K=65535 turns on A ND→NZ, changes
each kernel key, and increases workspace by `2MK` bytes. The source BASE
kernel calls `MatrixAtoNZV2`, synchronizes, then executes Cube in the same
profiled kernel; Split-K has its own conversion and reduction path. The
measured 65408→65536 rise is 964.56 µs for BASE and 1171.06 µs for
Split-K. The model predicts only 326.62 and 325.57 µs, respectively.
These jumps measure the whole source path change, not isolated conversion
primitive time. Above the boundary, the measured times do not grow
monotonically from K=65536 to 73728, despite identical parsed spatial
tiles. The source's **conversion tile does change**: `baseAN × baseAD` is
19 × 2048 at K=65536 and 47 × 1024 at K=73728. An
[isolated measurement of the exact source AIV routine](../converted_a_probe_v1/README.md)
on devices 4–7 found its median task time drops from 1251.32 to 963.28 µs
at that tile transition. This accounts for a substantial part of the
nonmonotonic full-kernel behavior. A subsequent
[in-context AIC phase sweep](../converted_a_probe_v1/PHASE_MEASUREMENT.md)
measured BASE's post-conversion AIC interval at 2389.62 µs versus
Split-K's 1715.65 µs at K=73728; conversion waits differed by only
54.82 µs. The Split-K reduced objective's per-base-packet read startup
term appears to account for most of its stage overestimate.

The model adds `ConversionUs(M,K)` as a **common term** to both family
objectives. Changing only its rate cannot alter the BASE-versus-Split-K
ranking. The reduced BASE objective uses aggregate GM bytes plus a busiest
AIC packet startup and `max(Cube, GM)` overlap; the Split-K objective uses
panel-level `max(Cube, read)` and a vector chain. Neither has calibrated
costs for the emitted long-K converted-A paths, including their distinct
memory reuse and overlap. The fact that the model already chooses BASE at
K=65408, where neither family converts A, shows the ranking problem is
broader than the threshold term.

## Follow-up L1-step objective

The local standalone model now counts fixed Split-K read startup per
source L1 K step, leaving operand byte service unchanged. For the same
source-emitted geometries it predicts 2598.55 µs BASE and 2369.29 µs
Split-K at K=73728, restoring the measured family ranking. At K=65536
the pair is 2310.34 versus 2113.03 µs, also ranking Split-K first.
See the [phase report and additional-case audit](../converted_a_probe_v1/PHASE_MEASUREMENT.md).
Absolute converted-A estimates remain inaccurate. A subsequent
[source-backed 20-shape comparison](../online_grouped_vs_master_v1/README.md)
built the revision in a separate experimental package: it chose the
same deterministic Split-K source tile as master for this shape and
removed the earlier BASE-family regression. The branch change remains
uncommitted, and one unrelated mode-5 regression remains.

## Corrective direction

1. Score **source-emitted family tiles** for online family selection, and
   represent the source's one-buffer BASE L0C policy and actual Split-K
   workspace requirement. Treat the 512 MiB profile workspace value as a
   configured budget, not a hardware execution limit.
2. Use the [isolated conversion sweep](../converted_a_probe_v1/README.md)
   and [completed in-context phase sweep](../converted_a_probe_v1/PHASE_MEASUREMENT.md)
   to add conversion-tile geometry to the model. L1-step read grouping is
   implemented locally; validate path-specific memory service and overlap across
   source-emitted geometries; do not add a fitted penalty to the common
   conversion term.
3. Keep the committed selector behavior and the revised experimental
   result distinct. The former was 23.3% slower on this target case;
   the latter matched the default source family and tile in the follow-up
   device run. Validate other long-K converted-A shapes before treating
   this branch-specific correction as general.
