# FP32 family-boundary test on 910B3

This extends [the first plain-FP32 profile](../fp32_plain_v1/README.md)
from six shapes to **27 cases** around three clean-master MatMulV3 family
transitions. The test ran on 910B3 device 4 with CANN 9.0.0. It covers
same-type FP32 NN, ND/ND/ND, no bias, and HF32 disabled. The clean-master
package was `/home/xianfei/solvebase_bench_20261025/ops_nn_master_bf16_proto_package`;
the model came from the current `codex/matmul-v3-mip-selector` branch's
`experimental/mip_selector/model`. The **product selector's FP32 gate remains
closed**, so this is a source-versus-model audit, not an online-versus-master
performance A/B test.

## Cases and methods

The [27-case dataset](dataset.json) brackets source thresholds at K=6144 for
small M/N, K=7680 (=20 AICs × 384) for low spatial occupancy, and K=27392
for the general long-K path. It also checks narrow N=64 at M=2048 and 4096.
The [source audit runner](run_source_audit.py) launched each case through the
installed CANN MatMulV3 API, checked full-output correctness, verified the
clean-master tiling provenance, and recorded emitted tiles. The
[profile runner](profile_source.py) measured **16** paired boundary cases with
`msprof`: median of five kernel tasks after five warmups. Device 4 was checked
idle around each run.

The [model runner](run_model.py) rebuilt the `mmopt_selector` CLI and used
`ExperimentalFp32910B3()` and
`SourceCompatibleDomain()` on the same shapes. The
[family diagnostic](family_scores.cpp) recorded the separate BASE and
deterministic Split-K objectives. For source tiles admitted by the model, the
[source-tile scorer](../fp32_plain_v1/score_source.cpp) recomputed the reduced
objective. The [summarizer](summarize.py) joins these records. The compact
[comparison table](results/comparison.csv), [all model family scores](results/family_scores.csv),
and per-case audit and profile JSONs under `results/` are checked in. Raw
`msprof` traces remain on 910 under
`/home/xianfei/solvebase_bench_20261025/cases/fp32_family_boundary_v1`.

## Family decisions

The model matched master on **14/27** cases. All 27 audits passed the FP32
output check. Both master and model selected only BASE (mode 0) and
deterministic Split-K (mode 3) in this dataset.

| Boundary group | Agreement | Main mismatch |
| --- | ---: | --- |
| Small M/N, K≈6144 | 2/6 | Master switches to Split-K at 6144; model keeps BASE. |
| M/N=128 or 256, K≈7680 | 2/6 | Master switches to Split-K at 7680; model keeps BASE. |
| General K≈27392 | 8/9 | At 512×512×27328, model selects Split-K while master uses BASE. |
| N=64, K≈27392 | 2/6 | Master switches to Split-K at 27392; model keeps BASE. |

These are **decision disagreements**, not measured speedups or regressions.
Master's source heuristic is a comparison point, not an optimality oracle.

## Kernel timing at adjacent boundary shapes

The following compares the source-selected kernels on *neighboring K values*.
It does not time both candidate families at the same shape.

| M×N | K below / source mode / measured µs | K at threshold / source mode / measured µs | Model family at threshold; BASE / Split-K objective µs |
| --- | --- | --- | --- |
| 16×16 | 6080 / BASE / 16.62 | 6144 / Split-K / 16.78 | BASE; 5.69 / 12.26 |
| 64×64 | 6080 / BASE / 18.06 | 6144 / Split-K / 19.44 | BASE; 8.87 / 12.89 |
| 128×128 | 7616 / BASE / 32.26 | 7680 / Split-K / 27.88 | BASE; 17.91 / 20.65 |
| 256×256 | 7616 / BASE / 34.68 | 7680 / Split-K / 37.84 | BASE; 33.00 / 39.53 |
| 512×512 | 27328 / BASE / 219.62 | 27392 / Split-K / 264.98 | Split-K; 252.01 / 232.35 |
| 2048×64 | 27328 / BASE / 218.62 | 27392 / Split-K / 250.68 | BASE; 187.50 / 195.22 |

The most useful contrast is around K=7680: the source switch coincides with
a lower time for 128×128, but a higher time for 256×256. This suggests that a
shape-dependent decision can improve upon a single threshold, but a same-shape
forced-family test is needed to establish the benefit. At 512×512×27392 the
model's Split-K preference is only 7.8% over its BASE objective, while the
neighboring source measurements differ in the opposite direction by 20.6%.

## Model accuracy and coverage

The current source-tile scorer admitted **20/27** emitted tiles; seven are
outside its geometry rules. These include source padding beyond the logical
M/N shape and source Split-K tiles whose base N exceeds the clipped
single-core N. The model's plan domain should represent those source layouts
before treating them as illegal. For the 12 tiles both scored and profiled,
median absolute timing error was **9.7%**. Mean absolute error was **25.8%**,
dominated by the 16×16×6080 BASE tile (44.71 µs model versus 16.62 µs
measured). Excluding that tiny tile, mean absolute error was **12.8%** for
11 cases. These errors describe *source-tile scoring*, not runtime of the
model's selected tile.

The model's optimized BASE objective is especially optimistic near the
7680 transition: 17.91 µs for 128×128×7680 versus 32.26 µs for the
source BASE tile at the neighboring K=7616. The source tile itself scores
38.40 µs at K=7616, so the model's selected BASE geometry differs greatly
from the source-emitted tile. The model also undershoots the measured
2048×64×27392 source Split-K tile by 19.3%. Its Split-K score is flat over
some adjacent K values because the reduced panel calculation has discrete
round counts; this is a modeling approximation, not a measured plateau.

### FP32 BASE K-step diagnostic

For FP32 BASE, the current generic score counts A and B GM-read startups
as `2 × ceil(K/baseK)` per output packet. Source tiling instead groups
`baseK` slices into `stepKa` and `stepKb` L1 windows, so the corresponding
request count is `ceil(K/(stepKa×baseK)) + ceil(K/(stepKb×baseK))`.
The existing `SourceBaseL1Steps` helper is only used by the BF16 narrow-B
path and hardcodes two-byte packets; FP32 needs four-byte packet sizing.
With that size changed, the helper reproduced `stepKa` and `stepKb` for all
seven source BASE cases in this dataset.

I tested a **temporary, undeployed** model variant that sizes those packets
for FP32 and counts read startups per L1 window. The
[diagnostic records](kstep_diagnostic.csv) show the result for source-emitted
tiles:

| M×N×K | Modeled versus source L1 read requests per packet | Current score µs | K-step variant µs | Measured µs |
| --- | ---: | ---: | ---: | ---: |
| 16×16×6080 | 380 versus 6 | 44.71 | 3.71 | 16.62 |
| 128×128×7616 | 238 versus 34 | 38.40 | 16.04 | 32.26 |
| 256×256×7616 | 238 versus 38 | 50.00 | 28.08 | 34.68 |
| 512×512×27328 | 854 versus 214 | 232.42 | 220.33 | 219.62 |

Thus the present per-baseK startup charge is structurally wrong, but simply
correcting it makes the small BASE predictions too low. It currently masks
another omitted cost. The reduced model explicitly has no fixed per-packet
Cube latency; FP32 transfer rates were inherited from FP16/BF16 and Cube
service was fitted at a large tile. Small output tiles need a measured
K-loop/Cube and transfer-service term before changing the production model.

### FP32 best-tiling search audit

The CLI minimizes its declared reduced objective correctly within its
enumerated candidates, but **that candidate set is not the MatMulV3 FP32
emission domain**:

- `SourceCompatibleDomain()` only requires BASE `coreM/N=baseM/N`. The
  default base grid is mostly `{16,32,64,128,256}`, so the source's
  `112×128` BASE tile is absent. Base and core values above the logical
  axis are generally rejected, while source uses padded N=128 or 384
  for several N=64 and small-output cases.
- The geometry builder adds a 384-sized Split-K core only if `K>65535`;
  the FP32 model rejects all `K>65535` workloads. Hence none of the
  source-emitted 384-sized Split-K cores in this dataset were in the
  search grid. Only **3/27** complete source spatial geometries occurred
  in the default grid.
- Generic BASE scoring has a decreasing `ceil(K/baseK)` startup term, so
  `SolveBase` chooses the largest feasible `baseK` analytically instead
  of searching it. This is sound for that surrogate, but not for a
  physical L1-step model whose packet counts and pipeline overlap change
  at discontinuous `baseK` values. The model chose `baseK=512` for
  `16×16×6080`; source emitted `baseK=32`.
- The CLI uses the default two-buffer BASE L0C preflight, whereas source
  plain BASE can use one L0C buffer. That excludes source tiles such as
  FP32 `128×256` from the default domain. FP32 modes 2/5/6 and the
  converted-A path above `K=65535` are not searched at all.
- The product `DoSelectTiling` gate excludes FP32. Its online lookup also
  does not select the experimental FP32 hardware profile. The CLI's FP32
  proposed tile therefore is neither the product's current tile nor an
  online FP32 candidate that has been device validated.

Two isolated solver builds tested whether simple domain widening fixes the
27-case decisions. Allowing 384-sized FP32 Split-K cores changed **0/27**
winners. Using one BASE L0C buffer also changed **0/27** winners, and did not
change the `2048×2048×4096` model winner. These changes were diagnostic only;
the product branch was not modified. The reduced objective still prefers
smaller Split-K cores and different BASE geometries when source choices are
available. Thus the search is an optimizer of a restricted surrogate, not a
verified best-tiling search for FP32 MatMulV3.
The remote `mmopt_dtype_tests` and FP16 golden test both passed; they do not
establish FP32 source-emittable search coverage.

## Decision and follow-up

Keep the FP32 product gate closed. The [same-shape forced-family experiment](forced_family_v1/README.md)
now times both source-emitted BASE and deterministic Split-K at seven
boundary shapes. It finds each of the model and master family choices faster
in 3/7 cases; neither rule consistently wins. The results isolate a wrong
fixed-tile family ranking at 256×256×7680 and free-search domain/objective
differences at 512×512. Fit the small-tile startup and narrow-N Split-K
costs and extend geometry legality to source padding before another
family-boundary audit. These datasets do not cover FP32 single-core Split-K,
converted-A at K>65535, or HF32.

## Reproduction on 910

Copy this directory to
`/home/xianfei/solvebase_bench_20261025/fp32_family_boundary_v1`, then run:

```bash
python3 /home/xianfei/solvebase_bench_20261025/fp32_family_boundary_v1/run_source_audit.py
python3 /home/xianfei/solvebase_bench_20261025/fp32_family_boundary_v1/profile_source.py
python3 /home/xianfei/solvebase_bench_20261025/fp32_family_boundary_v1/run_model.py
```

`run_model.py` builds the model programs, writes both objective tables, and
invokes `summarize.py`. It accepts no device arguments; only the source audit
and kernel profiler use device 4. The model CLI accepts one `M N K fp32` row
per shape.
