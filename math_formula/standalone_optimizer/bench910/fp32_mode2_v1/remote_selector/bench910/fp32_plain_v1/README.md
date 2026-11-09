# First plain-FP32 MatMulV3 model on 910B3

This is the first extension beyond same-type FP16/BF16 in the online
branch's CANN-independent reduced model. It covers **same-type FP32 A/B/C,
FP32 accumulation, NN, ND/ND/ND, no bias, HF32 disabled**, and the source's
plain BASE and deterministic multi-core Split-K families. It is a
model/CLI experiment; the product MatMulV3 selector still invokes the
online model only for FP16/BF16. The experiment ran on Ascend 910B3 device
4 with installed CANN 9.0.0, against the clean-master MatMulV3 package.

## Source audit and modeled constraints

The [source audit](source_audit.json) checked emitted tiling and full-output
correctness on six FP32 shapes. The maximum observed relative output error
was zero for the exact-input pattern. Five of these shapes are inside the
current model domain:

| M×N×K | Source mode | Source `baseM×baseN×baseK` | Source core `M×N×K` | FP32 model |
| --- | ---: | --- | --- | --- |
| 256×256×1024 | BASE | 32×128×64 | 32×128×1024 | admitted |
| 128×128×32768 | deterministic Split-K | 128×128×64 | 384×128×192 | admitted |
| 2048×2048×4096 | BASE | 128×256×32 | 128×256×4096 | admitted |
| 512×512×4096 | BASE | 112×128×64 | 112×128×4096 | admitted |
| 49152×384×1536 | BASE | 128×256×32 | 128×256×1536 | admitted |
| 16×16×65536 | deterministic Split-K + A conversion | 128×128×64 | 384×16×192 | deferred |

The model now uses four-byte FP32 A/B/output, 256-byte transfer
alignment (`K` and directly transferred `N` multiples of 64), and the
existing FP32 L0C accumulator footprint. Deterministic Split-K uses
`baseK=64` for FP32; FP16/BF16 retain `baseK=128`. The Split-K reduced
timeline and L1 step bound use that dtype-specific baseK. The source's
FP32 converted-A path at `K>65535`, FP32 single-core Split-K paths, and
HF32 are explicitly excluded. The existing no-bias and layout constraints
remain in effect. These legality changes are in the remote branch
`codex/matmul-v3-mip-selector`, under `experimental/mip_selector/model`.

The source can launch 20 blocks for `256×256×1024` even though only 16
output tiles do work. For source scoring, the model uses 16 **occupied**
AICs; its plan variable does not represent idle launched blocks.

## Initial timing profile

The [profiler records](source_profiles.json) contain five MatMulV3 kernel
tasks per shape after five warmups. The experimental FP32 profile retains
the measured FP16/BF16 byte-service rates as provisional transfer rates
and sets effective Cube service to **3.6 million FLOP/µs**. That value was
chosen using the large `2048×2048×4096` BASE source tile. The remaining
source tiles are checks, not additional fits. These are aggregate reduced
model coefficients; the rate is not a claim about peak FP32 hardware
throughput.

| M×N×K | Measured kernel µs | Reduced score of emitted tile µs | Error |
| --- | ---: | ---: | ---: |
| 256×256×1024 | 11.12 | 9.23 | −17.0% |
| 128×128×32768 | 42.12 | 43.00 | +2.1% |
| 2048×2048×4096 | 516.08 | 524.64 | +1.7% (fit shape) |
| 512×512×4096 | 39.52 | 37.42 | −5.3% |
| 49152×384×1536 | 821.56 | 840.85 | +2.3% |

The [source-tile scorer](score_source.cpp) evaluates the audited tile
without replacing its geometry. Its [inputs](source_score.in),
[outputs](source_scores.csv), and the [Cube-rate sweep](score_grid.csv)
are saved. For the five admitted shapes, the [model CLI decisions](model_decisions.csv)
selected the same implementation **family** as the source. Their tiling
dimensions sometimes differ, so the family check is not a claim that
the model can emit the chosen tile through MatMulV3.

## Status and next validation

The branch's dtype tests and FP16 golden decisions passed after the
extension. The FP32 profile is intentionally separate from FP16/BF16,
and the product selector's FP32 gate remains closed. Before enabling FP32
online selection, broaden the device dataset around BASE/Split-K
boundaries, model the FP32 single-core and converted-A paths that occur
there, verify source-emittable tiling, and recalibrate FP32 transfer and
small-shape startup terms. HF32 should have its own compute-mode choice
and numerical validation.

Reproduction programs: [FP32 MatMulV3 harness](fp32_bench.cpp),
[source audit](audit_source.py), [kernel profiler](profile_source.py),
and [source-tile scorer](score_source.cpp). Raw `msprof` traces remain on
910 under `/home/xianfei/solvebase_bench_20261025/cases/fp32_plain_v1`.
