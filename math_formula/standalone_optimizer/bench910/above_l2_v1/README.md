# Above-L2 SolveBase comparison on 910B3

The current packet-aware SolveBase optimizer chose the source default tile for
9 of 12 declared FP16 workloads whose combined A, B, and C tensor footprint
exceeds the documented 192 MiB L2 capacity. On the three workloads where it
chose a different tile, the optimizer's 256×128 tile was slower than the
source default's 128×256 tile in all four measured pairs per workload. This
suite was selected to probe large working sets, not sampled randomly from all
MatMulV3 workloads; the 9 same-tile cases have no separate device comparison.

## Selection and workload audit

The [declared shapes](screen_shapes.json) were fixed before device timing.
For each FP16 ND×ND NN MatMulV3, footprint is
`2(MK + KN + MN) / 2^20` MiB, including one output tensor. This is a
capacity screen, not a measured L2-miss count or a proof that all three
tensors must reside in L2 concurrently. The A tensor alone is 192 MiB for
10 shapes, 256 MiB for one, and 384 MiB for one.

The [local solver audit](screen_solver_audit.csv) used the independently
measured source-step packet profile, the balanced-class BASE load abstraction,
20 AICs, `base_core_equals_base=true`, one FP32 L0C buffer, and the
source-compatible `{64,128,256}` M/N tile grid. The fresh
[source-default audit](defaults.jsonl) and solver audit were hashed into the
[frozen selection](selection.json) before timing. The optimization objective
has no whole-MatMul fitted correction. All source defaults emitted
`(singleCoreM,singleCoreN,baseM,baseN,baseK,cores) =
(128,256,128,256,64,20)`.

| M × N × K | A+B+C MiB | Optimizer M×N tile | Outcome |
| --- | ---: | ---: | --- |
| 8192 × 384 × 12288 | 207 | 256×128 | Device paired comparison |
| 8192 × 384 × 24576 | 408 | 256×128 | Device paired comparison |
| 8192 × 640 × 12288 | 217 | 128×256 | Same as source |
| 8192 × 1024 × 12288 | 232 | 128×256 | Same as source |
| 6144 × 640 × 16384 | 219.5 | 256×128 | Device paired comparison |
| 6144 × 1024 × 16384 | 236 | 128×256 | Same as source |
| 4096 × 1024 × 24576 | 248 | 128×256 | Same as source |
| 4096 × 2048 × 24576 | 304 | 128×256 | Same as source |
| 12288 × 384 × 8192 | 207 | 128×256 | Same as source |
| 12288 × 640 × 8192 | 217 | 128×256 | Same as source |
| 12288 × 1024 × 8192 | 232 | 128×256 | Same as source |
| 16384 × 384 × 8192 | 274 | 128×256 | Same as source |

## Device method and results

Device 0 was a Huawei 910B3 with CANN 9.0.0. The source-built MatMulV3
plain BASE kernel used key 65536, FP16 ND×ND inputs, NN layout, no bias,
no A/B conversion, and 20 launched AIC blocks. The optimizer arm requested
the six-field tile `(256,128,256,128,64,20)` through the research tile
override; the default arm left the override unset. Every run checked all
output elements exactly, the emitted tile, kernel key, and source path. Ten
warmups preceded each run. Four paired rounds randomized shape and arm order.
Each response was the median of eight profiled **device kernel task times**,
not a host event duration. The six [smoke runs](smoke_raw.jsonl) and their
[48 tasks](smoke_tasks.jsonl) were excluded. The [24 timed runs](repeat_raw.jsonl),
[192 task times](repeat_tasks.jsonl), and [paired analysis](analysis.json)
are retained.

| M × N × K | Model default / optimizer µs | Device default / optimizer µs | Paired optimizer/default and 95% interval |
| --- | ---: | ---: | ---: |
| 8192 × 384 × 12288 | 367.510 / 344.882 | 412.419 / 422.154 | 1.0385 [0.9838, 1.0962] |
| 8192 × 384 × 24576 | 730.000 / 685.067 | 1021.551 / 1069.685 | 1.0410 [1.0096, 1.0734] |
| 6144 × 640 × 16384 | 563.146 / 549.456 | 686.044 / 773.231 | 1.1233 [1.0866, 1.1613] |

Device columns are medians of four run medians. Each ratio is the geometric
mean of four within-round ratios; its interval is a two-sided Student-t
interval over those four paired log ratios. The first interval includes 1,
so those four pairs do not resolve that small difference at this interval
level. The across-shape geometric mean ratio is 1.0669, descriptive only.
These intervals describe repeat variation for each tested shape, not
performance over unseen shapes.

## Interpretation

The solver ranked 256×128 ahead by 2.4–6.2% in modeled time, while device
measurements ranked it behind by 3.8–12.3% in paired geometric mean time.
For the matched 8192×384 pair, doubling K from 12288 to 24576 nearly
doubled the modeled default time (367.5 to 730.0 µs) but increased measured
default time by about 2.48× (412.4 to 1021.6 µs). The model's absolute
underprediction of the default rises from 12.2% to 39.9% when stated as
measured/model minus one. This is consistent with a missing K-dependent
traffic or scheduling cost, but this comparison alone cannot attribute it
specifically to L2 misses. A useful next calibration experiment would hold
M, N, tile, and source path fixed while sweeping K across the L2-capacity
boundary and collecting L2/HBM traffic counters alongside kernel times.

For this tested large-working-set suite, the selector often agrees with the
source default, but its three distinct recommendations are not improvements.
It should remain diagnostic until the above-L2 objective and tile ranking
are validated on additional workloads.
