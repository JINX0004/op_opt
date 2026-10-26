# Abstract balanced-class BASE rerun on 910B3

The source-inspired balanced-class load abstraction in
[`BaseCoreLoads`](../../mip_solver.cpp) **did not correct the tile ranking**
on the three previously regressing shapes. It still selected 256×128 where
the source default emitted 128×256. Every optimizer and default run produced
exact output, but the optimizer tile was slower in all 12 randomized paired
rounds. These shapes were chosen because their earlier device trials had
already shown a regression; this is a retest, not a new population sample.

## Frozen selection and device method

The [declared shapes](abstract_balance_three_shapes.json) were fixed before
this rerun. The updated local C++ solver used the independently measured
source-step packet profile, `base_core_equals_base=true`, one FP32 L0C
buffer, 20 AICs, and the `{64,128,256}` M/N tile grid. Its
[selection audit](abstract_balance_three_solver_audit.csv) and the fresh
[source-default audit](abstract_balance_three_defaults.jsonl) were hashed
into the [frozen selection](abstract_balance_three_selection.json) before
timed comparisons. There was no whole-MatMul fitted correction.

Device 0 was a Huawei 910B3 with CANN 9.0.0. The source-built MatMulV3
plain BASE kernel used key 65536, FP16 ND×ND inputs, NN layout, no bias,
and no A/B conversion. The optimizer arm requested
`(singleCoreM,singleCoreN,baseM,baseN,baseK,cores) =
(256,128,256,128,64,20)` through the research tile override. The default
arm left the override unset and emitted `(128,256,128,256,64,20)`.
Each run checked the complete output, the emitted six-field tile, kernel
key, source path, and eight profiled kernel tasks. Ten warmups preceded
each run. Four paired rounds randomized shape and arm order. Each run's
response was the median of eight device kernel task times, not a host event
duration. The six [smoke runs](smoke_raw.jsonl) and their
[48 tasks](smoke_tasks.jsonl) were excluded. The [24 repeated
runs](repeat_raw.jsonl), [192 underlying task times](repeat_tasks.jsonl),
and [paired analysis](analysis.json) are retained.

| M × N × K | Model default / optimizer µs | Device default / optimizer µs | Optimizer/default, paired 95% interval |
| --- | ---: | ---: | ---: |
| 4096 × 384 × 4096 | 72.095 / 71.806 | 69.351 / 73.717 | 1.0621 [1.0460, 1.0785] |
| 6144 × 640 × 8192 | 284.315 / 277.308 | 277.170 / 294.178 | 1.0613 [1.0530, 1.0696] |
| 8192 × 384 × 16384 | 488.340 / 458.277 | 670.249 / 758.684 | 1.1388 [1.0654, 1.2172] |

The device columns are medians of four run medians. Each ratio is the
geometric mean of four within-round optimizer/default ratios; its interval
is a two-sided Student-t interval over those four paired log ratios. The
across-shape geometric mean ratio is 1.0868, a descriptive statistic for
these deliberately selected shapes. The confidence intervals describe
variation in repeated runs for each shape, not performance over unseen
workloads.

## Interpretation

The abstraction balances the number of full and tail tiles without
enumerating output tiles or assigning different performance rates to
individual Cube cores. It corrected the predicted choice for one earlier
tail-heavy shape, `(6144,384,8192)`, but all three shapes here still have
the wrong predicted ranking. This retest shows that balanced tile counts
alone do not capture the orientation cost; source L2 reuse, transfer
dependence, and packet slope remain possible missing terms. The current
selector should remain diagnostic rather than replacing the source default.
