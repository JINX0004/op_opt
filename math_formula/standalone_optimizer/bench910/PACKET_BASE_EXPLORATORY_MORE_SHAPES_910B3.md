# More source-backed optimizer/default shapes on 910B3

The current packet-aware `SolveBase` selector **did not improve source BASE
kernel time** on four new shapes where it chose a tile or core count
different from MatMulV3's default. Three swapped 256×128 choices were
5.1–13.0% slower by four-round paired geometric mean; a 19-versus-20-core
choice was effectively tied. All optimizer outputs were exact. A fifth
declared control shape selected the default tile and therefore had no
distinct optimizer treatment. This is an exploratory, model-screened
suite, not a random sample of matrix shapes.

## Frozen selection and device method

The local C++ solver first screened a 108-shape grid with the optional
source-step packet primitive profile. It selected the common 128×256,
20-core tile on 90 shapes, a swapped 256×128 tile on 15, and 128×256 with
19 cores on three. These are **solver choices**; source defaults were not
audited for the entire grid. The five
[declared study shapes](packet_base_exploratory_shapes.json) were fixed
before any new device timing to cover four different choices and one
same-choice control. Their [solver audit](packet_base_exploratory_solver_audit.csv)
and [source-default audit](packet_base_exploratory_defaults.jsonl) were
hashed into the [frozen selection](packet_base_exploratory_selection.json)
before paired repeats.

The solver used `base_core_equals_base=true`, one FP32 L0C buffer, 20
available AICs, M/N grid `{64,128,256}`, and the independently measured
source-step A/B packet parameters. It used no whole-kernel fitted
correction. All five source defaults emitted `(singleCoreM,singleCoreN,
baseM,baseN,baseK,cores)=(128,256,128,256,64,20)`.

Device 0 was Ascend910B3 with CANN 9.0.0, FP16 ND×ND, NN, no bias, and
source-built MatMulV3 BASE key 65536. Every optimizer and default run
verified the full output exactly, the six-field emitted tile, 20 or 19
launch blocks as appropriate, the source path, and eight profiler kernel
tasks. Ten warmups preceded each run. Four paired rounds randomized shape
and arm order with seed `20261026`; each cell below is the median of four
run medians. The [eight smoke runs](packet_base_exploratory_smoke_raw.jsonl)
were excluded. The [32 repeated runs](packet_base_exploratory_repeat_raw.jsonl),
[256 underlying task times](packet_base_exploratory_repeat_tasks.jsonl),
and [paired analysis](packet_base_exploratory_analysis.json) are saved.
The ratio uses paired log times; its interval is a two-sided Student-t
95% interval over four round-level pairs for that one shape.

| M × N × K | Optimizer change from default | Default µs | Optimizer µs | Optimizer/default, 95% interval |
| --- | --- | ---: | ---: | ---: |
| 4096 × 384 × 4096 | 256×128 instead of 128×256 | 68.474 | 73.552 | 1.0707 [1.0558, 1.0858] |
| 6144 × 640 × 8192 | 256×128 instead of 128×256 | 277.825 | 292.134 | 1.0506 [1.0219, 1.0800] |
| 4096 × 896 × 8192 | 19 instead of 20 cores | 252.965 | 253.352 | 1.0019 [0.9988, 1.0049] |
| 8192 × 384 × 16384 | 256×128 instead of 128×256 | 678.764 | 765.520 | 1.1301 [1.0825, 1.1797] |
| 6144 × 768 × 8192 | Same tile | — | — | No distinct treatment |

The optimizer was faster in **zero of 16** paired rounds. The across-four
shape geometric mean ratio was 1.0623, but its shape-level 95% interval
was [0.9814, 1.1499]; with only four deliberately selected shapes, that
aggregate is descriptive and does not establish a population-wide
regression. The three orientation cases each have intervals above one.

The prior [tail-heavy packet study](PACKET_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
found that the reduced model's round-robin AIC assignment did not match
the source BASE L2/tail scheduler, which can reverse a tile ranking.
This follow-up shows the problem extends beyond its single earlier
shape. The 19-core case gives no evidence that withholding an AIC helps.
Before using the selector for automatic tiling, reproduce source block
assignment and improve the packet/tail timeline, then validate on new
shapes chosen before viewing their device outcomes.

This comparison is specifically **plain BASE key 65536**. The
[B-in-L1 full-load implementation](FULL_LOAD_CORE_TILING_SUPPORT_910B3.md)
does support larger single-core tiles, but the current optimizer has no
separate objective or legal override for it. Its small-K shapes cannot
be included in a fair optimizer/default comparison by merely disabling
`base_core_equals_base`.
