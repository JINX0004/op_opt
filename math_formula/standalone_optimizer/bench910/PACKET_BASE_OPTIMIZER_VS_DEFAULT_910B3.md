# Packet-aware `SolveBase` versus MatMulV3 default on 910B3

**The new objective did not improve the source BASE tiling in this
experiment.** On 11 of the 12 audited shapes, it selected exactly the
default `(singleCoreM,singleCoreN,baseM,baseN,baseK,cores) =
(128,256,128,256,64,20)`, so there was no distinct optimizer treatment.
On `(6144,384,8192)`, the only different selection, it chose
`(256,128,256,128,64,20)`. The source kernel computed every output
correctly, but that tile was **15.94% slower** than the default in six
randomized paired rounds (optimizer/default geometric mean 1.1594,
two-sided paired 95% interval 1.1524–1.1665). This is evidence against
improvement on this declared suite, not a population-wide claim about all
shapes.

## Selection frozen before repeated timing

The earlier [seven large-shape study](FULL_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
found that the byte-only objective tied 128×256 and 256×128 and selected
the slower orientation. With the [source-step MTE2 primitive profile](l1_step_read/README.md),
`SolveReducedMip` now selects the source's 128×256 default on all seven;
no new device A/B run is needed when the plans are identical.

For a new comparison, [`evaluate_packet_base.py`](evaluate_packet_base.py)
audited eight different aspect-ratio shapes. Four thin-matrix cases were
added as a separate cohort after the first defaults matched the optimizer,
but **before the selection file was frozen or any paired repeat timing**.
This is an exploratory, deliberately chosen suite. The final 12 shapes,
source defaults, and solver choices are preserved in
[`packet_base_defaults.jsonl`](l1_step_read/packet_base_defaults.jsonl),
[`packet_base_suite_audit.csv`](l1_step_read/packet_base_suite_audit.csv),
and the SHA-linked [`packet_base_selection.json`](l1_step_read/packet_base_selection.json).

The C++ solver used `base_core_equals_base=true`, one FP32 L0C buffer,
20 available AICs, and the explicit M/N grid `{64,128,256}`. It optimized
legal aligned baseK and occupied cores for each geometry. The optional A/B
packet coefficients came only from independent transfer probes; no
whole-MatMul timing was fitted. The ordinary source default was read from
the emitted tiling, not guessed. The isolated source package's tiler
library SHA-256 was
`a878814476ced09a16dbb1b69f7b2f3e43ae0cb52b7cd4fb21c13bab996b56ec`.
Its research override was used only for the optimizer arm; the default
arm left the override unset.

## Device measurement of the one different plan

The test used Ascend910B3 device 0, 20 AICs, CANN 9.0.0, NN FP16 ND×ND,
and MatMulV3 BASE key `65536` with no A/B conversion. Every run checked
all output elements, source path, the entire emitted six-field tile tuple,
20 launch blocks, and eight AICore profiler tasks. Ten warmups preceded
each run. Each paired round randomized arm order with seed `20261026`;
the response is the median of eight device **kernel task times**, not an
AscendCL host event duration. The two [smoke runs](l1_step_read/packet_base_smoke_raw.jsonl)
were excluded. The [12 repeated runs](l1_step_read/packet_base_repeat_raw.jsonl)
and [96 verified underlying task times](l1_step_read/packet_base_repeat_tasks.jsonl)
are retained. [`analyze_packet_base.py`](analyze_packet_base.py) produces
the [analysis JSON](l1_step_read/packet_base_repeat_analysis.json).

| `(6144,384,8192)` | 128×256 default | 256×128 optimizer |
| --- | ---: | ---: |
| Reduced objective | 190.251 µs | 185.665 µs |
| Median of six run medians | 169.895 µs | 196.970 µs |
| Output tiles | 96 | 72 |
| Source `stepKa/stepKb` | 8 / 4 | 4 / 8 |

The model predicted a 2.4% improvement; the device measured a 15.94%
slowdown. All six paired ratios were above 1 (range 1.149–1.167). The
95% interval above uses a Student-t interval on the six **paired log
ratios**. It characterizes run-to-run variation for this one shape;
11 same-tile shapes cannot contribute independent speedup evidence.

## Why the reduced objective missed this case

[`BaseCoreLoads`](../mip_solver.cpp) assigns output tile $(i,j)$ to AIC
$(iJ+j)\bmod c$. Here the default has $J=2$ N tiles and $c=20$, so the
model sends all full 256-column tiles to even-numbered AICs and all
128-column tail tiles to odd-numbered AICs. Its busiest AIC gets five full
tiles, implying **182.37 µs of Cube service** at the independently measured
14.719 million FLOP/µs/AIC rate. The actual default kernel takes about
170 µs in total. Thus this modeled assignment cannot describe the source
execution for this shape. The source BASE kernel instead uses L2 tile
rounds and tail-aware block mapping in
[`MatmulBaseBlock::InitBlockIndex`](../../../mat_mul_v3/op_kernel/mat_mul_base_block.h)
and `UpdateBasicIndex`; the host recalculates L2 tiling after the
research override. The packet primitive law also had an 18.3% slope
error on one held-out transfer shape, so its extrapolation to tails is
another uncertainty. The experiment isolates neither contribution, but
the core-assignment discrepancy alone invalidates this selection.

The next model change should reproduce the source L2 tile and block
assignment, including full/tail output extents, before comparing per-AIC
Cube and MTE2 demand. Then rerun selection on unseen tail-heavy shapes;
the optional packet profile should not be treated as a validated general
BASE selector yet.
