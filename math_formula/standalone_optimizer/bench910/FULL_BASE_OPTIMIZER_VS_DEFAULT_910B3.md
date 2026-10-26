# Source-compatible `SolveBase` versus MatMulV3 default on 910B3

The current `SolveBase` selection **did not improve device time** on this
declared seven-shape FP16 BASE suite. After restricting the optimizer to
tuples that the tested plain BASE kernel computes correctly, it selected
`singleCoreM=baseM=256`, `singleCoreN=baseN=128`, `baseK=64`, and 20 AICs
on every shape. The source default used the transposed orientation,
`singleCoreM=baseM=128`, `singleCoreN=baseN=256`, `baseK=64`, and 20 AICs.
The reduced model assigns these orientations **exactly the same score**;
its one-hot geometry solve chose the slower tie. On actual MatMulV3 kernels,
the optimizer orientation took **1.197×** the default time (seven-shape
geometric mean; shape-level 95% interval **1.165–1.230×**).

## Source-executable scope and method

The unrestricted [reduced solver audit](large_wave_solver_audit.csv)
contains outer `singleCoreM/N` tiles larger than `baseM/N`. A research
override of the isolated MatMulV3 host tiler showed that the tested plain
BASE key `0x10000` does not compute these tuples correctly: for example,
`singleCoreM=256,baseM=128` left 4,194,304 output elements zero in
`(2048,4096,8192)`, beginning at row 128; `singleCoreN=256,baseN=128`
left the second 128 columns zero in a direct check. These plans were **not
admitted as timing treatments**. The source default also uses one FP32
L0C buffer for its 128×256 accumulator, whereas the prior standalone
preflight required two and excluded it.

The [source-compatible audit](large_source_compatible_solver_audit.csv)
therefore runs `SolveReducedMip` with
`MipTileDomain::base_core_equals_base=true` and
`CoreMemory::base_l0c_buffers=1`. The source-compatible choice is frozen in
that CSV before the repeated device run. The experimental
[host-tiler patcher](enable_full_base_tiling_probe.py) applies a requested
BASE tile only in an isolated `ops-nn-sribd` clone, checks aligned M/N/K,
L0 capacities, and core count, then invokes MatMulV3's `CalL1Tiling()` and
`DoL2CacheTiling()` for dependent fields. The ordinary default arm uses the
same research package with the override unset. The user's original source
file retained SHA-256
`5bb53987aa5ef6d46611eb5d577aeecdeb077913bab9e617be272fdbee48f4d1`;
the isolated research package's host-tiler library had SHA-256
`a878814476ced09a16dbb1b69f7b2f3e43ae0cb52b7cd4fb21c13bab996b56ec`.

Device 0 was Ascend910B3 with 20 AICs, CANN 9.0.0, driver 25.3.rc1,
and no reported AIC activity immediately before and after the experiment.
All workloads were ND×ND FP16, NN, without bias or ND→NZ conversion.
The [runner](evaluate_full_base.py) used the seven shapes declared in the
prior [large-shape experiment](LARGE_SHAPE_OPTIMIZER_VS_DEFAULT_910B3.md).
For each arm of each pair, it checked every output element against the
exact FP16 input-pattern reference, confirmed BASE key `0x10000`, the
source path, all six selected tile/core fields, 20 launch blocks, and eight
MatMulV3 AICore profiler tasks. Each run used 10 warmups and the median of
eight `msprof` kernel task times. Four rounds randomized both shape order
and arm order within adjacent shape pairs (fixed schedule seed
`20261025`). The 14 initial [smoke runs](source_compatible_base_smoke_raw.jsonl)
are separate and excluded from the inferential analysis.

## Device results

The [56 checked repeated runs](source_compatible_base_repeat_raw.jsonl)
and [448 underlying task times](source_compatible_base_repeat_tasks.jsonl)
are saved. The [analysis](source_compatible_base_analysis.json) reports
the geometric mean of four within-shape paired ratios and a two-sided
Student-t interval on their log ratios. Times below are the medians of
the four per-run medians; the ratio uses paired logs, so it need not equal
the quotient of displayed medians.

| Shape `(M,N,K)` | Default median µs | Optimizer median µs | Optimizer/default | 95% paired interval |
| --- | ---: | ---: | ---: | ---: |
| `(2048,2048,4096)` | 139.792 | 161.178 | 1.153× | 1.147–1.159× |
| `(2048,4096,8192)` | 500.419 | 613.983 | 1.225× | 1.218–1.233× |
| `(4096,2048,8192)` | 501.609 | 625.737 | 1.249× | 1.242–1.255× |
| `(4096,4096,4096)` | 499.276 | 589.771 | 1.181× | 1.180–1.182× |
| `(4096,4096,8192)` | 999.265 | 1224.728 | 1.225× | 1.210–1.240× |
| `(8192,4096,4096)` | 998.312 | 1163.548 | 1.167× | 1.160–1.175× |
| `(8192,8192,4096)` | 1985.454 | 2345.877 | 1.182× | 1.174–1.190× |

The across-suite 1.197× ratio and 1.165–1.230× interval use each shape's
mean paired log ratio as one observation ($n=7$, $t_6$). All seven shape
means favor the default; the exact one-sided sign-test probability under
an equal-direction null is $1/128=0.0078125$. All 28 individual pairs
also favor the default. The seven shapes were deliberately chosen, not
randomly sampled from all workloads, so the interval describes variation
over this declared suite rather than a population-wide performance claim.

## Why the selection fails

For these divisible dimensions, exchanging 128×256 and 256×128 leaves
the model's output tile count, per-core Cube work, packet count, and
aggregate operand-byte expression unchanged. For example, both plans
score **212.167 µs** on `(2048,2048,4096)` under the exploratory
primitive profile. The same equality holds across all seven shapes.
The objective has one combined GM read rate and does not capture the
different A and B transfer behavior, layout effects, or L2 reuse caused
by the tile orientation. Its tie choice is consequently arbitrary and
consistently worse than the source heuristic in this experiment. Absolute
model time is still uncalibrated; the selection result should be judged by
the checked kernel measurements above.

The reduced BASE objective now accepts an optional pair of separate A/B
read primitives and retains source-compatibility/L0C controls used in this
study. The original generic probe did not identify A/B-specific rates, so
the study's tied scores and negative device result remain unchanged. A
[later operand-shaped transfer probe](directional_read/README.md) measured
effective A and B service independently. It correctly predicts that the
source's 128×256 orientation wins, but the reduced objective predicts a
79.7 µs gap on the 2048 square versus the measured 21.4 µs. The probe's
cache-hot, isolated-transfer rates are therefore not installed as a
calibrated hardware profile. Resolve L2 reuse and A/B overlap, then freeze
a revised model and a new held-out suite before another speedup claim.
