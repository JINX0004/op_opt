# Optimizer versus MatMulV3 default on 910B3: current evidence

## Question and acceptance rule

The intended experiment is whether a tiling selected *before looking at its
device time* outperforms the source MatMulV3 default. A superiority claim
requires a positive lower 95% confidence bound for paired log speedup across
independent shapes; a useful practical gate is a lower bound above
`log(1.05)`, equivalent to at least 5% speedup. Kernel task time, not host
event time, is the outcome. Correctness and the emitted tiling must pass for
both arms. A source-executable plan is required before timing comparisons.

**Result: the current evidence does not validate a speedup.** In the
source-executable, M-only restriction described below, the model chose the
default tile for all 12 declared workloads. Thus there is no different
optimizer arm to time, and no valid 95% superiority test for that restriction.
The unconstrained `SolveReducedMip` proposed different multi-field tuples,
but these were not executed: the source benchmark currently supports only
a checked M-tile override, and the reduced equation fails absolute-time
validation. Neither result should be presented as proof that the full
optimizer is better or worse than the source default.

## Reproducible selection experiment

The [experiment script](compare_optimizer_default.py) fixes the 12 workloads
before selecting tiles: M/N/K values `(128,256,1024/4096)`,
`(256,256,1024/4096)`, `(256,512,1024/4096)`,
`(512,256,1024/4096)`, `(512,512,1024/4096)`, and
`(1024,512,4096)`, `(1024,1024,4096)`. They use FP16 ND inputs, no
transpose or bias, and the BASE path. The remote 910B3 ran the user's
source-matched MatMulV3 from the isolated package. All 12 defaults passed
the complete output check and emitted BASE with no ND→NZ conversion. Their
tiling tuples are in
[`default_cases_optimizer_comparison.jsonl`](default_cases_optimizer_comparison.jsonl),
and the eight raw profiler task times per workload are in
[`default_tasks_optimizer_comparison.jsonl`](default_tasks_optimizer_comparison.jsonl).
The original remote source file retained SHA-256
`5bb53987aa5ef6d46611eb5d577aeecdeb077913bab9e617be272fdbee48f4d1`.

For this source-executable restriction, `baseN`, `singleCoreN`, `baseK`,
and launch core count are fixed to the emitted default. Candidate
`baseM=singleCoreM` values are every 16-aligned value from 16 through the
default M tile. The override has been checked for downward M changes only;
the default is included. The selector minimizes the current BASE reduced
objective on those candidates, using the exploratory 20-AIC Cube, L2-hot
read/write, and packet coefficients from
[`PRIMITIVE_910B3_CANN9.md`](PRIMITIVE_910B3_CANN9.md). Selection happens
without using candidate device times. The [frozen choices and per-candidate
objective values](optimizer_comparison_selection.json) show `12/12`
default-to-default selections. Consequently `compare` has no nonidentical
pairs to run. The equality is a decision-level result, not an estimated
kernel speedup from repeated timing.

The earlier [five-round randomized controlled comparison](spatial_repeat_analysis.json)
tested eight checked M variants on three of these shapes. The default tile
won every round for 256×256×1024 and 1024×1024×4096 and four of five rounds
for 512×512×1024; `baseM=96` won the remaining 512 round by 0.038 µs.
The reduced objective also selected the default among these alternatives.
Those measurements support the absence of an observed improvement in this
small M-only set, but they are not a superiority test of a distinct
optimizer-selected arm.

## Full solver audit and why it was not timed

The [full solver audit source](full_solver_audit.cpp) runs the actual
`SolveReducedMip` API on the same 12 shapes with the exploratory probe
coefficients. [Its output](full_solver_audit.csv) proposes BASE plans with
two occupied AICs on all 12 shapes and changes M/N tile extents; the source
default launched 20 blocks. The solver optimizes a global-packet service
equation, and measured coefficients inserted into that equation predict
35, 44, and 485 µs for source kernels observed at roughly 8.5, 10.0, and
45.1 µs. Therefore the solver certificate is for a demonstrably inaccurate
surrogate, and its selected tuple is not an executable source tiling record.
Projecting one field of that tuple into the source would test a different
method. The existing isolated override has only passed correctness for
downward `baseM=singleCoreM` changes. It cannot safely apply the full
M/N/K/core tuple or establish the required 95% performance claim.

## Required experiment for a full-method claim

First implement a source-backed plan finalizer that accepts the complete
solver tuple, recalculates dependent buffer depths and L2 fields, checks
local storage and tiling key, and rejects any plan that fails the full output
check. Repair the reduced BASE service term around per-core waves and
overlap, then choose a profile without using the validation timings. Freeze
an independent workload set and both arm choices before measuring. Run each
pair in randomized order across independent rounds, recording exact emitted
tiling and profiler kernel times. Analyze shape-level paired log speedups
with a 95% bootstrap confidence interval; do not count repeated kernels of
one shape as independent workloads. Report the fraction of shapes improved,
geometric-mean speedup, 95% interval, and worst regression. Claim practical
superiority only if the lower bound clears the declared 5% threshold.
