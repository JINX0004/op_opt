# Integrating timeline optimization into `mat_mul_v3`

This plan targets the existing Ascend 910B3 host tiler and kernel bodies in
[`mat_mul_v3`](../mat_mul_v3/README.md). It turns implementation selection
from a first-success rule into a minimum-predicted-runtime decision over
**legal, already executable plans**. The analytical objective and event
equations are in [README section 10](README.md); the source-level family
inventory is in [mat_mul_v3_plan_inventory.md](mat_mul_v3_plan_inventory.md).
The optimizer uses no fitted whole-kernel correction.

## 1. Current path and insertion point

The 910B3 path enters the legacy tiler through
[`MatmulV3TilingFunc`](../mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.cpp).
[`DoOpTiling`](../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp)
first accepts a knowledge-bank hit and returns; otherwise it initializes a
basic plan. `DoBasicTiling` calls `DoSelectTiling` (lines 669–717), where the
first method returning true wins (lines 1406–1432). The selected mutable
`args_`, `runInfo_`, and `tilingEnable_` are subsequently converted into
`MatmulTilingData`, a key, workspace size, and launch block dimension by
`DoLibApiTiling`, `GetWorkspaceSize`, and `PostTiling` (lines 2755–2941).

The kernel key has independent axes for full-load mode, split-core mode,
output optimization, ND-to-NZ handling, special optimization, and FP32 ADDMM
([key definition](../mat_mul_v3/op_kernel/mat_mul_v3_tiling_key.h),
[key creation](../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp)).
The legacy source has eight built-in `(LOADMODE,SPLITCOREMODE)` pairs and one
bank-only atomic pair; dtype/format compilation further restricts these
([kernel dispatch](../mat_mul_v3/op_kernel/mat_mul_v3.cpp)). L2 blocking,
buffer depths, core geometry, and loop order produce multiple complete plans
under one pair. Do not use a key pair alone as the plan identity.

The first integration point is a candidate-selection function **between the
common basic seed and final commit on the auto-tiling path**. Keep the
existing first-success path as the fallback and as one candidate. During
the initial rollout, preserve knowledge-bank hits as an early return. To
optimize those hits later, materialize the bank result as an independently
validated candidate and compare it only against candidates with the same
numerical and workspace contract. The new code must construct and score
candidates without writing to `context_`; only the winner runs the existing
serialization path once.

## 2. Optimization problem tied to actual fields

For workload $X$, queried platform profile $H$, and available workspace
budget $W_{\max}$, define a complete candidate

$$
x=(q,t,l,c,o),
$$

where $q=(\mathrm{LOADMODE},\mathrm{SPLITCOREMODE},\mathrm{FIXOPTI},
\mathrm{MIXND2NZ},\mathrm{SPECIALOPT},\mathrm{FP32ADDMM})$ is a **compiled**
key combination (the current legacy key builder fixes `FP32ADDMM` to false);
$t$ is the complete `TCubeTiling` (including `usedCoreNum`,
`singleCoreM/N/K`, `baseM/N/K`, `stepM/N/Ka/Kb`, `depthA1/B1`,
`dbL0A/B/C`, and `iterateOrder`); $l$ is `L2cacheTilePara` and cache policy;
$c$ is the conversion/output path; and $o$ fixes the executable reduction,
loop, and synchronization order. These are the fields actually consumed by
the [kernel](../mat_mul_v3/op_kernel/mat_mul_v3.cpp) and its included bodies.

The declared domain is

$$
\mathcal P_{v3}(X,H)=\{x:\;
\operatorname{compiled}(q,X,H)\land
\operatorname{covers}(x,X)\land
\operatorname{numeric}(x,X)\land
\operatorname{memory}(x,H,W_{\max})\land
\operatorname{emittable}(x)\}.
$$

The target is

$$
x^*\in\arg\min_{x\in\mathcal P_{v3}(X,H)}
T_{\mathrm{device}}(X,x;H),\qquad
T_{\mathrm{device}}=\max_{e\in V_{\mathrm{terminal}}(x)} f_e.
$$

The event graph must include launch/entry, any ND-to-NZ preprocessing, GM/L2
and L1/L0 transfers, Cube issue/completion, output Fixpipe/vector conversion,
split-K workspace traffic and merge, and final output readiness. These stages
overlap when the emitted kernel permits it; their isolated times are not
simply summed. For a first complete-plan domain $\mathcal C(X,H)\subseteq
\mathcal P_{v3}(X,H)$, evaluating every candidate gives the exact minimum of
$\mathcal C$, not of every tiling theoretically expressible by the keys.

Query `aicNum`, `aivNum`, L1/L0/UB/L2 capacities, and feature flags from
[`MatmulV3CompileInfo`](../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h)
as the tiler already does; the 910B3 values in the mathematical README are
reference ceilings, not replacements for runtime platform information.
The final block dimension must match the chosen `usedCoreNum`.

Treat correctness limits as hard constraints: compiled key/dtype/format,
transpose and bias support, alignment/tails, task coverage, accumulator and
reduction semantics, local storage, workspace, synchronization, and any
atomic one-round restriction. Shape cutoffs such as `K >= 27392`,
`singleCoreN < 512`, and `avgRatio < 0.7` currently help choose a fast
template; they are not automatically correctness constraints. Remove one
from candidate feasibility only after its kernel and resource requirements
are independently checked. In particular, a deterministic request must not
select the atomic plan.

## 3. Candidate representation and source refactor

Add a candidate value type, for example `MatmulV3Plan`, holding the full
`MatmulTilingData`, `TilingEnable`, relevant args-derived conversion flags,
workspace bytes, launch block dimension, and a stable plan identifier.
Keep workload inputs and `MatmulV3CompileInfo` immutable during candidate
generation. The current `MatmulV3RunInfo` omits some `TCubeTiling` fields,
including `dbL0A/B`; use the finalized tiling data as the canonical plan.

Extract the common initialization before `DoSelectTiling` into a seed builder.
Make each family builder accept a copy of this seed and return either one or
more complete candidates or an explicit rejection reason. Do not call the
existing `DoBL1...`, split-K, and L2 routines successively on the same object:
they mutate `args_`, `runInfo_`, `tilingEnable_`, `enableCache_`, L2 fields,
and sometimes `tilingData_` even when a branch later fails. Preserve the
legacy method order behind a wrapper so disabling optimization produces the
same key, tiling data, workspace, and block dimension as before.

Move post-selection work into a **candidate-local finalizer**: ND-to-NZ
choices, bias adjustments, basic step-K optimization where applicable,
library tiling fields, L2 cache flags, key construction, and workspace
calculation. Existing `DoLibApiTiling` and `GetWorkspaceSize` should commit
or serialize the winning plan, not recompute a different one. Some fields
are populated by the matmul library rather than `MatmulV3RunInfo`; the
finalizer must include that library step for each candidate before scoring.
Guard all integer products and workspace arithmetic against overflow.

Add a `ValidatePlan` routine separate from ranking. It should verify the
compiled dispatch tuple, family-specific support rules, all positive tile
fields, legal core count and launch mode, exact coverage of tails/K slices,
workspace, per-core memory occupancy, output layout, and numerical mode.
The current `CheckMMTilingDataIsVaild` checks many fields for zero but is not
by itself a complete legality proof. Reuse existing checks where they are
correct, and add assertions that the selected kernel can consume exactly
the serialized data.

## 4. Timeline evaluator and optimization driver

Implement a standalone **host-side C++** estimator linked into the tiler.
It takes an immutable workload/profile and a finalized `MatmulV3Plan`; it
does not call the Python calibrated model and does not alter the kernel.
Build events from the **actual loop and mapping formulas** in the selected
kernel body:

| Family | Required additional timeline work |
|---|---|
| BASE | L2 traversal and per-core M/N rounds; repeated K loads hidden inside `MatmulImpl::Iterate`; final C transfer |
| AL1/BL1 full load | Initial operand fill, L1 residency and capacity, reused operand traffic, output-path variants |
| Single-core split-K (MKN/NKM/GM-to-L1/AL1) | Explicit K loops, FP32 intermediate/workspace where used, AIC/AIV synchronization and output cast |
| Deterministic multi-core split-K | K-slice assignment, partial writes to workspace, synchronization, AIV UB reduction and cast |
| Atomic multi-core split-K | Output initialization, cross-core barrier, and atomic accumulation; bank-only legal gate |

The base loop is in
[`mat_mul_base_kernel.h`](../mat_mul_v3/op_kernel/mat_mul_base_kernel.h),
single-core split-K in
[`mat_mul_sc_splitk_kernel.h`](../mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel.h),
deterministic reduction in
[`mat_mul_deterministic_splitk_kernel.h`](../mat_mul_v3/op_kernel/mat_mul_deterministic_splitk_kernel.h),
and atomic accumulation in
[`mat_mul_multi_core_splitk_kernel.h`](../mat_mul_v3/op_kernel/mat_mul_multi_core_splitk_kernel.h).
Use the 910B3 primitive timing profile from independent transfer,
instruction, synchronization, and conversion measurements. Keep its units,
source, and version alongside the estimator; do not substitute an unexplained
per-shape multiplier.

Return predicted device time plus a breakdown: critical path, bytes per
memory level, core occupancy, workspace, Cube/transfer/reduction waits, and
plan identifier. Validate the event graph against small hand-calculated
cases from [README sections 11–12](README.md) before using its rankings.
For large M/N, use only a compression or periodic recurrence shown to
preserve engine availability, buffer lifetimes, and shared-service state.

Start with exhaustive evaluation of the finite candidates generated by
existing recipes. Seed the incumbent with the legacy plan. For later
independent choices of `singleCoreM/N/K`, `baseM/N/K`, step/depth, L2 tile,
core count, and loop order, use branch-and-bound. A partial-plan lower bound
must be optimistic for **all** its legal completions, for example the maximum
of compulsory useful compute time, compulsory HBM service, and an unavoidable
precedence chain. Visit promising regions first, but prune only when a valid
bound meets or exceeds the incumbent. On a time or candidate budget, return
the best validated plan and its remaining lower bound; if no optimizer plan
is valid, return the legacy plan.

Because selection runs on the host, record optimization time separately.
The primary objective above is device latency. For dynamic, one-shot shapes,
also track $T_{\mathrm{host\ tiling}}+T_{\mathrm{device}}$; for reused shapes,
evaluate amortization under the runtime's actual tiling cache behavior.
Set an explicit host-time/memory budget before enabling broad search online.

## 5. Phased implementation and acceptance gates

| Phase | Code changes | Completion evidence |
|---|---|---|
| 0. Baseline | Add observability for workload signature, source (bank/auto), full key, tiling fields, workspace, block dimension, and measured kernel time | Reproduce current outputs and collect representative 910B3 cases: small M/N long K, large M/N, tails, transpose, bias, FP16/BF16/FP32 |
| 1. Refactor | Add candidate/seed/finalizer types and retain the legacy selector | With optimization disabled, byte-identical tiling data, key, workspace, and block dimension across existing host test cases |
| 2. Legal domain | Generate BASE, ordinary single-core split-K, and deterministic split-K candidates first; then add full-load and split-K subvariants; import bank hits only after candidate-local validation exists, with bank-only atomic behind its exact gate | Every emitted candidate maps to a compiled kernel, passes hard legality, and passes kernel numerical tests |
| 3. Analytical score | Add event builders and 910B3 primitive profile; expose prediction breakdown | Hand-timeline regression checks pass; predicted rankings are compared with device measurements by family and shape regime |
| 4. Selection | Enumerate candidates, choose minimum predicted runtime, retain legacy fallback and deterministic tie-breaking | Selection is deterministic; no correctness failures; host overhead remains within the declared budget |
| 5. Wider tiles | Add explicit domains for independent block/L1/L0/buffer/L2 decisions and branch-and-bound | Exhaustive small-domain optimum matches branch-and-bound; incomplete runs report incumbent and gap |
| 6. Rollout | Enable by scope after A/B measurement on 910B3 | Track median and tail latency, regressions, candidate count, host tiling cost, fallback rate, and model ranking accuracy |

Use [`test_matmul_v3_tiling.cpp`](../mat_mul_v3/tests/ut/op_host/test_matmul_v3_tiling.cpp)
for host selection/serialization tests and
[`test_mat_mul_v3.cpp`](../mat_mul_v3/tests/ut/op_kernel/test_mat_mul_v3.cpp)
for representative kernel correctness. Add tests that deliberately cross
family boundaries, including no candidate, repository hit, deterministic
mode, unaligned output, workspace limit, and integer overflow. Extend the
[benchmark experiment](../mat_mul_v3/experiments/run_mm_bench.py) to capture
both key and latency on 20-AIC 910B3; its existing documented datasets were
chosen for 24-AIC A2/A3 and are not sufficient by themselves.

Suggested new host files are `matmul_v3_plan.h/.cpp` (candidate and
validation), `matmul_v3_timeline.h/.cpp` (events and estimator), and
`matmul_v3_optimizer.h/.cpp` (enumeration/bounds). Integrate them from
`matmul_v3_base_tiling.cpp` and add them to the host build if source
discovery does not pick them up. No kernel-body change is required for the
initial selection-only milestone.

The first shippable milestone is deliberately narrow: on 910B3 and a declared
FP16/BF16 ND contract, compare **the existing legal BASE, single-core split-K,
and deterministic split-K plans** with the analytical model, then commit the
best one or fall back to the original result. Expanding independent tiling
variables comes after the candidate-local finalizer and timeline scorer have
been validated on hardware. This avoids presenting the minimum of an
incompletely modeled candidate set as a global hardware optimum.

The [standalone driver](standalone_optimizer/README.md) implements finite
ranking, a two-level implementation/tiling solver, and fixed-resource
event-graph scoring without CANN. It consumes caller-vetted candidates and
derives each event's duration from candidate work amounts and shared hardware
primitive parameters. The inner solver can enumerate or use a caller-proven
optimistic bound for branch-and-bound.
The first [typed standalone preflight](standalone_optimizer/plan_finalizer.h)
now derives a key tuple, workspace, and launch fields for aligned BASE and
deterministic Split-K proposals. It checks the modeled memory footprint and
handles the A ND→NZ conversion needed by `[16,16,102400]`, including its
mixed key and extra workspace. This is not the CANN
candidate-local finalizer: host-generated `TCubeTiling` fields, the packed
compiled key, and device legality remain to be checked through CANN. Measured
primitive parameters are also still needed; the example manifest is not a
910B3 performance prediction.
The first [shape-derived event builder](standalone_optimizer/event_builder.h)
now covers a narrow BASE/deterministic Split-K ND, NN, FP16/BF16 contract.
It expands M/N/K tails and tile fields into logical transfers, Cube work,
partials, and reduction events. It does not yet model `MatmulImpl` buffering,
L1/L2 reuse, full L2 traversal, or every workspace layout. Compare rankings
against device measurements before enabling selection.

The optional [reduced MILP solver](standalone_optimizer/mip_solver.h) is the
first implementation that optimizes tiling variables with a mathematical
program rather than evaluating every complete assignment. It solves separate
BASE and deterministic Split-K models over a declared M/N block grid, then
scores the two selected plans with the event graph when expansion fits the
event cap. Its HiGHS optimality
result applies to the reduced compute/GM service formula and declared M/N
block grid only. To make this the `mat_mul_v3` selector, extend the legal
domain to the remaining compiled families, encode the
source's buffer and resource schedule, use measured 910B3 primitives, and
pass each selected plan through the actual CANN candidate-local finalizer.
