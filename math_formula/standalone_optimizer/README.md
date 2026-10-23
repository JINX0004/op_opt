# Standalone finite-plan optimization driver

This C++17 program is the first, CANN-independent implementation of the
[mat_mul_v3 integration plan](../MAT_MUL_V3_IMPLEMENTATION_PLAN.md). It
selects the minimum analytical timeline among supplied complete candidates.
It is exact for that finite candidate list. The typed `FinalizePlan` preflight
now validates a narrow 910B3 plan subset and derives its key tuple, launch
fields, and workspace. An optional HiGHS-backed MILP solves a reduced
block-tiling model for BASE and deterministic Split-K. It does not serialize `MatmulTilingData`, invoke CANN's
tiling finalizer, measure 910B3 primitives, or predict every supported shape.
The legacy `verified` marker remains an assertion by the manifest producer.

Build and run without CANN:

```bash
cmake -S math_formula/standalone_optimizer -B /tmp/mmopt-build
cmake --build /tmp/mmopt-build
ctest --test-dir /tmp/mmopt-build --output-on-failure
/tmp/mmopt-build/mmopt_cli math_formula/standalone_optimizer/example.manifest
/tmp/mmopt-build/mmopt_nested_example
```

The manifest has one `workload M N K dtype` record, one
`hardware aic_count workspace_limit_bytes` record, and shared hardware
`primitive name unit fixed_latency_us rate_per_us` records. Each candidate starts with
`candidate id family preflight` for source-backed preflight, or the legacy
`candidate id family key verified|unverified workspace_bytes`, followed by
`tile field value` records and either `generate_events` or explicit
`event id resource primitive work_units unit dependencies` records, then
`end`. Dependencies are comma-separated event IDs or `-`.
The required tile fields are `usedCoreNum`, `singleCoreM/N/K`, and
`baseM/N/K`. Preflight candidates require `generate_events`, derive their key
and workspace, and reject any additional tile field. Legacy candidates with
explicit event records can carry additional fields. Candidate IDs and event IDs
must be unique. Work amounts must be
nonnegative and finite, and each event unit must match its primitive.

The current primitive rule is
$d_e=L_{p(e)}+w_e/R_{p(e)}$, where $L$ is fixed latency in microseconds,
$R$ is work units per microsecond, and $w_e$ is the event's work amount.
Primitive parameters live in one hardware profile and are shared by every
candidate. The tiling and implementation determine **which** events occur,
their byte/FLOP/count amounts, resource assignments, and dependencies.
For real 910B3 use, a profile needs measured path-specific MTE, Cube,
conversion, synchronization, and reduction parameters. Shared bandwidth
contention and issue/completion overlap may require richer service rules;
the current exclusive-resource approximation cannot model every hardware
interaction.

Each event is a nonpreemptive interval. Explicit dependencies impose
precedence. Events on the same named resource also run serially in manifest
order; different resources can overlap. The evaluator topologically computes
the earliest start of each event and the latest completion, reports a
critical path, and rejects cycles or missing dependencies. This models a
**fixed resource schedule**. A producer must list the real kernel's resource
order and include launch, conversion, memory transfers, Cube work, workspace,
merge, and output events as applicable. The example primitive rates are
illustrative and must not be used to claim a 910B3 speedup.

`Optimize` in [optimizer.h](optimizer.h) is the reusable API. It rejects
unmarked candidates, zero core/tile dimensions, AIC and workspace-limit
violations, malformed event graphs, and duplicate candidate IDs. It ranks
valid candidates by predicted microseconds with stable ties. Its legacy
`backend_verified` flag means that the producer marked a plan as suitable for
the scorer; it is **not** proof that CANN accepted the plan. A future CANN
integration should populate this API from full `MatmulTilingData` and replace
that marker with actual backend verification. A measured
primitive timing profile will replace the illustrative example values; no fitted
whole-kernel correction is used here.

## Shape-derived event builder

`BuildMatmulEvents` in [event_builder.h](event_builder.h) takes the workload
and a finalized plan and fills its events. Its current declared contract is
contiguous ND, nontransposed A/B, no bias, FP16 or BF16 input/output, and
either `BASE` or `DETERMINISTIC_SPLIT_K`. The caller must enforce the
compiled-key rules. The builder rejects unsupported layout, transpose, bias,
family, and dtype combinations.
Only `usedCoreNum`, `singleCoreM/N/K`, and `baseM/N/K` are modeled tile
fields; the builder rejects additional tiling fields rather than silently
ignoring their timeline effects.

For each output tile, it computes the actual M/N tail extents. Each
base-M/base-N/base-K packet contributes $D_A=2m_tk_t$ bytes,
$D_B=2k_tn_t$ bytes, and
$F_{\mathrm{Cube}}=2\operatorname{align}_{16}(m_t)
\operatorname{align}_{16}(n_t)\operatorname{align}_{16}(k_t)$ issued FLOPs.
It counts each packet read independently, without assuming L1/L2 reuse.
`BASE` assigns output tiles round-robin to AICs, runs their full K locally,
and writes one FP16/BF16 output tile. Deterministic Split-K assigns each AIC
contiguous `singleCoreK` slices, models FP32 partial updates, reads all
partials for vector reduction, and writes the final output. It requires at
least $4CMN$ workspace bytes for $C$ AICs; actual kernel padding/layout may
require more.

For plans marked `nd2nz_a`, the builder adds an aggregate A-matrix read,
vector ND→NZ conversion, workspace write, and synchronization before Cube
work. Its `nd2nz` primitive uses elements as work units. Later A packet reads
represent accesses to the converted workspace. This is a logical service
model; actual conversion is tiled across AIVs. B conversion is still outside
the builder's scope and is rejected.

The [nested example](nested_example.cpp) now generates its event work from
`(M,N,K)` and candidate tiles after typed plan preflight. Its primitive rates
remain illustrative. The fixed HBM service order, omitted cache reuse,
simplified Cube issue/completion, and simplified partial-update path are
model assumptions, not a validated `MatmulImpl` trace. Explicit graphs are
capped at 200,000 events; larger workloads need validated compression.
The checked-in [example manifest](example.manifest) also uses
`generate_events`, so it no longer contains per-candidate work constants.

## Two-level solver API

`OptimizeImplementations` implements
$\min_{a\in\mathcal A}\min_{z\in\mathcal D_a}T(a,z)$.
Each `Implementation` declares its tiling variables and finite value domains,
a `build(assignment)` callback that creates a finalized candidate or returns
`nullopt` for an infeasible tile, and an optional `lower_bound_us(partial)`
callback. The driver enumerates implementation families; its inner solver
searches the family-specific tiling domain, validates and scores complete
plans, then returns each family's best complete candidate and the overall
winner, including the full tile record supplied by the builder.

With no lower bound, the inner solver evaluates every complete assignment.
With a **proven optimistic** lower bound, it uses branch-and-bound and prunes
a subtree only when that bound cannot beat the incumbent for the same
implementation. It has no time cap and therefore returns the exact minimum
of the declared finite, emittable domain when the builder is complete and
bounds are valid. An invalid bound can discard the true winner. The
[nested example](nested_example.cpp) shows the API and prints both per-family
and overall winners, but its hardware rates are illustrative. The manifest CLI
still accepts fully enumerated candidates;
the two-level solver is currently a C++ API until real `mat_mul_v3` family
builders and legality checks are connected.

## Reduced MILP with M/N block tiling

[`SolveReducedMip`](mip_solver.h) solves two integer programs, one for BASE
and one for deterministic Split-K, using the HiGHS C API. The C++ optimizer
does not construct and score every complete tiling in these programs. It
constructs a finite set of **spatial geometries** inside each MILP, then
optimizes K tiling and AIC count jointly with the geometry choice. HiGHS
may use branch-and-bound internally. The result reports each family's
optimal **reduced-model** objective, the selected tiling, and its separate
event-graph score when explicit expansion fits its 200,000-event cap. The
winner is chosen by the reduced objective. A solver
certificate does not establish optimality for the full event-graph timeline
in [section 10](../README.md#10-full-optimization-problem), nor for measured
910B3 execution time.

`singleCoreM`, `singleCoreN`, `baseM`, and `baseN` are independent choices,
with `baseM<=singleCoreM` and `baseN<=singleCoreN`. BASE owns all K and
chooses `baseK=16b` and an AIC count no larger than its output-tile count.
Deterministic Split-K fixes `baseK=128` and chooses `stepK=s`,
`singleCoreK=128s`, and occupied AIC count `c`. All choices must satisfy
the typed preflight's alignment, local-memory, workspace, and compiled-key
subset. The default M/N grid is the aligned values 16, 32, 64, 128, and 256
that fit the shape, plus the full extent when it is at most 256. A caller can
pass `MipTileDomain` to specify other aligned values; the optimizer proves
the optimum only over that declared grid, capped at 4096 spatial geometries.
The model still excludes alternative buffer policies and other `mat_mul_v3`
families.

Let $C_M,C_N$ be the single-core output extents and $B_M,B_N$ the base
extents. Define $N_o=\lceil M/C_M\rceil\lceil N/C_N\rceil$ output tiles.
Let $N_M$ be the sum of base-M packet counts over all output-M tiles,
including the tail, and define $N_N$ analogously. Then $S=N_MN_N$ is the
number of spatial base packets per K packet, and the no-cache operand
traffic is $D=2K(MN_N+NN_M)$ bytes. These counts match the current event
builder's output and base tile loops. Let $O=MN$, $F=2MNK$, $R_C$ be per-AIC Cube
FLOP/µs, $R_R,R_W$ GM byte/µs, and $\lambda_R,\lambda_W$ fixed
GM read/write costs. $L$ contains launch and, when required, A ND→NZ
read, conversion, workspace write, and synchronization. With
$P_B=\lceil K/(16b)\rceil$, the BASE objective is

$$
T_B=L+\max\left(\frac{F}{cR_C},
 \frac{D}{R_R}+\frac{2O}{R_W}
 +2SP_B\lambda_R+N_o\lambda_W\right),\qquad c\le N_o.
$$

For deterministic Split-K, let $P=\lceil K/(128s)\rceil$ be the K-slice
count and $G=\lceil K/128\rceil$ the base-packet count. With vector rate
$R_V$ and fixed vector/sync costs $\lambda_V,\lambda_S$, the objective is

$$
\begin{aligned}
T_D=L+\max\Bigg(&\frac{F}{cR_C},
 \frac{D+4PO}{R_R}+\frac{(4P+2)O}{R_W}\\
 &+\bigl(2SG+N_o(P-c+1)\bigr)\lambda_R
 +N_o(P+1)\lambda_W\Bigg)\\
 &+N_o(\lambda_S+\lambda_V)+\frac{(c-1)O}{R_V}.
\end{aligned}
$$

The programs use one-hot geometry variables, integer variables for $b$,
$s$, $c$, and packet counts, plus a continuous variable for the `max` stage.
Geometry-dependent traffic and workspace coefficients are part of the MILP,
not chosen after solving. Tangent inequalities
for $1/x$ at every feasible integer value make the reciprocal epigraph
exact **at integer points**; integer packet counts implement the ceilings.
One-hot core-count constraints enforce $c\le\lceil K/(128s)\rceil$
against the **true** slice count, rather than allowing the packet-count
variable to grow artificially. The solver therefore proves the
minimum of these two stated formulas over this restricted feasible domain.
The `max` stage is a service-demand approximation: it omits the detailed
precedence, buffering, shared-resource contention, and issue timing of the
event graph. The current formulation requires zero fixed Cube latency.
For large shapes, the selected plan and surrogate remain available even when
the explicit event graph exceeds its cap; `event_graph_score` is then absent.

HiGHS is optional. Install its development package or pass the package
prefix to CMake, then run:

```bash
cmake -S math_formula/standalone_optimizer -B /tmp/mmopt-mip-build \
  -DMMOPT_HIGHS_ROOT=/path/to/highs-prefix
cmake --build /tmp/mmopt-mip-build
ctest --test-dir /tmp/mmopt-mip-build --output-on-failure
/tmp/mmopt-mip-build/mmopt_mip_example
```

Without HiGHS, the existing event scorer and two-level search API still
build. The new test compares each MILP optimum with a small exhaustive
oracle; that enumeration is confined to the test. The example's primitive
rates are hypothetical. Before using the result for device selection, the
next steps are to measure the primitive profile, expand the MILP's legal
domains and event/schedule encoding, finalize selected plans with CANN, and
compare predicted rankings with device measurements.

## Typed 910B3 plan preflight

[`FinalizePlan`](plan_finalizer.h) accepts a typed `PlanDecision`, a workload,
the allocated AIC count and workspace budget, and per-core capacities. The
defaults for L1/L0A/L0B/L0C are the 910B3 reference capacities from
[section 2.3](../README.md#23-ascend-910b3-hardware-profile); pass queried
values for the actual device. Initial scope is FP16/BF16, ND/ND/ND, NN,
no bias, M/N/K and tile extents divisible by 16, K 256-byte aligned, and N
either 256-byte aligned or on the host tiler's direct-transfer list. The
reference feature profile assumes `supportL0c2out=true` and
`supportL12BtBf16=false`; callers can supply queried feature values.
For `[16,16,102400]`, `GetMoreArgs()` requests ND→NZ conversion of A because
K exceeds 65,535, while N=16 is a direct-transfer size for B. The preflight
selects MIXND2NZ=0 and reserves the converted A matrix in workspace.
It supports BASE `(0,0,0,mix,0,0)` and deterministic Split-K
`(0,3,0,mix,0,0)` key **tuples** from `mat_mul_v3_tiling_key.h`, where
`mix=0` for head A conversion and `mix=1` for no conversion. The packed
`GET_TPL_TILING_KEY` value depends on CANN headers and is not synthesized.

BASE must own the whole K dimension and cannot assign more AICs than output
tiles. Deterministic Split-K requires `singleCoreK = baseK * stepK`, at least
two occupied K partitions, and sufficient FP32 partial workspace. Both
branches check a modeled local-memory footprint: two L0A/L0B operand
tiles, two FP32 L0C tiles, and two K-step windows for each L1 operand.
Overflow, insufficient local memory, too many AICs, or an insufficient
workspace budget reject the plan. The derived workspace is 20 MiB for BASE
and `20 MiB + usedCoreNum * singleCoreM * singleCoreN * 2 * 4` bytes for
Split-K, plus `align16(M) * align16(K) * 2` bytes when A is converted,
following `GetWorkspaceSize()` in `matmul_v3_base_tiling.cpp`. Block dimension
is `usedCoreNum` and schedule mode is 1. On `[16,16,102400]`, the A-conversion
addition is 3,276,800 bytes.

This is a standalone preflight, not a full `mat_mul_v3` host tiler port. The
actual host may choose a different tiling, depth, cache split, conversion
mode, or key; the CANN tiling API must still finalize and serialize a proposed
plan before the kernel can run. The current event model represents only seven
tile dimensions and is approximate even for plans that pass preflight.
