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
partials for vector reduction on up to two AIVs per occupied AIC, and writes
the final output. It requires at
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

For each branch's objective, decision variables, constraints, and numerical
example, see [`SOLVE_BASE_FORMULATION.md`](SOLVE_BASE_FORMULATION.md) and
[`SOLVE_SPLIT_FORMULATION.md`](SOLVE_SPLIT_FORMULATION.md). The Split-K document
also compares the standalone domain with the current `mat_mul_v3` host tiler.
For device experiments, use the
[`SolveBase` 910B measurement plan](SOLVE_BASE_910B_MEASUREMENT_PLAN.md)
to collect primitive timings and validate the reduced objective. The
[installed-CANN harness](bench910/README.md) has been compiled and run on an
Ascend910B3; its [first pilot](bench910/PILOT_910B3_CANN9.md) records emitted
BASE tilings and per-kernel times. A
[source-matched follow-up](bench910/SOURCE_MATCHED_910B3_CANN9.md) uses the
user's remote `ops-nn-sribd` MatMulV3 and exercises its core-count control.
These whole-kernel observations are not yet a fitted primitive profile.
The subsequent [910B3 primitive and controlled-tiling pilot](bench910/PRIMITIVE_910B3_CANN9.md)
measures Cube, ND→L1 NZ, and Fixpipe probe slopes and exposed the old
serial global-packet objective's 2-AIC bias. BASE now charges packet
startup to the busiest assigned AIC; its measured coefficients remain
exploratory and require held-out runtime validation.
The [large-shape optimizer-versus-default experiment](bench910/LARGE_SHAPE_OPTIMIZER_VS_DEFAULT_910B3.md)
finds that the source-executable M-only restriction selects the default tile
on seven larger workloads. A checked `baseM=64` alternative is significantly
slower than the default `baseM=128` on three tested large shapes. This does
not validate a speedup for the unexecuted full multi-field MILP choice. A
[revised full-solver audit](bench910/large_wave_solver_audit.csv) with
per-AIC packet startup selects 20 AICs on all seven large shapes; its
unrestricted full tile tuples still need source-backed execution and runtime
validation. The
[source-compatible BASE evaluation](bench910/FULL_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
filters out outer M/N tiles that the tested plain BASE kernel did not compute
correctly and admits the source's one-buffer L0C tile. Its model-tied 256×128
orientation is 1.197× slower than the 128×256 default across the seven
checked large shapes.

[`SolveReducedMip`](mip_solver.h) solves two small integer programs, one for
BASE and one for deterministic Split-K, using the HiGHS C API. It constructs
a finite set of **spatial geometries**. For each BASE geometry, it eliminates
the K-tiling decision analytically. For each Split-K geometry, it evaluates
the bounded feasible K-step and occupied-AIC pairs. A
one-hot master program then selects the geometry. The result reports each family's
optimal **reduced-model** objective, the selected tiling, and its separate
event-graph score when explicit expansion fits its 200,000-event cap and
whole-output Split-K partial-workspace representation. The
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
subset. The default **base** M/N grid is the aligned values 16, 32, 64,
128, and 256 that fit the shape, plus the full extent when it is at most
256. The default **single-core** grid also
includes 512, 1024, and 2048 when they fit, the full axis, and aligned
$\lceil\text{shape}/p\rceil$ for
$p\in\{1,2,4,5,10,A\}$ when $p\le A$, where $A$ is the allocated AIC
count. For $K>65535$, the Split-K grid also admits the source 384-row
or 384-column tile when the corresponding shape axis is 128–383.
BASE discards oversized core tiles. This follows the full-axis and
core-partition patterns in
[`matmul_v3_base_tiling.cpp`](../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp),
so a single-core tile can span many base tiles. A caller can pass
`MipTileDomain` to specify other aligned values; the optimizer proves the
optimum only over that declared grid, capped at 4096 spatial geometries.
The model still excludes other `mat_mul_v3` families and most buffer-policy
choices. For the tested plain BASE key `0x10000`, full-output checks showed
that larger `singleCoreM/N` values left output regions zero. Source-backed
studies therefore set `MipTileDomain::base_core_equals_base=true`, which
filters BASE geometries to `singleCoreM=baseM` and `singleCoreN=baseN`.
This restriction does not apply to every MatMulV3 BASE implementation:
the [B-in-L1 full-load path](bench910/FULL_LOAD_CORE_TILING_SUPPORT_910B3.md)
uses larger core tiles on validated small-K cases and must be modeled as a
separate family.
The source default's 128×256 tile uses one FP32 L0C buffer, so those studies
also set `CoreMemory::base_l0c_buffers=1`. Standalone defaults retain the
wider analytical domain and two-buffer preflight.

BASE can use multiple Cube cores: its occupied count is bounded by the
number of spatial output tiles and the allocated AICs, rather than by one.
For ND input with $K>65535$, the host tiler's
[`CalcNd2nzWorkspace`](../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp)
reserves a converted copy of the full A matrix before BASE runs. This
$2MK$-byte allocation is independent of `singleCoreM/N` and the number of
Cube cores. For example, FP16 $(M,N,K)=(64000,48000,256000)$ needs
32,768,000,000 bytes for A conversion plus the 20 MiB RPC reservation.
A 32 MiB workspace budget rejects that workload before either family is
solved; with sufficient workspace, BASE can occupy multiple AICs.

Let $C_M,C_N$ be the single-core output extents and $B_M,B_N$ the base
extents. Define $N_o=\lceil M/C_M\rceil\lceil N/C_N\rceil$ output tiles.
Let $N_M$ be the sum of base-M packet counts over all output-M tiles,
including the tail:

$$
N_M=\left\lfloor\frac{M}{C_M}\right\rfloor
    \left\lceil\frac{C_M}{B_M}\right\rceil
  +\left\lceil\frac{M\bmod C_M}{B_M}\right\rceil,
$$

and define $N_N$ analogously. Then $S=N_MN_N$ is the
number of spatial base packets per K packet. Separate no-cache A and B
read requests are $D_A=2KMN_N$ and $D_B=2KNN_M$ bytes, with
$D=D_A+D_B$. These counts match the current event builder's output and
base tile loops; they are not physical HBM miss counts. Let $O=MN$,
$F=2MNK$, $R_C$ be per-AIC Cube FLOP/µs, $R_A,R_B,R_W$ aggregate
A-read/B-read/GM-write byte/µs, and $\lambda_A,\lambda_B,\lambda_W$
their fixed operation costs. $L$ contains launch and, when required, A
ND→NZ read, conversion, workspace write, and synchronization. With
$P_B=\lceil K/(16b)\rceil$. The reduced BASE objective groups output
tiles into at most four full/tail classes. For each class, it gives every
core $\lfloor H_t/c\rfloor$ tiles and rotates the $H_t\bmod c$ extra tiles
to consecutive cores. Let $E_q$ be core $q$'s assigned output elements,
$S_q$ its assigned spatial base packets, and $O_q$ its assigned output-tile
count. This $O(c)$ calculation is a source-inspired load-balance
abstraction, not the exact MatMulV3 L2 scheduler. The explicit event builder
still uses its separate row-major assignment. The BASE objective is

$$
T_B=L+\max\left(\max_q\frac{2K E_q}{R_C},
 \frac{D_A}{R_A}+\frac{D_B}{R_B}+\frac{2O}{R_W}
 +\max_q\bigl(S_qP_B(\lambda_A+\lambda_B)+O_q\lambda_W\bigr)\right),
 \qquad c\le N_o.
$$

Supply `gm_read_a` and `gm_read_b` together in `Hardware::primitives` to
set the optional directional BASE read services. If both are absent, each
inherits `gm_read`, reproducing the original symmetric objective. A single
directional entry is rejected. A later
[operand-shaped 910B3 probe](bench910/directional_read/README.md) measures
different effective A/B transfer rates. It predicts the observed tile
orientation but overstates the whole-kernel gap, so no A/B-specific rates
have been accepted as the default profile. This step exposes the asymmetry
without deriving coefficients from whole-kernel MatMul timings. Split-K
and the optional event-graph score still use the generic `gm_read` service.

For the checked plain BASE branch, an optional
[source-step MTE2 packet profile](bench910/l1_step_read/README.md) models
the host tiler's `stepKa/stepKb` L1 transfers with separate A/B packet,
byte, and ND-row terms. Supply all four `base_mte2_a`, `base_mte2_b`,
`base_mte2_a_rows`, and `base_mte2_b_rows` primitives and set
`base_core_equals_base=true`. With the earlier row-major load assignment,
this path correctly ranked the checked 128×256 versus 256×128 tiles on
seven device-measured shapes (maximum individual runtime error 4.49%),
but one held-out primitive packet shape had 18.3% slope error, so it
remains opt-in. Its objective and source-step
derivation are
in [the BASE formulation](SOLVE_BASE_FORMULATION.md#optional-source-step-mte2-packet-objective).
An additional [source-backed selector rerun](bench910/PACKET_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
found that its only different choice among 12 new shapes was 15.94% slower
than the default. Those device experiments predate the balanced-class
load abstraction. In a local solver check after this change, it chose the
source's 128×256 tile for $(6144,384,8192)$, but still chose 256×128 for
$(4096,384,4096)$, $(6144,640,8192)$, and $(8192,384,16384)$.
A [fresh four-round device rerun](bench910/abstract_balance_three_v1/README.md)
found those three swapped tiles 6.1–13.9% slower than the source defaults,
with exact output on every run. Balanced class counts alone do not make
the reduced objective a reliable selector; exact source L2/tail scheduling
remains unmodeled.
An [eight-case larger-shape audit](bench910/PACKET_BASE_XLARGE_910B3.md)
also found identical optimizer and default tiles on every case; its
objective underpredicted source kernel time by 3.7–20.5%.
A [new five-shape paired study](bench910/PACKET_BASE_EXPLORATORY_MORE_SHAPES_910B3.md)
found three additional source-correct swapped-tile choices 5.1–13.0%
slower than the default, one 19-core choice effectively tied, and one
same-tile control. These exploratory results reinforce the need to fix
the source block-assignment model before automatic deployment.
An [above-L2 working-set experiment](bench910/above_l2_v1/README.md)
screened 12 shapes with 207–408 MiB of combined FP16 A+B+C data. Nine
selected the same 128×256 tile as the source; three selected 256×128 and
measured 3.8%, 4.1%, and 12.3% slower in four randomized pairs each. In
the matched 8192×384 pair, doubling K from 12288 to 24576 exposed a
larger model underprediction, consistent with missing K-dependent cost.
For the next source path, the [K>65535 Split-K experiment](bench910/k_gt_65535_v1/README.md)
found that crossing the A ND→NZ conversion boundary increased measured
deterministic Split-K time by 37–64% across three paired shapes. The
original reduced solver chose BASE in those cases. The revised model
admits the source's `singleCoreM=384` when `M=256`, and follows the
busiest AIC's assigned K slices. It now selects Split-K for the 256²
case but still selects BASE for the 512² and 1024² cases. These are
model decisions, not device-validated replacements for the source tiler.

For deterministic Split-K, let $q=K/128$, $P=\lceil q/s\rceil$ be the
K-slice count, $r=\lceil P/c\rceil$ the busiest AIC's slice count, and
$q_{\rm crit}=\min(q,sr)$ its K-packet count. For each output
tile $t$ with $O_t$ elements, the two-AIV-per-AIC approximation places at
most $\lceil O_t/(2c)\rceil$ elements on one AIV. Let
$Q_g(c)=\sum_t\lceil O_t/(2c)\rceil$. With per-AIV vector rate $R_V$ and
fixed vector/sync costs $\lambda_V,\lambda_S$, the objective is

$$
\begin{aligned}
T_D=L+\max\Bigg(&\frac{2O(128q_{\rm crit})}{R_C},
 \frac{D+4PO}{R_R}+\frac{(4P+2)O}{R_W}\\
 &+\bigl(2Sq_{\rm crit}+N_or\bigr)\lambda_R
 +N_o(r+1)\lambda_W\Bigg)\\
 &+N_o(\lambda_S+\lambda_V)+\frac{(c-1)Q_g(c)}{R_V}.
\end{aligned}
$$

### Feasibility constraints and reduced master

For every allowed spatial geometry $g=(C_M,C_N,B_M,B_N)$, a binary $y_g$
selects that geometry. Exactly one is active: $\sum_g y_g=1$. The API
rejects misaligned tile values and filters out geometries with $B_M>C_M$,
$B_N>C_N$, or fixed L0 storage violations before adding variables. Other
constraints are enforced while computing each geometry's cost, rather than
as rows of the master. Here $A$ is the
allocated AIC count, $C_{0A},C_{0B},C_{0C},C_1$ are the modeled local-memory
capacities, and
$W_0=20\,\mathrm{MiB}+W_{A\to NZ}$ is the fixed workspace. The relevant
limits are:

| Family | Local feasibility constraints |
| --- | --- |
| BASE | $1\le c\le\min(A,N_o)$; $1\le b$ and $\text{baseK}=16b\le K$; $4B_M(16b)\le C_{0A}$; $4B_N(16b)\le C_{0B}$; $4(B_M+B_N)(16b)\le C_1$; $4d_CB_MB_N\le C_{0C}$ for configured one- or two-buffer BASE L0C policy $d_C$; $W_0\le W_{\max}$ |
| Deterministic Split-K | $2\le c\le\min(A,\lceil K/(128s)\rceil)$; $1\le s\le K/128$; $\text{baseK}=128$; $\text{singleCoreK}=128s$; $4B_M(128)\le C_{0A}$; $4B_N(128)\le C_{0B}$; $4(B_M+B_N)(128s)\le C_1$; $8B_MB_N\le C_{0C}$; $W_0+8cC_MC_N\le W_{\max}$ |

For a fixed BASE geometry $g$ and core count $c$, the generic byte-only objective is
nonincreasing in $b=\text{baseK}/16$. The largest legal $b$ is therefore
optimal; the solver checks each legal core count because balanced-class
assignment of unequal output tails can change the busiest AIC's load:

$$
\begin{aligned}
b_g^*&=\min\!\left(\frac K{16},
 \left\lfloor\frac{C_{0A}}{64B_M}\right\rfloor,
 \left\lfloor\frac{C_{0B}}{64B_N}\right\rfloor,
 \left\lfloor\frac{C_1}{64(B_M+B_N)}\right\rfloor\right),\\
c_g^*&\in\underset{1\le c\le\min(A,N_o)}{\arg\min}
T_B(g,b_g^*,c),\qquad
P_{B,g}^*=\left\lceil\frac K{16b_g^*}\right\rceil.
\end{aligned}
$$

For the optional source-step packet objective, L1 step widths vary with
`baseK`, so this largest-$b$ elimination is not used: the solver checks
each feasible aligned `baseK` and core count for each source-compatible
geometry. Geometries with $b_g^*<1$ or invalid fixed storage/workspace are discarded.
Substituting $b_g^*,c_g^*$ into $T_B-L$ gives the local cost $V_{B,g}$.
This remains exact when a latency is zero: the largest feasible $b$ is still
one of the minimizers. The code chooses the largest $c$ among ties.

For Split-K, the upper bound from L1 is

$$
s_{\max,g}=\min\!\left(q,
 \left\lfloor\frac{C_1}{512(B_M+B_N)}\right\rfloor\right).
$$

The solver evaluates $T_D-L$ for each feasible integer
$c\in[2,\min(A,q,\lfloor(W_{\max}-W_0)/(8C_MC_N)\rfloor)]$ and each
$s\in[1,s_{\max,g}]$ satisfying $c\le\lceil q/s\rceil$. The smallest
value is $V_{D,g}$. Increasing $s$ reduces partial-result traffic but
may raise $q_{\rm crit}$ when whole K slices become imbalanced across
occupied AICs, so the largest legal step need not be optimal. Fixed L0
checks can remove a geometry before this bounded local scan.

For either family $X$, the remaining master is

$$
\min_{y}\ \sum_{g\in\mathcal G_X}V_{X,g}y_g,
\qquad\sum_{g\in\mathcal G_X}y_g=1,\qquad y_g\in\{0,1\}.
$$

It has one binary column per feasible geometry and one equality row, rather
than K-packet and core-count rows for every geometry. Constant
launch/conversion cost $L$ is added to the selected objective after solving.
`FinalizePlan` checks the selected plan again, and the event-graph scorer
runs separately when its explicit graph fits the event limit.

The workload's dtype, layout, transpose/bias mode, global alignment, 910B3
feature profile, and standalone family/key-tuple rules are **not MILP rows**.
The current `FinalizePlan` checks that narrow contract on the chosen plan; an
unsupported workload can therefore raise an error after the MILP solve.
Passing this standalone preflight still does not prove that CANN can emit the
plan.

HiGHS proves the minimum of these formulas over the declared spatial grid
and the encoded feasibility constraints. The BASE elimination and Split-K
local scan preserve
the optimum of the former per-geometry integer formulation; they do not
remove the finite geometry grid or make the surrogate a full timeline.
The `max` stage is a service-demand approximation: it omits the detailed
precedence, buffering, shared-resource contention, and issue timing of the
event graph. The current formulation requires zero fixed Cube latency.
For large shapes, the selected plan and surrogate remain available even when
the explicit event graph exceeds its cap or its whole-output Split-K partial
representation exceeds the per-tile reusable workspace; `event_graph_score`
is then absent.

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
build. The test compares each master optimum with a small exhaustive
oracle; that enumeration is confined to the test. The example's primitive
rates are hypothetical. Before using the result for device selection, the
next steps are to replace the reduced BASE packet service term with a
source-aligned per-core critical-path model, validate its use of the
[exploratory device coefficients](bench910/PRIMITIVE_910B3_CANN9.md), expand
the legal domains and schedule encoding, finalize selected plans with CANN,
and compare predicted rankings across a broader held-out tile set.

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
