# Mathematical model for matmul tiling and execution scheduling

This document specifies an analytical execution-timeline model for selecting
matmul tiling, buffering, task mapping, and implementation logic. It is a design
for a generalized estimator and optimizer; creating this document does not
extend the executable Python or C++ models.

The target hardware profile is **Huawei Ascend 910B3**. Section 2.3 supplies
its reference capacities and nominal rates. The earlier 24-core 910B2 model
outputs are retained only as repository references in sections 12 and 13.

For a fixed workload and hardware profile, the objective is

$$
\boxed{
x^\star \in \operatorname*{arg\,min}_{x\in\mathcal L(X,H)}
T_{\mathrm{timeline}}(X,x;H)
}
$$

where:

- $X$ is the workload: shape, data types, layouts, and numerical requirements.
- $H$ describes hardware resources, capacities, and primitive service times.
- $x$ is a complete implementation and tiling decision.
- $\mathcal L(X,H)$ is the set of executable, semantically correct plans.
- $T_{\mathrm{timeline}}$ is the completion time obtained by scheduling the
  plan's events under dependencies and resource limits.

The objective contains no whole-kernel residual multiplier, shape-time lookup,
or fitted correction applied after scheduling. It still requires explicit
hardware service parameters. Those parameters may come from architectural
assumptions or independent primitive measurements. Removing residual fitting
does not determine bandwidth or instruction latency automatically.

## Contents

- [Scope and assumptions](#1-scope-and-modeling-assumptions)
- [Workload and hardware notation](#2-workload-hardware-and-notation)
- [Ascend 910B3 hardware profile](#23-ascend-910b3-hardware-profile)
- [Decision variables](#3-decision-variables)
- [Task decomposition and coverage](#4-logical-task-decomposition-and-coverage)
- [Storage and legality](#5-storage-and-legality-constraints)
- [Static work and traffic](#6-deriving-static-work-from-the-program)
- [Primitive service times](#7-primitive-service-times)
- [Execution-timeline equations](#8-event-graph-and-execution-timeline-equations)
- [Shared bandwidth](#9-shared-bandwidth-and-multicore-contention)
- [Optimization and search](#10-full-optimization-problem)
- [Worked buffer timeline](#11-worked-timeline-one-versus-two-buffers)
- [Original and large-shape examples](#12-applying-the-formulation-to-the-discussed-shapes)
- [Existing repository behavior](#13-relationship-to-the-existing-repository)
- [Implementation boundary](#14-suggested-implementation-boundary)
- [Paper-derived refinements and scope](#15-paper-derived-refinements-and-scope)

## 1. Scope and modeling assumptions

The initial operation is

$$
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj},
\qquad A\in\mathbb R^{M\times K},\quad
B\in\mathbb R^{K\times N}.
$$

The execution interval begins at the modeled device entry and ends when the
final output is ready. Include required conversion, initialization, partial
accumulation, synchronization, reduction, casting, and output transfers.
Host allocation and warm-up are outside this interval unless explicitly added.

The model must declare:

1. The input, accumulator, workspace, and output data types.
2. The supported layouts, alignment rules, and instruction granularities.
3. The cache policy or cache-traffic assumption.
4. The transfer-arbitration policy across cores.
5. The kernel's actual loop order, prefetch policy, and synchronization.
6. Whether numerical reassociation, atomic accumulation, or a nondeterministic
   reduction order is permitted.

A longest-path result is exact for the specified graph and service assumptions.
It is an estimate of hardware execution. In particular, specification peak
rates describe idealized service unless additional behavior is modeled.

## 2. Workload, hardware, and notation

All times below use microseconds, all byte counts use bytes, and bandwidths use
bytes per microsecond. Convert cycles using an explicit clock frequency.

### 2.1 Fixed workload

$$
X=(M,N,K,\tau_A,\tau_B,\tau_{\mathrm{acc}},\tau_C,
   \mathrm{layout},\mathrm{strides},\mathrm{numerics}).
$$

Let $w_A,w_B,w_{\mathrm{acc}},w_C$ be the corresponding bytes per element.
For an FP16-input, FP32-accumulator example these are $2,2,4,2$, respectively.
The workspace type is a separate choice when the implementation permits it.

For a positive integer alignment $g$, define

$$
\operatorname{align}_g(z)=g\left\lceil\frac{z}{g}\right\rceil.
$$

A barred dimension, such as $\bar M_1$, denotes the allocated, padded
dimension for the selected layout. Different operands can require different
padding. An allocated tile dimension and an event's valid data extent are
different quantities.

### 2.2 Hardware profile

The profile $H$ contains:

| Symbol | Meaning |
|---|---|
| $C_{\max}$ | Maximum usable physical Cube cores (AICs) |
| $V_{\max}$ | Maximum usable physical Vector cores (AIVs) |
| $C_1,C_{0A},C_{0B},C_{0C}$ | L1, L0A, L0B, and L0C capacities per AIC |
| $C_{\mathrm{UB}}$ | UB capacity per AIV |
| $W_{\max}$ | Available global workspace |
| $g_M,g_N,g_K$ | Matrix-instruction or allocation granularities |
| $B_{\mathrm{HBM}}$ | Shared memory-service bandwidth |
| $B_r$ | Other explicitly modeled transfer-resource bandwidths |
| $\ell_e(H),\eta_{e,r}(H)$ | Event completion latency and resource initiation interval |
| $Q_r$ | Command/in-flight capacity, if explicitly modeled |

Local engines are identified by both core and engine type, for example
$(p,\mathrm{MTE1})$ for core index $p$. A shared HBM resource has one
identity across all cores.
Vector engines, shared caches, and interconnect links can be additional
resources.

Hardware parameters stay fixed while comparing candidates. Changing a service
rate separately for each candidate to match a measured kernel time would
reintroduce the fitting that this formulation is intended to avoid.

### 2.3 Ascend 910B3 hardware profile

Sources checked on 2026-10-22. Use the following full-device reference profile
when no target-specific resource restrictions are supplied. Distinguish
architectural storage limits, nominal performance ceilings, and parameters
that still require a target query or an explicit modeling assumption.
Here KiB means 1,024 bytes; TB/s and TFLOP/s use decimal powers of ten.

| Parameter | 910B3 reference value | Scope and provenance |
|---|---:|---|
| $C_{\max}$ | 20 | Physical Cube cores; the B3 experimental platform in [SGEMM-cube, section 6.1](https://arxiv.org/html/2507.23387v5#S6.SS1) reports 20 AI cores |
| $V_{\max}$ | 40 | Vector-core reference count; see topology qualification below |
| $f_{\mathrm{AIC}}$ | 1.8 GHz = 1,800 cycles/us | Nominal clock of that research platform; verify the operating clock on the target |
| $B_{\mathrm{HBM}}^{\mathrm{nom}}$ | 1.6 TB/s = 1,600,000 bytes/us | Whole-device main-memory bandwidth reported for that research platform |
| $F_{\mathrm{peak,chip}}^{16}$ | Approximately 294.91 TFLOP/s = 294,910,000 FLOP/us | FP16/BF16 reference peak from [Huawei Cloud's MFU methodology](https://github.com/huaweicloud/huaweicloud-skills/blob/master/skills/ai/modelarts/huawei-cloud-ascend-op-mfu-calculator/references/mfu-calculation-methodology.md) |
| $C_1$ | 512 KiB = 524,288 bytes | L1, per AIC |
| $C_{0A}$ | 64 KiB = 65,536 bytes | L0A, per AIC |
| $C_{0B}$ | 64 KiB = 65,536 bytes | L0B, per AIC |
| $C_{0C}$ | 128 KiB = 131,072 bytes | L0C, per AIC |
| $C_{\mathrm{UB}}$ | 192 KiB = 196,608 bytes | UB, per AIV |

The local capacities are the Atlas A2/A3 entries in Huawei's
[PTO memory-capacity table](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/compiler/ptoisa/docs/isa/TASSIGN.md).
Use the A2/A3 column, not the different Ascend 950 column. Validate available
capacities against the installed CANN platform information.

**Cube and Vector topology.** The 20/40 profile combines the B3 core count
above with Huawei's documented separated execution model: a mixed group has
one AIC and two AIVs. Huawei's
[910B3 IVFFLAT configuration](https://www.hiascend.com/doc_center/source/en/mindsdk/730/Indexn/indexug/mxindexfrug_0587.html)
also specifies a count of 40 for its vector workload, although that page uses
the generic term AI Cores. Confirm both counts with the platform APIs before
executing a plan. See the official
[kernel launch rules](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0014.html)
for the AIC/AIV distinction and mixed-group limits.

In this formulation, $c$ counts physical AIC workers. Therefore

$$
1\le c\le 20.
$$

For a mixed implementation using two AIVs per AIC, also require
$2c\le V_{\max}^{\mathrm{available}}$ and
$c\le C_{\max}^{\mathrm{available}}$. Model the two AIVs separately;
their two 192 KiB UBs are not one freely shared 384 KiB allocation.
Logical launch blocks and physical workers are different: a supported
Cube-only launch can have more logical blocks than physical cores, but the
timeline must schedule those blocks onto the available workers. Mixed
launches must satisfy their documented group limits.

#### Concrete storage constraints

For FP16/BF16 operands and FP32 accumulation, a fixed-slot layout must satisfy

$$
2q_{1A}\bar M_1\bar K_{1A}
+2q_{1B}\bar K_{1B}\bar N_1
+W_{1,\mathrm{extra}}\le 524288,
$$

$$
2q_{0A}\bar M_0\bar K_0+W_{0A,\mathrm{extra}}\le 65536,
\qquad
2q_{0B}\bar K_0\bar N_0+W_{0B,\mathrm{extra}}\le 65536,
$$

$$
4q_C\bar M_0\bar N_0+W_{0C,\mathrm{extra}}\le 131072,
\qquad
W_{\mathrm{UB},v}^{\mathrm{live}}(t)\le196608
\quad\text{for each AIV }v.
$$

All terms are bytes. Extra storage includes alignment gaps and any other
simultaneously live allocations in that memory. Section 5 gives the more
general lifetime formulation when fixed-size slots are insufficient.

For example, $M_0=N_0=K_0=128$ with $q_{0A}=q_{0B}=q_C=2$ uses
64 KiB L0A, 64 KiB L0B, and 128 KiB L0C. It fills these capacities exactly
and leaves no room for extra allocations there. With
$M_1=N_1=128$, $K_{1A}=K_{1B}=256$, and $q_{1A}=q_{1B}=2$,
the two operand rings together use 256 KiB of L1.
These are storage checks; executable buffering and useful overlap still
depend on the selected instructions and dependencies.

For b16 operands, the
[Load2D documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/Load2D.md)
describes a 16-by-16-element fractal occupying 512 bytes. A practical initial
tile search can use $g_M=g_N=g_K=16$, with tail padding/masking.
This is a layout/search choice, not a requirement that the logical
$M,N,K$ be multiples of 16. Apply instruction-specific address constraints:
for example,
[Load2DV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/Load2DV2.md)
requires 32-byte L1 and 512-byte L0A/L0B address alignment.

#### Nominal rates in the time equations

With FMA counted as two floating-point operations, the equal-share reference
peak for one AIC is

$$
F_{\mathrm{peak,core}}^{16}
=\frac{294910000}{20}
=14745500\ \mathrm{FLOP/us}.
$$

For an event executing $W_e$ FP16/BF16 Cube FLOPs, the ideal throughput term
is $W_e/14745500$ us. This does not specify the instruction's completion
latency or initiation interval. Those remain distinct parameters in
sections 7 and 8. Do not apply this peak to vector arithmetic, conversions,
reductions, or a different matrix input type.

At the nominal AIC clock, an AIC-domain latency of $n$ cycles converts to

$$
t_{\mathrm{us}}=\frac{n}{1800}.
$$

For $D$ bytes actually serviced by HBM, the nominal shared-service term is

$$
t_{\mathrm{HBM,ideal}}(D)=\frac{D}{1600000}\ \mathrm{us}.
$$

Use this bandwidth once across the whole device in section 9. It is not the
bandwidth of each core or of L1-to-L0 transfers. Add explicitly modeled
startup, arbitration, and path constraints. A peak-rate scenario is useful
for optimization and lower bounds; actual sustained rates may be smaller.
Using independent primitive measurements is compatible with the absence
of a whole-kernel fitted correction.

#### Parameters to obtain from the actual target

The following are deliberately not assigned guessed B3 constants:

| Parameter | How it enters the formulation |
|---|---|
| L2 capacity, bandwidth, and cache policy | Determine cache traffic and any explicit shared-L2 resource |
| Usable HBM capacity and free workspace | Constrain allocations; $W_{\max}$ is the workspace budget after other live allocations |
| L1-to-L0, output-path, and vector service rates | Set the corresponding local resource service terms |
| Instruction/startup latencies, initiation intervals, synchronization, and command credits | Determine issue, completion, and buffer-release times |
| Available AIC/AIV counts and operating clocks | Override full-device nominal assumptions when resources or clocks differ |

Inside a CANN host tiling function, the following illustrates retrieving
resource counts and capacities; it is not a standalone C++ program:

```cpp
#include <cstdint>
#include "tiling/platform/platform_ascendc.h"

// context is the host tiling context supplied by CANN.
auto platform =
    platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
const auto available_aic = platform.GetCoreNumAic();
const auto available_aiv = platform.GetCoreNumAiv();
using Mem = platform_ascendc::CoreMemType;
uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0, ub = 0, l2 = 0, hbm = 0;
platform.GetCoreMemSize(Mem::L1, l1);
platform.GetCoreMemSize(Mem::L0_A, l0a);
platform.GetCoreMemSize(Mem::L0_B, l0b);
platform.GetCoreMemSize(Mem::L0_C, l0c);
platform.GetCoreMemSize(Mem::UB, ub);
platform.GetCoreMemSize(Mem::L2, l2);
platform.GetCoreMemSize(Mem::HBM, hbm);
// Check the returned values against the target and required resources.
// The capacities are bytes; HBM capacity is not free workspace.
```

See [PlatformAscendC](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00059.html),
[GetCoreNumAic](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1030.html),
[GetCoreNumAiv](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1031.html),
and [GetCoreMemSize](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1034.html).
The separate
[GetCoreMemBw API](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1035.html)
reports **bytes/cycle**, not bytes/second, and documents L2/HBM support.
Convert a returned value $b_r$ as $B_r=b_rf_r$, where $f_r$ is the
documented clock domain in cycles/us; do not automatically use the AIC clock
for a memory-domain value.

Record each parameter's source, units, and status (architectural limit,
nominal ceiling, explicit assumption, or primitive measurement) in the
hardware profile. Missing timing parameters must remain visible; silently
setting them to zero produces a different, idealized model.

## 3. Decision variables

Use the following structured decision vector:

$$
x=(a,c,S_M,S_N,\kappa,
   M_1,N_1,K_{1A},K_{1B},
   M_0,N_0,K_0,
   \mathbf q,\pi,\mu).
$$

| Variable | Decision |
|---|---|
| $a$ | Implementation family or realizable kernel strategy |
| $c$ | Number of physical AIC workers, $1\le c\le C_{\max}$; at most 20 in the full 910B3 profile |
| $S_M,S_N$ | Nominal output-task dimensions, corresponding to singleCoreM/N |
| $\kappa$ | K chunking, ownership, local accumulation, and reduction policy |
| $M_1,K_{1A}$ | Resident A-tile dimensions in L1 |
| $K_{1B},N_1$ | Resident B-tile dimensions in L1 |
| $M_0,N_0,K_0$ | Inner computation/L0 tile dimensions |
| $\mathbf q=(q_{1A},q_{1B},q_{0A},q_{0B},q_C)$ | Resident buffer-slot counts |
| $\pi$ | Loop order, reuse, prefetch, and buffer-assignment policy |
| $\mu$ | Task-to-core mapping and task order |

The two L1 K spans allow A and B to be loaded at different frequencies. A
simpler family can constrain $K_{1A}=K_{1B}=K_1$.

### 3.1 Task size versus memory tile size

$S_M$ is a task extent; it need not equal $M_0$ or $M_1$. A core can
process a large task by iterating over smaller resident tiles.

One possible family-specific parameterization is

$$
S_M=u_M M_0,\qquad S_N=u_N N_0,\qquad
K_{1A}=u_AK_0,\qquad K_{1B}=u_BK_0,
$$

with positive integer multipliers. These are optional kernel constraints,
not universal identities. In particular, do not assume that an existing
kernel's $\mathrm{stepM}$ always means $S_M/M_0$.

### 3.2 Implementation logic must be executable

A strategy $a,\pi,\mu$ must define the program that realizes the selected
reuse and overlap. Example choices include:

- Full-K output tasks distributed over cores.
- K chunks kept on the same owner core with local accumulation.
- K chunks distributed over cores, followed by a specified reduction.
- A loop nest that retains an A tile while visiting multiple N tiles.
- A loop nest that retains a B tile while visiting multiple M tiles.
- Single buffering or a specified ping-pong/prefetch schedule.

An algorithm name alone is insufficient. The plan must specify layout,
accumulator placement, transfer order, and synchronization. If a new schedule
cannot be emitted by an existing kernel, it requires a corresponding kernel
implementation or code generator.

## 4. Logical task decomposition and coverage

### 4.1 Output tasks

Define

$$
G_M=\left\lceil\frac{M}{S_M}\right\rceil,\qquad
G_N=\left\lceil\frac{N}{S_N}\right\rceil.
$$

For $0\le i<G_M$ and $0\le j<G_N$,

$$
m_i=\min(S_M,M-iS_M),\qquad
n_j=\min(S_N,N-jS_N).
$$

There are $G_MG_N$ output regions. A physical core may process several
regions. Thus singleCoreM is not the total number of rows processed by that
core during the kernel.

### 4.2 K chunks and ownership

For a uniform chunking policy with nominal span $S_K$, let

$$
G_K=\left\lceil\frac{K}{S_K}\right\rceil,\qquad
k_h=\min(S_K,K-hS_K).
$$

Logical contribution task $(i,j,h)$ computes a contribution of shape
$m_i\times n_j$ over a K interval of length $k_h$. A full-K strategy can
use $S_K=K$.

If assignment is optimized explicitly, binary variables satisfy

$$
y_{ijhp}\in\{0,1\},\qquad
\sum_{p=0}^{c-1}y_{ijhp}=1.
$$

Here $c$ is the chosen core count and $p$ is a core index.
A local-K strategy constrains all K chunks of one output
region to the same owner. A distributed strategy permits multiple owners and
requires a valid partial-output merge.

Every required product contribution must be computed exactly once, except
deliberately modeled redundant work, and accumulated exactly once into the
intended output. Tail masking and padding must preserve this condition.

### 4.3 Partial outputs are not necessarily K chunks

Let $P_{ij}$ be the number of final partial outputs materialized for output
region $(i,j)$. Under per-core aggregation,

$$
P_{ij}=
\left|
\left\{p:\exists h,\ y_{ijhp}=1\right\}
\right|.
$$

Under a one-partial-per-chunk strategy, $P_{ij}=G_K$. The implementation must
state which policy it uses.

For a direct reduction that reads each materialized partial once:

$$
W_{\mathrm{add}}=\sum_{i,j}(P_{ij}-1)m_i n_j,
$$

$$
D_{\mathrm{partial\,read}}
=w_{\mathrm{partial}}\sum_{i,j}P_{ij}m_i n_j,\qquad
D_{\mathrm{final\,write}}=w_CMN.
$$

These are logical work counts. Add allocation padding and transaction effects
when constructing physical transfers. Partial stores, intermediate atomic
updates, workspace initialization, reduction-tree intermediate traffic, and
casts are separate events when present. Counting final partial outputs alone
does not count every writeback performed during computation.

## 5. Storage and legality constraints

### 5.1 General live-storage constraint

For memory level $h$ and core index $p$,

$$
\boxed{
\max_t
\sum_{b\in\mathrm{live}(h,p,t)}
\operatorname{allocatedBytes}(b)
\le C_h
}
$$

Include every simultaneously resident operand, accumulator, staging buffer,
layout-conversion buffer, and scratch allocation. Aliased buffers can share
storage only when their live ranges do not overlap.

### 5.2 Fixed-slot specialization

For separately allocated L1 A/B rings:

$$
w_Aq_{1A}\bar M_1\bar K_{1A}
+w_Bq_{1B}\bar K_{1B}\bar N_1
+W_{\mathrm{extra},1}
\le C_1.
$$

For separately allocated L0 operand and accumulator rings:

$$
w_Aq_{0A}\bar M_0\bar K_0+W_{\mathrm{extra},0A}\le C_{0A},
$$

$$
w_Bq_{0B}\bar K_0\bar N_0+W_{\mathrm{extra},0B}\le C_{0B},
$$

$$
w_{\mathrm{acc}}q_C\bar M_0\bar N_0
+W_{\mathrm{extra},0C}\le C_{0C}.
$$

The last equation assumes each L0C slot contains one inner output tile. If a
strategy keeps a larger accumulator resident, use its actual dimensions.
Likewise, preserving partial output elsewhere requires that storage and its
spill/reload events.

The entire $S_M\times S_N\times K$ task does not need to be resident. Its
loop nest determines which smaller tiles are live together.

### 5.3 Other legality conditions

The legal set also enforces:

- Positive integer tile dimensions and supported buffer counts.
- Instruction, DMA, layout, and address-alignment requirements.
- Supported divisibility relationships, or explicit handling of remainders.
- Legal nonempty K partitions and correct ownership of partial outputs.
- Required numerical accuracy and deterministic ordering, if requested.
- Valid barriers, command credits, and race-free output updates.
- Workspace and vector/local scratch capacity.
- Backend limits on descriptors, strides, task counts, and integer widths.

Do not impose $M\bmod S_M=0$ universally: tail-aware kernels can handle
nondivisible shapes. Conversely, do not assume every kernel supports tails.
An otherwise memory-feasible candidate can still be unimplementable.

## 6. Deriving static work from the program

### 6.1 Arithmetic

Logical useful work is $2MNK$ floating-point operations under the convention
that a multiply-add counts as two operations.

If a matrix event operates on active extents $m_e,n_e,k_e$ with granularity
$g_M,g_N,g_K$, a padded-work model is

$$
W_e^{\mathrm{cube}}
=2\,\operatorname{align}_{g_M}(m_e)
     \operatorname{align}_{g_N}(n_e)
     \operatorname{align}_{g_K}(k_e).
$$

Use this only when it matches the emitted instruction behavior. Padding every
boundary task to its nominal macro-task size can overcount work. The source
adapter must identify the actual active/padded instruction extents.

### 6.2 Operand traffic and reuse

For a logical A packet of dimensions $m\times k$ and B packet $k\times n$,

$$
D_A^{\mathrm{logical}}=w_A mk,\qquad
D_B^{\mathrm{logical}}=w_B kn.
$$

Count a transfer each time the program actually fetches a packet into the
destination memory level. Reusing a retained packet adds reader events but
does not add another fetch into that level. Loading it into L0 again can
still incur L1-to-L0 traffic.

For an illustrative dense, no-cache, full-K schedule with one output tile
per independent task and no reuse across tasks:

$$
D_A^{\mathrm{logical}}=w_A MK\,G_N,\qquad
D_B^{\mathrm{logical}}=w_B KN\,G_M.
$$

These totals change if tasks share retained operands, if a macro-task contains
inner tiles that reload operands, or if a cache serves repeated requests.
Derive traffic from the selected loops and residency rather than applying
these formulas to every implementation.

### 6.3 Physical transactions

For a row beginning at byte address $a$, containing $b>0$ requested bytes,
and transaction granularity $g$, an elementary transaction-count model is

$$
n_{\mathrm{txn}}(a,b,g)
=\left\lfloor\frac{a+b-1}{g}\right\rfloor
 -\left\lfloor\frac{a}{g}\right\rfloor+1.
$$

The associated footprint is $g\,n_{\mathrm{txn}}$. Summing such footprints
assumes independently serviced rows. If rows or requests share cache lines,
an explicit coalescing/cache policy determines whether those lines are fetched
again. This formula is a selectable traffic assumption, not a universal DMA
law.

Maintain separate byte counts for GM/HBM, L1-to-L0, workspace, and final output.
Do not charge a cache hit as a new HBM fetch when the chosen cache model says
it is served locally.

### 6.4 Explicit L2 path and layout transformations

[NPUMeter](3820380.pdf), sections 2.3 and 4.3, separates MTE2, MTE1, and
Fixpipe transfers and accounts for format transformations. For each actual
operand fetch $r$, record its source layout, destination layout, physical
bytes $d_r$, and whether it misses L2. Under a policy in which all such
requests pass through L2, define the miss indicator $m_r\in\{0,1\}$:

$$
D_{\mathrm{HBM}}=\sum_r m_r d_r,\qquad
D_{\mathrm{L2\to L1}}=\sum_r d_r.
$$

The first count is the off-chip traffic; the second counts every delivery
to L1, including L2 hits. Add output, workspace, and spill traffic on their
actual paths. A different cache or bypass policy needs its own path counts.
Determine $m_r$ from a declared cold/warm-cache assumption or a capacity and
replacement model. In particular, a large working set can evict an input
between uses, so "first access misses, every later access hits" is not a
general law.

For an ND input path that uses Nz in L1, the A and B transforms may be:

| Stage | A path | B path | Resource to model |
|---|---|---|---|
| MTE2 | ND to Nz | ND to Nz | Off-chip/L2 service, MTE2, and conversion |
| MTE1 | Nz to Zz | Nz to Zn | MTE1, L1-to-L0 transfer, and conversion |
| Cube | L0A Zz | L0B Zn | Cube execution, L0C accumulation |
| Fixpipe | L0C to output layout | L0C to output layout | Output transfer and any output conversion |

This is the layout route described in the paper's Figure 3 for its GEMM
example. The chosen implementation must supply its real instruction modes
and layouts; a preconverted operand or a different supported format changes
the event sequence. A transform may be fused into a transfer command, but
its service demand must still be represented.

## 7. Primitive service times

Each event has a completion latency $\ell_e\ge0$. Each issue resource that
it occupies has an initiation interval $\eta_{e,r}\ge0$. A simple pipeline
usually has $\eta_{e,r}\le\ell_e$. A nonpipelined engine uses
$\eta_{e,r}=\ell_e$.

Examples of possible primitive models are:

$$
t_{\mathrm{setup}}=n_{\mathrm{desc}}\lambda_{\mathrm{desc}},
\qquad
t_{\mathrm{transfer}}=\frac{D^{\mathrm{physical}}}{B_r},
$$

$$
t_{\mathrm{cube\,sequence}}
=L_{\mathrm{first}}+(n_{\mathrm{inst}}-1)I_{\mathrm{inst}},
\qquad n_{\mathrm{inst}}\ge1.
$$

The sequence formula assumes identical eligible instructions can issue at
interval $I_{\mathrm{inst}}$. Data or accumulator dependencies can force
larger gaps; model those dependencies explicitly instead of using the formula
unconditionally. Empty sequences have zero work.

An alternative primitive cost representation is

$$
\ell_e=\sum_j\theta_{r,j}q_{e,j},
$$

where $q_{e,j}$ are static bytes, descriptor counts, instruction counts, or
other primitive work, and $\theta_{r,j}$ are fixed service coefficients.
This representation remains residual-free when no candidate-specific
whole-kernel correction is applied afterward. Coefficients obtained from
primitive measurements should record their measurement conditions.

Completion latency and initiation interval are separate inputs. A model that
only knows completion latency cannot infer issue throughput uniquely.

### 7.1 Native Cube blocks and two levels of pipelining

For a 16-by-16-by-16 FP16/BF16 Cube block, a tile with padded extents
$(\bar m_e,\bar n_e,\bar k_e)$ contains

$$
b_e=\frac{\bar m_e}{16}\frac{\bar n_e}{16}
    \frac{\bar k_e}{16},\qquad
W_e^{\mathrm{cube}}=8192b_e\ \mathrm{FLOP}.
$$

If that block stream can issue with first-block latency
$L_{\mathrm{block}}$ cycles and interval $I_{\mathrm{block}}$ cycles, the
completion of an otherwise uninterrupted tile stream is

$$
L_{\mathrm{tile}}
=L_{\mathrm{block}}+(b_e-1)I_{\mathrm{block}},\qquad b_e\ge1.
$$

This instantiates the paper's section 4.2, Equation (1). Convert cycles
using the operating AIC clock. For example, the *conditional* assumption
$I_{\mathrm{block}}=1$ cycle at 1.8 GHz yields an asymptotic
$8192\times1800=14{,}745{,}600$ FLOP/us per AIC, close to the
14,745,500 FLOP/us rounded reference in section 2.3. This is a consistency
check, not a measured 910B3 issue interval. The first-block latency,
accumulator hazards, instruction mode, and tail handling still matter.

There is a second pipeline between tiles. If tile $j+1$ needs an operand
that has not reached L0A/L0B, its Cube start stalls even when the block
issue pipeline is otherwise available. Section 8.5 gives the readiness
recurrence. An instruction-level graph may replace this tile formula when
its individual dependencies are needed; do not charge both representations.

### 7.2 Transfer and conversion service

For a transfer stage $r$ carrying $D$ physical bytes, a selectable primitive
service model is

$$
\ell_r(D,\mathrm{mode},\mathrm{shape})
=\lambda_r+\frac{D}{B_r}
 +\phi_r(D,\mathrm{mode},\mathrm{shape}).
$$

Here $\lambda_r$ is a command/startup term and $\phi_r$ is the cost of the
chosen format transformation. NPUMeter's section 4.3 uses an additive
shape-dependent coefficient, such as
$\phi_{\mathrm{ND\to Nz}}=\alpha_{\mathrm{ND\to Nz}}(\mathrm{shape})D$
for MTE2. For this formulation, such a coefficient must be fixed for a
primitive transfer mode from hardware documentation or an isolated
microbenchmark; it is never fitted to the candidate kernel's total time.
If copy and conversion overlap on the target, replace the sum by separate
resource events or a measured *combined primitive* service function.

Treat HBM-to-L2, L2-to-L1/MTE2, L1-to-L0/MTE1, and L0C-to-output/Fixpipe
as distinct services when the selected path traverses them. Give each its
own latency, issue interval, contention domain, and layout mode. The
whole-device 1.6 TB/s reference applies only to HBM service. Section 9's
shared HBM event represents this same HBM stage; do not charge it twice.

## 8. Event graph and execution-timeline equations

### 8.1 Events and edges

The program builder produces

$$
G(X,x;H)=(V,E).
$$

Events include scalar commands, memory service, operand loads, matrix
instructions, output transfers, conversions, reductions, and synchronization.
An event may be split into issue, service, and completion events when those
parts have distinct resource behavior.

Let $s_e$ and $f_e$ be event start and finish times:

$$
f_e=s_e+\ell_e,\qquad s_e\ge0.
$$

Data and control dependencies require

$$
s_e\ge f_u,\qquad u\in D(e).
$$

On issue resource $r$, with preceding event $p_r(e)$,

$$
s_e\ge s_{p_r(e)}+\eta_{p_r(e),r}.
$$

Resource order is per core for private engines and global for shared engines.
Using previous completion instead of previous issue availability would
serialize a pipelined engine unnecessarily.

### 8.2 Finite buffers

Assign packet $j$ to slot $j\bmod q$ in a ring of $q$ slots. Its next
writer must wait for every remaining reader of the old contents:

$$
s_{\mathrm{next\ writer}}
\ge \max_{u\in\mathrm{old\ readers}}f_u.
$$

Apply this dependency at the appropriate memory level:

- L1 A/B: release after the last required L1-to-L0 read.
- L0 A/B: release after the consuming computation finishes, unless a supported
  earlier-consumption rule is explicitly modeled.
- L0C: preserve the accumulator across its K updates and release it after
  its final writeback completes.

When an operand is reused by several output tiles, all those readers extend
its lifetime. If K updates are spilled between chunks, represent the spill,
reload, or atomic accumulation rather than assuming uninterrupted residence.

### 8.3 Commands, barriers, and accumulation

Scalar issue overhead is represented by scalar-resource events. A barrier
depends on all required participants; subsequent work depends on that barrier.
A queue of $Q_r$ outstanding commands creates a credit dependency between
command $j$ and the event releasing the credit used by command $j-Q_r$.
Define precisely which event releases that credit.

Updates to one accumulator have an ordering dependency. Completion-based
dependencies are conservative when no forwarding information is available.
If the hardware permits earlier dependent issue, use its explicit forwarding
delay instead. Any proposed reduction order must satisfy the workload's
numerical requirements.

### 8.4 Earliest-start recurrence

For a fixed legal event/resource order,

$$
\boxed{
s_e=
\max\left(
0,\;
\max_{u\in D(e)} f_u,\;
\max_{r\in\mathcal R(e)}
  [s_{p_r(e)}+\eta_{p_r(e),r}],\;
R_e
\right)
}
$$

where $R_e$ is the latest release time of the buffer slots required by the
event. Include barrier, command-credit, and initialization readiness in
$D(e)$ or as explicit additional edges. Missing predecessors contribute zero.

Then

$$
\boxed{
T_{\mathrm{timeline}}(X,x;H)
=\max_{e\in V_{\mathrm{terminal}}}f_e.
}
$$

Terminal events must include all required final writes and completion
synchronization. Entry overhead can be a root event; do not add it again
outside the graph.

Equivalently, use weighted start-time edges $s_v\ge s_u+w_{uv}$. Data edges
have weight $\ell_u$, issue-order edges have weight $\eta_{u,r}$, and
buffer-reuse edges use the old consumer's release delay. A sink edge of
weight $\ell_e$ accounts for each terminal event's completion.

For an acyclic graph, earliest starts are longest-path distances from the
source. Evaluation costs $O(|V|+|E|)$. Reject cyclic orderings in this DAG
formulation; an inconsistent resource order can introduce a dependency cycle.
Do not count the sum of all event durations as the kernel time, because
independent resources can overlap.

### 8.5 Operand-readiness stalls between Cube tiles

For consecutive Cube tiles $j-1,j$ on one AIC, let
$r_j=\max(f_{\mathrm{MTE1},A,j},f_{\mathrm{MTE1},B,j})$ be the time both
required operands are ready in L0. Let $u_j$ be the earliest Cube issue time
after the prior tile's issue interval, accumulator dependencies, scalar
commands, and slot releases:

$$
u_j=\max\left(
s_{j-1}^{\mathrm{cube}}+I_{\mathrm{tile},j-1},
R_j^{\mathrm{acc}},
R_j^{\mathrm{cmd}},
R_j^{\mathrm{slot}}
\right),
\qquad
s_j^{\mathrm{cube}}=\max(u_j,r_j).
$$

The exposed operand wait is

$$
\delta_j^{\mathrm{operand}}=\max(0,r_j-u_j).
$$

When blocks of two tiles share one continuous Cube issue pipeline,
$I_{\mathrm{tile},j-1}=b_{j-1}I_{\mathrm{block}}/f_{\mathrm{AIC}}$
is a possible start-to-start issue separation. The tile's completion is
$f_j^{\mathrm{cube}}=s_j^{\mathrm{cube}}
+L_{\mathrm{tile},j}/f_{\mathrm{AIC}}$, subject to any finer instruction
dependencies. This makes the stalls in NPUMeter's Figures 5 and 8
candidate-specific consequences of actual operand arrival. Summing
$\delta_j^{\mathrm{operand}}$ is useful for diagnosis; do not add that sum
to the event graph's makespan because the waits are already present in
$s_j^{\mathrm{cube}}$.

### 8.6 Cube-to-Vector handoff when the implementation uses it

Some implementation families send Cube output to an AIV for conversion,
reduction, or an epilogue. For a genuinely consumable output subtile $h$,
create a publication event and require

$$
s_{h}^{\mathrm{AIV}}
\ge f_{h}^{\mathrm{publish}},\qquad
s_{\mathrm{overwrite}(h)}^{\mathrm{AIC}}
\ge f_{h}^{\mathrm{last\ AIV\ read}}.
$$

Publication includes the required Fixpipe/workspace transfer and visibility
or synchronization. Charge the corresponding L2/HBM bytes and buffer
occupancy. If the AIV must wait for a complete Cube tile or for the final
Split-K reduction, publish only at that later boundary. This captures the
paper's Figure 10 idea of starting a consumer after the smallest valid
producer result; it applies only when the kernel and numerical contract
actually permit that subdivision.

An AIV computation can likewise use per-instruction latency and two issue
intervals: one between blocks of the same vector operation and one for
switching operation types. The paper's section 4.2 uses 256-byte blocks for
its evaluated vector operations. For this model, derive the block granularity
and instruction sequence from the selected B3 kernel and represent each
resource's issue order in the graph.

## 9. Shared bandwidth and multicore contention

### 9.1 A shared service queue

A tractable baseline uses one shared HBM service resource. Each request $e$
has a memory-service event of duration

$$
d_e^{\mathrm{HBM}}
=\frac{D_e^{\mathrm{physical}}}{B_{\mathrm{HBM}}}.
$$

For the selected arbitration order,

$$
s_e^{\mathrm{HBM}}
\ge f_{p_{\mathrm{HBM}}(e)}^{\mathrm{HBM}},\qquad
f_e^{\mathrm{HBM}}=s_e^{\mathrm{HBM}}+d_e^{\mathrm{HBM}}.
$$

All cores submit work to this same resource. A request's path can be

```text
descriptor setup -> shared memory service -> local return/completion -> consumer
```

Descriptor setup and return overhead may overlap other memory service when
their resource constraints allow it. Charge the shared $D/B_{\mathrm{HBM}}$
service once. Adding the same full-bandwidth service time again to a local
DMA latency would double-count it. Distinct links, such as HBM and an on-chip
transfer path, can each have their own service events if bytes traverse both.

A fixed request ordering produces a graph directly. A dynamic policy such as
FIFO among ready requests requires a discrete-event scheduler to choose the
next request, including a declared tie-break rule; the resulting ordering can
then be represented as graph edges.

Whole-request serialization is a modeling assumption. It can exaggerate
head-of-line blocking when real hardware interleaves smaller transactions.
Splitting requests into service chunks improves that model at the cost of
more events. Shared throughput is constrained in either case.

### 9.2 Optional concurrent-rate model

A more detailed alternative assigns instantaneous rates $b_e(t)$:

$$
b_e(t)\ge0,\qquad
\sum_{e\in\mathrm{active}(t)}b_e(t)\le B_{\mathrm{HBM}},
$$

$$
\int_{s_e^{\mathrm{service}}}^{f_e^{\mathrm{service}}}
b_e(t)\,dt=D_e^{\mathrm{physical}}.
$$

The arbitration policy determines the rates. Under this alternative,
transfer finish times depend on competing requests and are not fixed
independent $D_e/B_{\mathrm{HBM}}$ latencies. This generally requires a
discrete-event or fluid-service simulation rather than a single fixed-weight
DAG evaluation.

Select one shared-service model consistently. Giving every core the full
shared bandwidth independently would allow impossible aggregate throughput.

### 9.3 L2 service alongside HBM service

Under section 6.4's cache path, an L2 hit still consumes L2-to-L1/MTE2
service. An L2 miss additionally consumes shared HBM service and may require
an L2 fill before MTE2 delivery. Define a separate L2 service resource with
its own capacity and arbitration; use $D_{\mathrm{L2\to L1}}/B_{\mathrm{L2}}$
only as a service-demand bound. The graph determines how its individual
requests overlap with Cube work, other L2 requests, and HBM fills.

The paper's Equation (8) separates HBM and L2 byte terms. In a multicore
910B3 model, both bandwidths are shared resources. A single
$D_{\mathrm{HBM}}/B_{\mathrm{HBM}}
+D_{\mathrm{L2\to L1}}/B_{\mathrm{L2}}$ sum describes a serialized path,
not necessarily the elapsed time of overlapping requests from many cores.
Cache-residency assumptions and resource arbitration must be explicit.

## 10. Full optimization problem

For a fixed workload $X$ and hardware profile $H$, let
$\mathcal A_{\mathrm{emit}}(X,H)$ be the finite set of kernel families
that the backend can actually compile and launch for this workload.
For family $a$, let $z$ contain its supported choices of task geometry,
K ownership, L1/L0 tiles, buffer depths, loop order, mapping, and core
count. Write the complete plan as $x=(a,z)$. The joint problem is

$$
\begin{aligned}
\min_{x,\mathbf s,\mathbf f,T}\quad &T\\
\text{subject to}\quad
&x\in\mathcal L(X,H),\\
&f_e=s_e+\ell_e(X,x;H), &&e\in V(X,x),\\
&s_e\ge0, &&e\in V(X,x),\\
&s_e\ge f_u, &&u\in D(e),\\
&s_e\ge s_{p_r(e)}+\eta_{p_r(e),r},
  &&r\in\mathcal R(e),\\
&s_e\ge R_e, &&e\in V(X,x),\\
&T\ge f_e, &&e\in V_{\mathrm{terminal}}(X,x).
\end{aligned}
$$

The graph includes shared-service events, buffer release, barriers, and
initialization. Storage, semantic, alignment, workspace, and kernel support
constraints are part of $\mathcal L$. For the concurrent-rate memory
alternative, replace the fixed-duration equations for shared-memory service
events with Section 9.2's rate and byte-conservation constraints. Retain fixed
latencies for the other primitive stages. An end-to-end transfer's completion
then depends on both shared-service completion and its local return stages.

The problem is discrete and generally nonlinear: ceilings, tile products,
task counts, dependencies, and even the event set vary with $x$. It is not
automatically a small linear program.

The first [HiGHS-backed reduced MILP](standalone_optimizer/README.md#reduced-milp-with-mn-block-tiling)
is a concrete solver step toward this formulation. It optimizes BASE and
deterministic Split-K separately over a declared grid of output/base M/N
blocks, K tiles or steps, and AIC counts, with an analytical compute/GM
service objective.
It avoids enumerating complete tilings in the C++ optimizer; HiGHS proves
optimality for that **restricted surrogate only**. The chosen plans are then
scored with the event graph for comparison. Section 10's complete
tiling-dependent event set, resource schedule, and backend legality remain
outside the MILP, so the current solver does not certify the global optimum
of the full problem above.

### 10.1 Make the feasible set explicit

The optimization is useful only if $\mathcal L(X,H)$ describes
**emittable** plans rather than mathematically attractive tile sizes:

$$
\mathcal L(X,H)=
\bigcup_{a\in\mathcal A_{\mathrm{emit}}(X,H)}
\left\{(a,z):
z\in\mathcal D_a(X,H),\
\operatorname{cover}(X,a,z),\
\operatorname{numeric}(X,a,z),\
\operatorname{storage}(X,a,z;H),\
\operatorname{backend}(X,a,z)
\right\}.
$$

$\mathcal D_a$ is a finite, family-specific decision domain. The four
predicates have concrete responsibilities:

| Predicate | Required check |
|---|---|
| $\operatorname{cover}$ | Every valid $(i,j,k)$ contribution is assigned once, tails are masked correctly, and all partials reach the final output |
| $\operatorname{numeric}$ | Accumulator type, K partition, reduction order, and output cast meet the workload's numerical contract |
| $\operatorname{storage}$ | For every core and time, live L1/L0/UB use fits section 2.3; global workspace fits $W_{\max}$ |
| $\operatorname{backend}$ | Instruction shapes, layouts, alignments, address ranges, synchronization, launch mode, and chosen loop order are supported by an emitter |

For example, the storage predicate contains constraints of the form

$$
\sum_{\rho\in\operatorname{Live}(p,h,t;a,z)}
\operatorname{bytes}(\rho)\le C_h
\quad\forall p,h,t,\qquad
W_{\mathrm{workspace}}^{\mathrm{peak}}(X,a,z)\le W_{\max}.
$$

Here $h$ ranges over each AIC's L1/L0 memories and each AIV's UB;
the available physical core limits apply separately. The coverage
predicate includes section 4's owner variables and all required
reductions. Buffer-slot release and engine ordering then become graph
edges in the timeline constraints above.

A family can be an output-tiled full-K kernel, a deterministic
Split-K kernel with a specified merge, or another concrete loop and
buffer program. Naming a family without an emitter and legality checker
does not add it to $\mathcal A_{\mathrm{emit}}$. To support every
*feasible* shape in a declared dtype/layout/stride contract, first
provide a tail-aware full-K fallback that streams K through bounded
tiles. Specialized families may then compete with it. Shape alone
does not specify the accumulator type, layouts, strides, numerical
rules, available memory, or timing assumptions needed to rank plans.

Equivalently, implementation selection is an outer discrete choice:

$$
T^\star(X,H)=
\min_{a\in\mathcal A_{\mathrm{emit}}(X,H)}
\ \min_{\substack{z\in\mathcal D_a(X,H)\\(a,z)\in\mathcal L(X,H)}}
T_{\mathrm{timeline}}(X,(a,z);H).
$$

The returned plan must contain both $a$ and $z$: a winning tile tuple
without the loop, buffer, mapping, and reduction program is not an
executable tiling strategy.

#### 10.1.1 Instantiation for `mat_mul_v3` on 910B3

For the supplied source, instantiate $\mathcal A_{\mathrm{emit}}$ from the
compiled tiling-key dispatch and its dtype, layout, transpose, bias, and
numerical gates. The legacy 910B3 path has eight automatically selectable
`(LOADMODE, SPLITCOREMODE)` pairs and one bank-only atomic pair. These are
implementation-family keys, not nine complete tilings: each legal pair may
have different core geometry, L1/L0 tiles, buffer depths, L2 blocking, and
loop order. Do not form an unconstrained Cartesian product of key bits and
tile sizes; include only combinations the corresponding kernel body can
execute. The [source inventory](mat_mul_v3_plan_inventory.md) records the
dispatch pairs and their current selector gates.

Use a complete candidate $x=(q,t,l,c,o)$: $q$ is the compiled tiling key;
$t$ is the finalized `TCubeTiling` with `usedCoreNum`, `singleCoreM/N/K`,
`baseM/N/K`, steps, depths, buffering, and `iterateOrder`; $l$ is the L2
blocking/cache policy; $c$ is the input-conversion and output path; and
$o$ is the executable task/reduction/synchronization order. This candidate
also carries the derived workspace and launch block dimension. Its final
serialized tiling data, not the selector's intermediate `runInfo_`, is the
source of truth for validation and timing. The optimizer must check
compilation, exact coverage, numerical mode, memory, workspace, and kernel
support before evaluating $T_{\mathrm{timeline}}$.

The existing `DoSelectTiling` picks the first successful family, while a
knowledge-bank hit can bypass that selection. Integrate the optimizer after
the common seed is built and before the winning plan is serialized. Generate
and finalize candidates independently, retain the existing plan as a safe
incumbent, and commit only the winner. Preserve knowledge-bank early returns
initially; bring them into selection only after bank plans can pass the same
candidate-local validation. The detailed source changes and
acceptance gates are in the
[`mat_mul_v3` implementation plan](MAT_MUL_V3_IMPLEMENTATION_PLAN.md).

### 10.2 Exact optimization over implementation plans

For a complete legal plan and a fixed executable event order, the graph in
section 8 determines its earliest completion time. With fixed service
durations, evaluate it by following the dependency/resource edges in
topological order. The concurrent-rate service alternative in section 9.2
requires its stated rate-allocation evaluation instead. The outer
optimization selects the family and all discrete plan decisions:

```text
partial plan -> branch on family, task tile, resident tile, buffer policy,
                K ownership, loop order, mapping, and core count
             -> complete legal plan
             -> execution graph -> earliest completion time
```

A conceptual C++ evaluator for each complete plan is:

```cpp
double evaluate_plan(const Plan& plan) {
    if (!is_legal(problem, hardware, plan))
        return std::numeric_limits<double>::infinity();

    const auto graph = build_execution_graph(problem, hardware, plan);
    return evaluate_earliest_completion(graph, hardware);
}
```

This is an interface sketch, not an implementation supplied by this folder.
All strategy-dependent work belongs in the plan/graph builder. The scheduler
consumes events and resources without special runtime multipliers for
particular shapes. The evaluator's value is the objective of the declared
mathematical model, not a fitted correction.

For a modest finite domain, enumerate every legal complete plan and take the
minimum. For a larger domain, use branch-and-bound. Let a partial plan $q$
represent its legal completions $\operatorname{Desc}(q)$. Maintain a feasible
complete incumbent $x_I$ with $U=T_{\mathrm{timeline}}(x_I)$. A node bound
$L(q)$ must satisfy

$$
L(q)\le \min_{x\in\operatorname{Desc}(q)}
T_{\mathrm{timeline}}(x).
$$

If $L(q)\ge U$, every completion of $q$ can be pruned. Otherwise branch on
one undecided variable, checking partial legality as early as possible.
Evaluate a leaf only after completing and validating its executable plan.
When no unpruned node remains, $x_I$ is globally optimal **within the
declared finite plan domain and timeline model**. If stopped early, report
the incumbent, the minimum bound on remaining nodes, and their gap. A
candidate-ranking heuristic may choose which node to visit next, but it
must not eliminate nodes without a valid bound.

```text
U = +infinity; best = none; open = {root partial plan}
while open is not empty:
    q = remove a node (preferably with the smallest lower bound)
    if q cannot have a legal completion or L(q) >= U: continue
    if q is complete:
        t = evaluate_plan(q)
        if t < U: U = t; best = q
    else:
        add every legal next-decision branch of q to open
return best, U, and the minimum remaining bound if stopped early
```

One safe, possibly weak, node bound uses compulsory useful Cube work,
minimum compulsory traffic under the stated cache state, and any precedence
chain forced by the partial decisions:

$$
L(q)=\max\!\left(
  \frac{W_{\mathrm{useful}}(q)}{C_{\max}F_{\mathrm{peak,core}}},
  \frac{D_{\mathrm{compulsory}}(q)}{B_{\mathrm{HBM}}},
  T_{\mathrm{forced\ path}}(q)
\right).
$$

Each term must be a lower bound for **every** legal completion, including
the selected family, core count, tail handling, and assumed cache residency.
For example, use zero for the HBM term if the contract permits all operands
to be resident before the timed region. A zero node bound is valid when no
tighter bound has been established. Do not use the exact traffic or critical
path of one guessed completion as a bound for its siblings.

Cache evaluation by the canonical complete plan, workload, and hardware
profile. A cache entry is a previously computed estimate, not a measured
shape-time lookup.

### 10.3 Optimizing schedule order explicitly

When the event set is fixed but ordering is a decision, issue-resource
constraints become disjunctions. For two events on a single issue resource:

$$
s_v\ge s_u+\eta_{u,r}
\quad\lor\quad
s_u\ge s_v+\eta_{v,r}.
$$

A mixed-integer formulation can select the order with binary variables and
valid time bounds. A CP-SAT formulation can use one interval of length
$\eta_{e,r}$ for each event's issue occupancy on resource $r$, enforce
non-overlap on that resource, and separately constrain completion by
$f_e=s_e+\ell_e$ and data dependencies by $s_v\ge f_u$. For a buffer slot,
reserve the slot from its fill until its final consumer releases it; enforce
slot capacity with a cumulative constraint or explicit ownership/order
variables. This is the same scheduling structure as a
[job-shop model](https://developers.google.com/optimization/scheduling/job_shop),
with additional memory, launch, and kernel-program constraints. Integer-time
solvers require a declared time resolution and consistent rounding; an
optimal solver status then certifies only that discretized model.

Completion intervals may overlap on a pipelined resource even though its
issue-occupancy intervals cannot. Buffer live ranges and command credits
provide additional constraints on in-flight work.

The selected order must be executable by the kernel's control program.
Unrestricted scheduling can find an ideal schedule that no available kernel
can produce. One approach is to let the outer search choose among finite
implementable loop and prefetch policies; their resource orders are then
fixed, and the graph evaluator suffices. Use CP-SAT or mixed-integer
scheduling only where the emitter can realize the selected order. A single
integrated solver over all tilings is possible only after giving it a bounded
event universe and encodings for all optional events, changing dependencies,
storage lifetimes, and implementation alternatives. That encoding may be
larger than the direct outer branch-and-bound search.

### 10.4 Safe lower bounds for pruning

For consistent work/traffic assumptions, candidate lower bounds include:

$$
T_{\mathrm{LB}}
=\max\left(
T_{\mathrm{precedence}},
\frac{D_{\mathrm{HBM}}^{\mathrm{physical}}}{B_{\mathrm{HBM}}},
\frac{W_{\mathrm{cube}}}{cF_{\mathrm{peak,core}}}
\right).
$$

$T_{\mathrm{precedence}}$ is a critical path after dropping resource
contention while retaining true dependencies. The compute bound assumes
identical cores and a defined peak rate $F_{\mathrm{peak,core}}$. It can be
weak when few cores are active. Local transfer resources and control issue
capacity can supply further lower bounds.

Do not sum completion latencies to form a pipelined resource bound; overlapping
latencies are not additive service demand. Prune a candidate only using bounds
that are valid for the same model as the objective.

### 10.5 Large workloads

A fully expanded graph can be expensive for large M/N/K. Candidate generation
should be finite and bounded, and the evaluator can memoize identical task
patterns or use a proven periodic recurrence.

Compression must preserve pipeline state: engine availability, outstanding
commands, live operand/accumulator slots, and shared-service interactions.
Multiplying an isolated tile time by the tile count generally loses overlap
and contention. Such multiplication is valid only after establishing the
required independence or steady-state recurrence.

### 10.6 Traffic-guided candidate generation

The paper's section 5 grows native Cube tiles, then chooses L1 buffer
allocation and iteration order using a data-movement metric. Use this as
a **candidate-generation and ranking heuristic** for the full timeline
objective:

1. Start with legal 16-element tile multiples; explore K growth and M/N
   growth while checking L0A/L0B/L0C occupancy and tail behavior.
2. For each Cube tile, enumerate realizable A/B L1 residency, buffer
   depths, and loop orders. A-first and B-first orders can retain different
   operands and change reload counts.
3. Derive the traffic vector
   $(D_{\mathrm{HBM}},D_{\mathrm{L2\to L1}},
   D_{\mathrm{L1\to L0}},D_{\mathrm{Fixpipe}})$ and compute inexpensive
   service-demand bounds. Prioritize promising candidates for complete
   event-graph evaluation.
4. Branch first into regions with promising traffic and bottleneck profiles.
   Preserve all other legal regions until a valid node lower bound permits
   pruning. A heuristic may improve the incumbent early; it does not change
   the exactness of branch-and-bound.

Minimizing the paper's total off-chip-to-on-chip byte count alone is not
equivalent to minimizing elapsed time: unequal path bandwidths, conversion,
Cube utilization, overlap, and multicore contention can reverse the ranking.
Traffic therefore guides evaluation order. Discard a candidate only when a
valid lower bound for that candidate is no better than an existing complete
plan's timeline.

### 10.7 From formulation to a usable optimizer

Build the following artifacts in order. Each one has an observable
completion condition:

1. **Problem contract and safe fallback.** Accept positive $M,N,K$,
   explicit dtypes, layouts, strides, numerical requirements, and
   workspace budget. For `mat_mul_v3`, begin with a declared FP16/BF16 ND
   scope on 910B3. Capture the current selector's complete emitted plan
   as the safe incumbent; preserve its result when optimization is disabled
   or produces no additional valid candidate. Test small, large, and
   nonmultiple dimensions in that scope.
2. **Plan representation and candidate builders.** Add a structured plan
   containing the full finalized tiling data, key, physical AIC/AIV usage,
   workspace, launch dimension, L2 policy, and reduction path. Build
   independent candidates for the existing BASE, ordinary single-core
   Split-K, and deterministic Split-K kernels first. Extend to the other
   compiled families only after their legality and timeline semantics are
   represented. Every candidate must either map to a compiled kernel with
   its stated schedule or fail legality checking.
3. **Primitive timing profile.** Query capacities and available cores on
   the actual 910B3. Supply first-block latency and issue interval,
   transfer service by path/layout/size, conversion, synchronization,
   reduction, and launch costs. Use documentation where sufficient and
   isolated primitive measurements elsewhere; record units and source.
   No required event may silently receive zero cost.
4. **Deterministic evaluator.** Decode a plan, check section 10.1,
   derive exact tail work and traffic, build resource/slot dependencies,
   and evaluate the section 8/9 timeline. Return latency together with
   critical path, stage waits, bytes by level, occupancy peaks, and
   the emitted-plan identifier. Section 11 and the explicit section 12
   schedule are small arithmetic regression cases.
5. **Exact search and device validation.** Enumerate the first finite domains
   exhaustively; then add branch-and-bound with safe node bounds and
   traffic-guided visit order. Report the plan, predicted runtime, number
   of evaluated plans, and either an optimality certificate or remaining
   lower bound and gap.
   Compile and measure the top candidates on representative shapes,
   including long K, large M/N, and tails. Check correctness and whether
   predicted *rankings* identify fast plans; revise primitive service
   assumptions or missing events when they do not.

The current [local search](../local_search/src/main.cpp) has one gene
that selects a reviewed template and calls the 910B2 Python model.
For that existing finite template list, a loop that evaluates every candidate
and chooses the minimum is already an exact solve of its restricted model;
a mathematical-programming solver adds no benefit. The one-index
representation still cannot search independent buffer depths, block tiles,
loop order, or new implementation families. For `mat_mul_v3`, first
enumerate and score its independently finalized, legal candidates in C++.
Then broaden the independent tile and buffer domains and use
branch-and-bound when exhaustive online enumeration exceeds the host-time
budget. The [implementation plan](MAT_MUL_V3_IMPLEMENTATION_PLAN.md)
specifies the refactor, family order, tests, and rollout gates.
The [standalone C++ driver](standalone_optimizer/README.md) now implements
both finite-candidate ranking and a two-level implementation/tiling solver,
with fixed-resource event-graph evaluation. It
derives event durations from shared primitive timing parameters and each
candidate's work amounts. It does not yet supply a measured 910B3 timing
profile. Its optional reduced MILP optimizes an M/N block grid for
BASE and deterministic Split-K without scoring every complete tiling, then
evaluates selected plans with the event graph when expansion fits its cap. A typed
standalone preflight now derives key tuples, workspace, and launch fields for
an aligned subset including A ND→NZ conversion on `[16,16,102400]`; CANN
still must finalize kernel tilings.
Its initial BASE/deterministic Split-K event builder derives
logical work from shape and tiling for a narrow FP16/BF16 ND, NN contract;
cache and pipeline behavior still needs source-accurate modeling and device
validation.

An exhaustive search or branch-and-bound with valid pruning identifies the
minimum of the declared finite model domain. A time-limited solve gives a
feasible incumbent and a bound, not an optimality certificate. Neither
result guarantees the fastest physical kernel when primitive timings or
cache behavior are uncertain; measured top candidates close that loop.

## 11. Worked timeline: one versus two buffers

This example uses hypothetical service times to demonstrate the equations.
It does not claim Ascend hardware timing.

Three independent output-tile jobs each have one K update and four stages:

| Stage | Meaning | Latency and initiation interval |
|---|---|---:|
| D | DMA into L1 | 2 us |
| L | L1-to-L0 load | 1 us |
| C | Computation | 4 us |
| W | Output writeback | 1 us |

The four engines are distinct and nonpipelined. Each processes jobs in order
0, 1, 2. All L1, L0, and accumulator rings have the same depth $d$. There
is no additional shared-memory interference in this example.

Let finish times with negative job indices be zero. The recurrences are

$$
s_{D,j}=\max(f_{D,j-1},f_{L,j-d}),\qquad
f_{D,j}=s_{D,j}+2,
$$

$$
s_{L,j}=\max(f_{D,j},f_{L,j-1},f_{C,j-d}),\qquad
f_{L,j}=s_{L,j}+1,
$$

$$
s_{C,j}=\max(f_{L,j},f_{C,j-1},f_{W,j-d}),\qquad
f_{C,j}=s_{C,j}+4,
$$

$$
s_{W,j}=\max(f_{C,j},f_{W,j-1}),\qquad
f_{W,j}=s_{W,j}+1.
$$

The cross-stage dependencies enforce buffer reuse: an L1 slot waits for its
old load, an L0 slot waits for its old computation, and an accumulator slot
waits for its old writeback.

| Buffer depth | Job | DMA interval | Load interval | Compute interval | Write interval |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0–2 | 2–3 | 3–7 | 7–8 |
| 1 | 1 | 3–5 | 7–8 | 8–12 | 12–13 |
| 1 | 2 | 8–10 | 12–13 | 13–17 | 17–18 |
| 2 | 0 | 0–2 | 2–3 | 3–7 | 7–8 |
| 2 | 1 | 2–4 | 4–5 | 7–11 | 11–12 |
| 2 | 2 | 4–6 | 7–8 | 11–15 | 15–16 |

All intervals are in microseconds. The resulting makespans are

$$
T(d=1)=18\ \mathrm{us},\qquad T(d=2)=16\ \mathrm{us}.
$$

The total event-duration sum is 24 us in both cases. Buffer depth changes the
legal overlap, which changes completion time despite identical arithmetic and
transfer work.

For a separate illustration of initiation interval, take two independent
events on one engine, each with latency 5 us and interval 2 us. Their earliest
starts are 0 and 2 us, and their finishes are 5 and 7 us. Waiting for the first
completion before issuing the second would incorrectly predict 10 us for
this specified pipeline.

## 12. Applying the formulation to the discussed shapes

### 12.1 Deterministic Split-K on 910B3

```text
shape:                   M=16, N=16, K=102400
physical AIC workers:    20 (910B3 reference)
base_m/base_n/base_k:    128/128/128
step_k:                  2
K task span:             128 * 2 = 256
```

The existing recipe supplies nominal $\mathrm{singleCoreM}=384$. Its current
small-M/N event builder uses actual task dimensions $M=16,N=16$, and splits
K into

$$
G_K=\left\lceil\frac{102400}{256}\right\rceil=400
$$

chunks. A balanced partition onto the 20 B3 workers gives

$$
400=20\cdot20:
$$

each worker handles 20 chunks, or 40 base-K iterations.

If the selected FP16 kernel emits 16-by-16-by-128 Cube tiles without
padding M and N to 128, section 7.1 gives
$b_e=(16/16)(16/16)(128/16)=8$ native blocks. Each K chunk has two
such tiles, or 16 blocks; under this assumption the whole operation has
$400\times16=6400$ blocks and
$6400\times8192=52{,}428{,}800$ Cube FLOPs before any extra reduction or
conversion work. A balanced 20-AIC assignment has 320
native blocks per AIC. Its ideal Cube issue demand under the conditional
one-cycle interval is $320/1800\approx0.178$ us; operand transfers,
chunk boundaries, startup, dependencies, and reduction determine the
actual timeline. A backend that pads to larger Cube tiles must count those
emitted blocks instead. Use the event graph to expose the waits.

Under this implementation's per-core partial-accumulation policy, the final
reduction combines 20 per-core partial outputs. It does not combine 400
separately materialized final partial outputs. Intermediate writes/atomic
updates still need to be counted according to the actual program.

The user's original request used 24 cores, matching the existing 910B2
model context. That historical partition was $400=16\cdot17+8\cdot16$,
with 24 final per-core partials. It cannot represent 24 simultaneously
active AICs on a 20-AIC B3. Retargeting the graph changes ownership,
reduction work, contention, and the hardware profile; scaling the old
predicted time by $24/20$ is insufficient. This section describes the
proposed B3 schedule, not an already implemented B3 mode in the Python API.

The model must include any required input conversion, chunk control, operand
transfers, matrix operations, partial updates, reduction, and final cast/write.
A generalized optimizer can explore different task sizes only when its
selected implementation also supports the corresponding execution logic.

#### Worked event schedule for the small output

To make the timeline calculation explicit, consider one **proposed** B3
implementation with contiguous ownership
$h(p,j)=20p+j$, where $p=0,\ldots,19$ is an AIC and
$j=0,\ldots,19$ is its local K-chunk index. Each core updates one
16-by-16 FP32 accumulator across its 20 chunks, writes one FP32 partial,
and a reducer combines the 20 partials in a fixed order before casting
the 16-by-16 answer to FP16. This fixed order makes the reduction
deterministic. It is a concrete modeling choice, not a claim that the
existing 910B2 kernel implements this B3 schedule.

Assume each base-K iteration operates on the actual 16-by-16 output
extent and does not pad M or N to 128. Each iteration loads an
$A[16,128]$ packet and a $B[128,16]$ packet:

$$
D_A^{(128)}=D_B^{(128)}=16\cdot128\cdot2=4096\ {\rm bytes}.
$$

Two iterations per chunk give $D_{\rm input,chunk}=4\cdot4096=16384$
bytes. If each such packet is fetched once from HBM, the 400 chunks
request $6{,}553{,}600$ input bytes in total. For this schedule, two
resident L1 chunk slots for each operand consume 32 KiB together.
Four 4-KiB base-K slots for each operand hold two complete chunks,
using 16 KiB in L0A and 16 KiB in L0B. The one live FP32 accumulator
consumes 1 KiB in L0C. These footprints fit
the section 2.3 capacities. The 20 FP32 partials occupy 20 KiB of
global workspace; the final FP16 output is 512 bytes.

Using only the nominal B3 HBM and Cube peak rates, two elementary
lower bounds are

$$
T_{\rm HBM,input}^{\rm ideal}
=\frac{6{,}553{,}600}{1{,}600{,}000}
=4.096\ {\rm us},
\qquad
T_{\rm Cube}^{\rm ideal}
=\frac{52{,}428{,}800}{294{,}910{,}000}
\approx0.1778\ {\rm us}.
$$

Thus $T_{\rm timeline}\ge4.096$ us for this cold-input traffic
assumption, even before counting partial stores, reduction, launch
work, or conversion. Neither bound is an end-to-end estimate.

Here is an event recurrence that turns the work into a timeline.
Let $H_{p,j}$ be completion of the chunk's shared HBM input service.
Let $F^{(2)}_{p,j}$, $F^{(1)}_{p,j}$, and $F^C_{p,j}$ be completion of
MTE2 delivery/conversion into L1, MTE1 delivery/conversion into L0,
and both Cube iterations. For an illustrative **nonpipelined**
per-core MTE2 and MTE1, two-chunk L1 and L0 rings, and a completion
dependency between updates of the same accumulator:

$$
\begin{aligned}
F^{(2)}_{p,j}
 &=\max\!\left(H_{p,j},F^{(2)}_{p,j-1},
                   F^{(1)}_{p,j-2}\right)+d_2,\\
F^{(1)}_{p,j}
 &=\max\!\left(F^{(2)}_{p,j},F^{(1)}_{p,j-1},
                   F^C_{p,j-2}\right)+d_1,\\
F^C_{p,j}
 &=\max\!\left(F^{(1)}_{p,j},F^C_{p,j-1}\right)+d_C .
\end{aligned}
$$

Negative-index finish times are zero. The $j-2$ terms release the L1
chunk slot after its MTE1 read and the L0 pair of base-K slots after
its Cube read.
$d_2$ and $d_1$ include the relevant layout transformations.
This schedule holds the accumulator in L0C through all 20 chunks.
An implementation with spills, atomics, different issue intervals,
or more command dependencies needs corresponding extra events.

To evaluate these equations numerically, make the following **illustrative
assumptions**. Initial launch, scalar command, and descriptor overheads
are set to zero. Only the 1.6 TB/s shared HBM rate and 1.8 GHz clock
come from section 2.3; the other timings below are placeholders for
primitive measurements on the intended kernel.

| Quantity | Illustrative value or rule |
|---|---:|
| HBM input arbitration | 20 active cores receive equal shares of 1,600,000 bytes/us; no HBM startup |
| MTE2 L2-to-L1 delivery and ND-to-Nz conversion, $d_2$ | 0.020 us per chunk |
| MTE1 L1-to-L0 delivery and operand conversion, $d_1$ | 0.030 us per chunk |
| Cube first-block latency / block interval | 16 cycles / 1 cycle |
| Fixpipe service for one partial from each core | 0.020 us, concurrently |
| Shared HBM service for all 20 partial stores | $20\cdot1024/1{,}600{,}000=0.0128$ us |
| Synchronization or reducer launch | 0.050 us |
| L2 read of all 20 partials | 0.020 us; assumes they are L2 hits |
| Vector reduction of the partials | 0.080 us |
| Final conversion and 512-byte write | 0.030 us |

Each core requests 16,384 bytes per chunk. Its equal HBM share is
80,000 bytes/us, so the HBM service interval is

$$
\Delta_H=\frac{16384}{1600000/20}=0.2048\ {\rm us},
\qquad H_{p,j}=(j+1)\Delta_H.
$$

Each 16-by-16-by-128 Cube iteration has eight native blocks.
Taking the two iterations serially on the same accumulator gives

$$
d_C=\frac{2\left(16+(8-1)\cdot1\right)}{1800}
=\frac{46}{1800}\approx0.025556\ {\rm us}.
$$

The input interval exceeds each local stage's service time, and the
two-chunk buffer reuse constraints never bind in this particular schedule.
Consequently, the recurrences evaluate to:

| Local chunk $j$ | HBM input finish | MTE2 finish | MTE1 finish | Cube finish |
|---:|---:|---:|---:|---:|
| 0 | 0.204800 | 0.224800 | 0.254800 | 0.280356 |
| 1 | 0.409600 | 0.429600 | 0.459600 | 0.485156 |
| 19 | 4.096000 | 4.116000 | 4.146000 | 4.171556 |

For example, the Cube waits after chunk 0 until the next operands
arrive: $0.459600-0.280356\approx0.179244$ us of exposed
operand-readiness stall. The same gap occurs between later chunks
under these assumptions. All 20 cores follow the same timeline, so
the last Cube result is ready at approximately 4.171556 us.

The tail of this explicitly serialized reducer schedule is

$$
\begin{aligned}
T_{\rm example}
&=\max_p F^C_{p,19}
 +d_{\rm Fixpipe}
 +\frac{20\cdot1024}{B_{\rm HBM}}
 +d_{\rm sync}
 +d_{\rm L2read}
 +d_{\rm reduce}
 +d_{\rm final}\\
&\approx4.171556+0.020+0.0128+0.050+0.020+0.080+0.030\\
&\approx\boxed{4.384356\ {\rm us}}.
\end{aligned}
$$

The 4.384356 us value is a **worked calculation for the declared
hypothetical primitive timings**, not a 910B3 prediction or a measured
runtime. A different L2 residency, HBM arbitration policy, instruction
latency, chunk implementation, or reduction tree changes the event
times. In particular, a cache miss while reading the partials adds
another HBM service request. For an actual estimate, replace the
illustrative timings with target-specific primitive parameters and
evaluate the complete event graph from section 8.

### 12.2 Large output dimensions

For $M=102400,N=30720,K=4096$, two hypothetical full-K task geometries give:

| $S_M,S_N$ | $G_M,G_N$ | Output tasks | Tasks/AIC under balanced mapping to 20 B3 AICs |
|---|---|---:|---:|
| 128, 256 | 800, 120 | 96,000 | 4,800 |
| 256, 512 | 400, 60 | 24,000 | 1,200 |

For FP16 inputs and output, the useful arithmetic is
$2MNK=25{,}769{,}803{,}776{,}000$ FLOPs. An ideal read-once input
and write-once output path moves at least
$2(MK+KN+MN)=7{,}381{,}975{,}040$ HBM bytes. The nominal B3 rates
therefore give

$$
T_{\rm Cube}^{\rm ideal}
=\frac{25{,}769{,}803{,}776{,}000}{294{,}910{,}000}
\approx87{,}382\ {\rm us},\qquad
T_{\rm HBM}^{\rm ideal}
=\frac{7{,}381{,}975{,}040}{1{,}600{,}000}
\approx4{,}614\ {\rm us}.
$$

These are lower bounds under ideal reuse and full-chip utilization.
For either task geometry, construct each assigned AIC's ordered
MTE2/MTE1/Cube/Fixpipe events and all cores' shared HBM/L2 requests,
then take the latest final-output completion. The 4,800 or 1,200
tasks per AIC determine repetition counts; they cannot simply be
multiplied by an isolated task duration because transfers and Cube
work can overlap across tasks and cores.

The task counts and bounds do not establish which plan is faster or legal.
The larger task can execute through the same small L1/L0
tiles, but its reuse benefit depends on the emitted loop nest and retained
operands. It can also change control costs, working-set behavior, and task
ordering.

This shape is outside the current public calibrated model's supported domain.
The generalized mathematical model has no intrinsic M/N=128 bound, but its
backend, integer widths, workspace requirements, service assumptions, and
graph-evaluation scalability still need to support the selected plan.

## 13. Relationship to the existing repository

The separate [910B3 `mat_mul_v3` plan inventory](mat_mul_v3_plan_inventory.md)
maps the supplied kernel source's implementation keys and selector gates to
the optimization domain in section 10.
The [integration plan](MAT_MUL_V3_IMPLEMENTATION_PLAN.md) maps that domain
to candidate generation, legality checks, analytical timeline scoring, and
selection inside the host tiler.

The existing model already contains useful scheduling mechanisms:

- [EventGraph](../cost_model/matmul_cost_model/_engine/costmodel/matmul_four_path_v301.py):
  separate issue/completion clocks and longest-path evaluation.
- [Four-path graph construction](../cost_model/matmul_cost_model/_engine/costmodel/matmul_four_path_v590.py):
  task construction, engine dependencies, operand-slot release, and output
  handling.
- [Current plan recipe](../cost_model/matmul_cost_model/_engine/matmul_reviewed_tiling_recipe_v2314.py):
  fixed construction rules for supported native plans.
- [Current plan validation](../cost_model/matmul_cost_model/_engine/matmul_static_unified_v2342.py):
  template, shape, and complete-plan checks.
- [Input schema](../cost_model/docs/INPUT_SCHEMA.md) and
  [model formula/scope](../cost_model/docs/MODEL_AND_SCOPE.md):
  current public contracts and calibration boundaries.

The current public API supports:

```python
prediction = model.predict(config, semantic=True)
runtime_us = prediction["predicted_us"]
```

The existing C++ wrapper can select the same mode:

```cpp
double runtime_us = model.predict(config, true).predicted_us;
```

The local-search executable also accepts `--semantic`. For the original
shape and 24-core deterministic configuration, the existing **910B2**
model's semantic outputs are:

| step_k | Current semantic prediction |
|---:|---:|
| 1 | 93.6224856803095 us |
| 2 | 93.20267861374995 us |
| 3 | 96.18641785485626 us |

These are historical 910B2 model outputs. They are neither 910B3 predictions
nor device measurements. Updating this document does not retarget the
existing Python model's calibrated hardware coefficients.

`semantic=True` removes external residual corrections but keeps fixed,
calibrated effective service coefficients and the current restrictive plan
rules. It does not expose arbitrary block sizes, buffer counts, or schedules.
Some graph policies are also fixed by implementation; for example, the current
deterministic graph overrides its nominal L1 slot counts with a schedule
hypothesis. Nominal plan depth fields cannot automatically be interpreted as
independently searchable pipeline slots.

Generalizing the optimizer therefore requires a plan generator and event
builder for the intended kernel families, together with explicit hardware
parameters. Weakening the existing validator alone does not provide that
behavior.

## 14. Suggested implementation boundary

Keep four components separate:

1. **Problem and hardware profile:** immutable shape, layouts, numerical
   contract, resource capacities, and primitive timing assumptions.
2. **Plan generator:** produces supported block/memory tiles, buffer policies,
   implementation choices, and task mappings.
3. **Execution graph builder:** derives concrete work, ownership, buffer
   lifetimes, command ordering, and shared-resource requests.
4. **Timeline evaluator and optimizer:** computes completion times and searches
   over the legal plans.

The evaluator should return more than a scalar during development:

```text
runtime_us
event start/finish times and resource identities
critical path
active cores and tasks per core
logical and physical bytes by memory level
peak live storage by memory level
workspace and reduction work
hardware-profile and schedule-policy identifiers
```

These outputs make a predicted improvement inspectable. For example, a better
buffering choice should show the specific waits it eliminates.

Before using a new kernel family for optimization, verify task coverage,
reduction ownership, buffer lifetime constraints, and emitted operation
counts. The worked timeline provides a small deterministic reference case.
Compare larger predictions with actual device timelines when available;
disagreements should lead to corrections in work, dependencies, or primitive
service assumptions, rather than an added whole-kernel residual.

## 15. Paper-derived refinements and scope

The source is [Zhang et al., *NPUMeter: Automatic Operator Optimization for
Ascend NPU with Accurate Analytical Performance Models*](3820380.pdf),
ACM TACO 23(2), 2026, DOI 10.1145/3820380. This formulation uses its
architectural decomposition and pipeline ideas as follows:

| Paper idea | Formulation here |
|---|---|
| Section 4.2, Equation (1): 16-by-16-by-16 blocks with separate first-block latency and issue interval | Section 7.1 supplies an explicit block count and tile service equation |
| Section 4.3, Equations (7)–(11): transfer paths and layout conversion | Sections 6.4 and 7.2 track bytes and conversion service for MTE2, MTE1, and Fixpipe |
| Section 4.4, Figures 8–10: tile stalls and Cube-to-Vector overlap | Sections 8.5 and 8.6 encode arrival, publication, and buffer-release dependencies |
| Section 5: grow legal tiles and prioritize memory-efficient loop/buffer choices | Section 10.6 uses traffic to guide search, then scores complete timelines |

The paper evaluates an Ascend **910B-series single AI core** and obtains
several primitive parameters through profiling. It does not provide a
validated full-device 910B3 timeline for the shapes in section 12.
Its particular block latencies, vector intervals, achieved bandwidths, and
conversion coefficients therefore require measurement or verification on the
target 910B3. Section 2.3 supplies the nominal B3 reference profile, while
section 7 keeps these primitive timing terms explicit.

For a multicore or Split-K plan, calculate cache misses and transfers from
actual ownership, loop order, residency, and capacity. The paper's compact
single-core transfer/stall formulas are useful special cases; the event
graph also handles tails, overlap, shared HBM/L2 contention, and reduction.
