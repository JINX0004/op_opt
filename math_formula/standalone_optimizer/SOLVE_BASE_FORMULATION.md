# `SolveBase`: mathematical formulation and worked example

This document describes the BASE branch of [`SolveReducedMip`](mip_solver.h),
as implemented by [`SolveBase`](mip_solver.cpp). It chooses the outer
`singleCoreM/N` extents, inner `baseM/N/K` tiles, and occupied Cube-core count
for one fixed matrix multiplication workload. Its objective is the current
**reduced service-time estimate** in microseconds. It is not the full
event-graph makespan or a measured 910B3 runtime. A
[910B3 probe and controlled-tiling pilot](bench910/PRIMITIVE_910B3_CANN9.md)
found that a serial sum of packet latencies over all output tiles
overpredicted kernel time substantially. The current objective charges
packet startup to the busiest assigned AIC instead. Its rates, overlap
assumptions, and absolute runtime are still exploratory.

## Inputs, derived counts, and decisions

The input workload is $(M,N,K)$, with $A$ allocated Cube cores. The
implementation forms a finite spatial grid from `MipTileDomain`. A geometry
is $g=(C_M,C_N,B_M,B_N)$, where $C_M,C_N$ are `singleCoreM/N` and $B_M,B_N$
are `baseM/N`. Each component is a positive, 16-aligned integer within its
corresponding shape axis, and $B_M\le C_M$, $B_N\le C_N$. Empty domain
vectors use the defaults described in [the optimizer README](README.md#reduced-milp-with-mn-block-tiling).

For a fixed geometry, let

$$
\begin{aligned}
N_o(g)&=\left\lceil\frac M{C_M}\right\rceil
        \left\lceil\frac N{C_N}\right\rceil,\\
N_M(g)&=\left\lfloor\frac M{C_M}\right\rfloor
        \left\lceil\frac{C_M}{B_M}\right\rceil
       +\left\lceil\frac{M\bmod C_M}{B_M}\right\rceil,\\
N_N(g)&=\left\lfloor\frac N{C_N}\right\rfloor
        \left\lceil\frac{C_N}{B_N}\right\rceil
       +\left\lceil\frac{N\bmod C_N}{B_N}\right\rceil,\\
S(g)&=N_M(g)N_N(g),\\
D_A(g)&=2KMN_N(g),\qquad D_B(g)=2KNN_M(g),\\
D(g)&=D_A(g)+D_B(g),\\
O&=MN,\qquad F=2MNK.
\end{aligned}
$$

$N_o$ is the number of outer output tiles. $N_M,N_N$ count inner base
packets along each axis, including an outer-tile tail; $S$ is their product.
$D_A,D_B$ are the separate modeled A/B GM read requests in bytes for
FP16/BF16 inputs under the present no-cache traffic assumption. They are
not measured HBM misses: L2 reuse can make physical HBM traffic smaller.
$O$ is the number of output
elements and $F$ is the total floating-point work.

The full discrete decisions are a geometry $g$, $b=\text{baseK}/16$, and
occupied core count $c=\text{usedCoreNum}$. BASE fixes
`singleCoreK = K` and `stepK = 1`. In the **actual master MILP**, only
one binary $y_g$ per feasible geometry remains a variable. The code
computes the best $b$ and $c$ for each geometry before constructing that
MILP.

## Objective

Let $R_C$ be FLOP/µs per Cube core, $R_A,R_B,R_W$ the modeled aggregate A
read, B read, and GM write byte rates, and $\lambda_A,\lambda_B,\lambda_W$
their fixed per-operation latencies. Define $P_B(b)=\lceil K/(16b)\rceil$. Index the
outer output tiles by row $i$ and column $j$, with actual extents $m_i,n_j$
(including tails). The reduced BASE objective uses a source-inspired
**balanced-class abstraction**, not an exact kernel block schedule. It has
at most four tile classes $t$: full/full, full/N-tail, M-tail/full, and
M-tail/N-tail. Their multiplicities are, respectively,
$f_Mf_N$, $f_M\mathbf1_{r_N>0}$,
$f_N\mathbf1_{r_M>0}$, and
$\mathbf1_{r_M>0}\mathbf1_{r_N>0}$, where
$f_M=\lfloor M/C_M\rfloor$, $r_M=M\bmod C_M$ and analogously for N.
Write $H_t$ for the class multiplicity and $(m_t,n_t)$ for its tile
extents. Class $t$ assigns $\lfloor H_t/c\rfloor$ tiles to every core and
one extra to $H_t\bmod c$ consecutive cores. The extra-tile window starts
at $\phi_t=(\sum_{h<t}H_h)\bmod c$. Thus, with
$x_{qt}=\lfloor H_t/c\rfloor+
\mathbf1_{((q-\phi_t)\bmod c)<(H_t\bmod c)}$, define

$$
E_q=\sum_t x_{qt}m_tn_t,\qquad
S_q=\sum_t x_{qt}\left\lceil\frac{m_t}{B_M}\right\rceil
                       \left\lceil\frac{n_t}{B_N}\right\rceil,\qquad
O_q=\sum_t x_{qt}.
$$

These satisfy $\sum_q E_q=MN$, $\sum_q S_q=S$, and
$\sum_q O_q=N_o$. All Cube cores use the same primitive rates; only
their assigned workloads differ. The four-class calculation costs $O(c)$
independently of the number of output tiles. It approximates the source's
rotation of surplus work but does not include L2 region order or cache
residency. The separate explicit event builder still uses its own
row-major round-robin mapping for diagnostic scoring. The candidate
objective is

$$
\boxed{\displaystyle
T_B(g,b,c)=L+\max\!\left\{
\underbrace{\max_{0\le q<c}\frac{2K E_q}{R_C}}_{T_{\rm Cube}},
\underbrace{\frac{D_A(g)}{R_A}+\frac{D_B(g)}{R_B}+\frac{2O}{R_W}
 +\max_{0\le q<c}\left(S_qP_B(b)(\lambda_A+\lambda_B)+O_q\lambda_W\right)}_{T_{\rm GM}}
\right\}.}
$$

The GM expression represents one modeled operand-read pair per spatial
base packet and K packet, plus an output write per outer tile. Fixed
packet latency follows the busiest AIC rather than a serial sum across
all AICs; aggregate bytes still use the modeled device rates. The factor
$2O$ is the modeled output byte count. Cube and GM service are combined
with `max` as a throughput approximation; the equation does not resolve
event precedence, transfer overlap, buffering, cache reuse, or
shared-device contention in detail.

`gm_read_a` and `gm_read_b` are an optional **pair** of `bytes` primitives
for BASE. Both must be present, with nonnegative finite startup and a
positive finite rate; otherwise neither may be present. When absent,
$R_A=R_B=R_R$ and $\lambda_A=\lambda_B=\lambda_R$ from the existing
`gm_read` primitive, exactly reproducing the previous objective. Split-K,
the A ND→NZ workspace conversion, and the separately reported explicit
event-graph score still use `gm_read`. The A/B pair requires independent
operand-transfer measurements before it should guide device selection.
No orientation coefficient is inferred from the seven whole-kernel results.
An [independent 910B3 operand-shaped probe](bench910/directional_read/README.md)
does resolve the sign of the 128×256 versus 256×128 tie. Its isolated-read
rates overpredict the full-kernel gap, so they remain diagnostic rather than
validated parameters of this objective.

### Optional source-step MTE2 packet objective

For the checked plain BASE source branch, a separate exploratory objective
uses the actual L1 K-step transfer geometry instead of one baseK-slice
latency. Supply all four `base_mte2_a`, `base_mte2_b` (`bytes`) and
`base_mte2_a_rows`, `base_mte2_b_rows` (`rows`) primitives, and set
`base_core_equals_base=true`. The A/B byte primitives each contain a
per-packet fixed time; row primitives have zero fixed time. The solver
derives `stepKa/stepKb` from source L1-depth arithmetic and 256/512-byte
alignment. It rejects an incomplete profile or non-source-compatible
BASE geometry.

For assigned AIC $q$, define $U^A_q$ as the sum of actual M extents over
its spatial base packets and $U^B_q$ as the corresponding sum of N extents.
Let $h_A,h_B$ be the source L1 step sizes and
$P_A=\lceil K/(h_A\,baseK)\rceil$,
$P_B'=\lceil K/(h_B\,baseK)\rceil$. With packet fixed times
$\alpha_A,\alpha_B$, per-AIC requested-byte rates $r_A,r_B$, and ND-row
rates $q_A,q_B$, the modeled per-AIC read service is

$$
R_q=S_q(P_A\alpha_A+P_B'\alpha_B)
 +\frac{2KU^A_q}{r_A}+\frac{2KU^B_q}{r_B}
 +\frac{P_AU^A_q}{q_A}+\frac{KS_q}{q_B}.
$$

The packet objective replaces $T_B$ by

$$
T_B^{\rm packet}=L+\max_q\max\left\{
\frac{2KE_q}{R_C},
R_q+\frac{2MN}{R_W}+O_q\lambda_W\right\}.
$$

Its serial A/B service and `max` overlap are still a reduced estimate;
MTE1 and source Matmul engine dependencies remain unresolved. The
[source-step probe and held-out packet check](bench910/l1_step_read/README.md)
show the correct 128×256 preference on all seven checked whole-kernel
shapes, with at most 4.49% absolute runtime error for that tile pair. One
held-out primitive packet shape still has an 18.3% slope error. Therefore
this profile is opt-in and is not the default parameter set for arbitrary
tile geometries or implementations.
An earlier [source-backed selection experiment](bench910/PACKET_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
using the row-major assignment found a tail-heavy shape on which that packet objective selected a tile
15.94% slower than the MatMulV3 default. The source L2/core assignment
is abstracted here by balanced full/tail classes. A
[new device rerun](bench910/abstract_balance_three_v1/README.md) on three
previously regressing shapes still found the chosen 256×128 tile 6.1–13.9%
slower than the default. Thus the abstraction has not been validated as a
general BASE selector.

$L$ is the part independent of tiling. In the code it is the launch fixed
latency, plus an A ND→NZ conversion when $K>65535$:

$$
L=\lambda_{\rm launch}+
\mathbf1_{K>65535}\left(
\lambda_R+\frac{2MK}{R_R}
+\lambda_{\rm ND\to NZ}+\frac{MK}{R_{\rm ND\to NZ}}
+\lambda_W+\frac{2MK}{R_W}+\lambda_{\rm sync}
\right).
$$

The conversion term and its workspace allocation depend on the original
shape, not the geometry. `SolveBase` requires zero fixed Cube latency,
because that latency is absent from this reduced objective.

For $(M,N,K)=(2048,2048,4096)$ with source-compatible tiles,
$(B_M,B_N)=(128,256)$ gives $(D_A,D_B)=(128,256)$ MiB, whereas
$(256,128)$ gives $(256,128)$ MiB. Both total 384 MiB. Thus a generic read
rate forces a tie; independently measured $R_A,R_B$ can distinguish them.
These byte counts still omit ND burst width/stride, L1 depths, Fixpipe row
shape, L2 order, and the distinct L1→L0A/L0B MTE1 paths, which are
separate later model steps. The
[diagram-based MTE1 orientation check](bench910/HUAWEI_DIAGRAM_RATE_COMPARISON.md)
favors 256×128 on nominal stage service, opposite the measured whole-kernel
winner; overlap and A/B buffering must be resolved before adding that term.

## Constraints and local elimination

Let $C_{0A},C_{0B},C_{0C},C_1$ be the modeled L0A, L0B, L0C, and L1
capacities in bytes, and let $d_C\in\{1,2\}$ be the configured BASE L0C
buffer count (`CoreMemory::base_l0c_buffers`, default 2). The BASE
feasibility conditions used to form a local
geometry cost are

$$
\begin{aligned}
1\le b&\le K/16, & 1\le c&\le\min(A,N_o(g)),\\
64B_Mb&\le C_{0A}, &64B_Nb&\le C_{0B},\\
64(B_M+B_N)b&\le C_1, &4d_CB_MB_N&\le C_{0C},\\
W_0&\le W_{\max},
&W_0&=20\,\mathrm{MiB}+\mathbf1_{K>65535}(2MK).
\end{aligned}
$$

The L0A/L0B and L1 inequalities are the code's double-buffer footprints
after substituting `baseK = 16b`; L0C stores $d_C$ FP32 accumulator tiles.
`SolveBase` first rejects the family if $W_0>W_{\max}$. It then discards a
geometry if L0C fails or no positive $b$ or $c$ exists. Geometric alignment,
$B_M\le C_M$, and $B_N\le C_N$ are enforced when the grid is built.

For the evaluated plain MatMulV3 BASE key `0x10000`, full-output device
checks showed that `singleCoreM>baseM` left the second M base tile zero,
and `singleCoreN>baseN` likewise left the second N base tile zero. The
source-compatible study therefore sets
`MipTileDomain::base_core_equals_base=true`, imposing $C_M=B_M$ and
$C_N=B_N$ for BASE. It sets $d_C=1$ because the source default emits a
single-buffer L0C tile of size $128\times256$; the conservative standalone
default remains $d_C=2$. These source-specific constraints are optional
for the mathematical domain, but required for the evaluated plain BASE
load mode; they do not change the separate Split-K domain. A separate
[B-in-L1 full-load path](bench910/FULL_LOAD_CORE_TILING_SUPPORT_910B3.md)
supports larger single-core output regions and needs its own objective
and eligibility constraints.

For the **generic byte-only objective**, at fixed feasible $g$ and $c$,
increasing $b$ cannot increase $T_B$:
it only reduces or preserves the packet count $P_B$. The balanced-class
assignment changes with $c$, and unequal tail tiles can make the busiest
AIC's load nonmonotone. The solver therefore evaluates each feasible
integer $c$ after selecting the largest feasible $b$:

$$
\begin{aligned}
b_g^*&=\min\!\left(
\frac K{16},
\left\lfloor\frac{C_{0A}}{64B_M}\right\rfloor,
\left\lfloor\frac{C_{0B}}{64B_N}\right\rfloor,
\left\lfloor\frac{C_1}{64(B_M+B_N)}\right\rfloor
\right),\\
c_g^*&\in\underset{1\le c\le\min(A,N_o(g))}{\arg\min}
                   T_B(g,b_g^*,c),\\
V_g&=T_B(g,b_g^*,c_g^*)-L.
\end{aligned}
$$

All divisions in the bound for $b_g^*$ are integer bounds. The C++ `Bound`
helper also caps each memory-derived bound at $10^6$ units. This is
irrelevant for the default 910B3 capacities but is part of the current code
if unusually large custom capacities are passed. The $b$ monotonic
argument remains valid when a latency is zero. With the optional source-step
packet objective, $h_A,h_B$ change discretely with `baseK`, so the solver
evaluates every legal aligned `baseK` and each legal core count for that
geometry instead of applying this elimination. If multiple core counts
tie, the current implementation chooses the larger count. Its periodic
full-row load calculation takes $O(c^2)$ work per geometry, independent of
the number of output tiles.

The remaining master problem is

$$
\min_y\sum_{g\in\mathcal G_B}V_gy_g,
\qquad\sum_{g\in\mathcal G_B}y_g=1,
\qquad y_g\in\{0,1\}.
$$

It has one binary column per locally feasible geometry and one equality
row. The selected result reports $L+V_g$. With no coupling between
geometries, this one-hot MILP is mathematically the same as taking the
smallest $V_g$; HiGHS is retained as the common solver interface. The
computed optimum is over the declared spatial grid and this reduced
objective, rather than every legal `mat_mul_v3` plan.

After selection, `Finish` runs the standalone `FinalizePlan` preflight and
recomputes `ReducedObjectiveUs` as a consistency check. It separately scores
the selected event graph when expansion fits the event cap. The graph score
does not change the BASE choice. Some dtype, layout, key, and backend rules
are checked only at finalization, so the MILP certificate alone does not
establish that CANN can emit the plan.

## Worked example: choosing among 16 geometries

Use $(M,N,K)=(256,256,1024)$ in FP16, $A=20$, and a 32 MiB workspace
budget. Let each of `singleCoreM/N` range over $\{128,256\}$ and each of
`baseM/N` over $\{64,128\}$: 16 spatial geometries. The example uses the
default `CoreMemory` capacities $C_{0A}=C_{0B}=65{,}536$,
$C_{0C}=131{,}072$, and the measured CANN 9.0.0 default
$C_1=524{,}032$ bytes. The primitive parameters
are **illustrative**, not measured 910B3 values:

| Parameter | Value |
| --- | ---: |
| $R_C$ | 14,745,600 FLOP/µs per core |
| $R_R=R_W$ | 1,600,000 bytes/µs |
| $\lambda_R=\lambda_W$ | 0.005 µs per operation |
| $\lambda_{\rm launch}$ | 0.2 µs |

Other required primitives may use the values in [`mip_example.cpp`](mip_example.cpp);
they do not enter this BASE calculation because $K<65535$.

For $g=(C_M,C_N,B_M,B_N)=(128,128,128,128)$:

$$
\begin{aligned}
N_o&=(256/128)^2=4, &N_M=N_N&=2, &S&=4,\\
D&=2(1024)(256\cdot2+256\cdot2)=2{,}097{,}152\ \mathrm{bytes},\\
O&=65{,}536, &F&=134{,}217{,}728\ \mathrm{FLOP},\\
b_g^*&=\min(64,8,8,32)=8, &\text{baseK}&=128,\\
P_B&=\lceil1024/128\rceil=8, &c_g^*&=4\quad\text{(checked over }c=1,2,3,4\text{)}.
\end{aligned}
$$

The L0C footprint is $8(128)(128)=131{,}072$ bytes, exactly its capacity;
the other local-memory and workspace limits also hold. Because $K<65535$,
$L=0.2$ µs. The two service terms are

$$
\begin{aligned}
T_{\rm Cube}&=\max_q\frac{2(1024)(128\cdot128)}{14{,}745{,}600}
=2.275556\ \mathrm{\mu s},\\
T_{\rm GM}&=\frac{2{,}097{,}152}{1{,}600{,}000}
+\frac{2(65{,}536)}{1{,}600{,}000}
+2(1)(8)(0.005)+1(0.005)\\
&=1.31072+0.08192+0.08+0.005
=1.47764\ \mathrm{\mu s},\\
T_B&=0.2+\max(2.275556,1.47764)
=2.475556\ \mathrm{\mu s}.
\end{aligned}
$$

Selected comparisons show why both outer and base geometry matter:

| $(C_M,C_N,B_M,B_N)$ | $c_g^*$ | `baseK` | $T_{\rm Cube}$ (µs) | $T_{\rm GM}$ (µs) | $T_B$ (µs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| $(128,128,128,128)$ | 4 | 128 | 2.275556 | 1.477640 | **2.475556** |
| $(128,128,64,128)$ | 4 | 128 | 2.275556 | 2.213000 | **2.475556** |
| $(128,128,64,64)$ | 4 | 256 | 2.275556 | 2.868360 | 3.068360 |
| $(128,256,128,128)$ | 2 | 128 | 4.551111 | 1.557640 | 4.751111 |
| $(256,256,128,128)$ | 1 | 128 | 9.102222 | 1.717640 | 9.302222 |

The first two shown geometries tie for the reduced BASE minimum across all
16 choices because Cube service dominates both. With the current geometry
ordering and HiGHS result, the emitted BASE decision has `singleCoreM=128`,
`singleCoreN=128`, `baseM=64`, `baseN=128`, `baseK=128`,
`singleCoreK=1024`, `stepK=1`, and `usedCoreNum=4`. The following API
call reproduces the selection when `hardware` uses the parameter table
above and the remaining primitives from `mip_example.cpp`:

```cpp
const mmopt::Workload workload{256, 256, 1024, "fp16"};
const mmopt::MipTileDomain domain{{128, 256}, {128, 256},
                                   {64, 128}, {64, 128}};
const auto result = mmopt::SolveReducedMip(workload, hardware, {}, domain);
// result.base->surrogate_us == 2.475556 (rounded to six decimals)
```

`result.winner` may select Split-K instead; `result.base` is the object
described here. The value 2.475556 µs is a **model estimate**, not a device
measurement.
