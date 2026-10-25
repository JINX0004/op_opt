# `SolveBase`: mathematical formulation and worked example

This document describes the BASE branch of [`SolveReducedMip`](mip_solver.h),
as implemented by [`SolveBase`](mip_solver.cpp). It chooses the outer
`singleCoreM/N` extents, inner `baseM/N/K` tiles, and occupied Cube-core count
for one fixed matrix multiplication workload. Its objective is the current
**reduced service-time estimate** in microseconds. It is not the full
event-graph makespan or a measured 910B3 runtime.

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
D(g)&=2K\bigl(MN_N(g)+NN_M(g)\bigr),\\
O&=MN,\qquad F=2MNK.
\end{aligned}
$$

$N_o$ is the number of outer output tiles. $N_M,N_N$ count inner base
packets along each axis, including an outer-tile tail; $S$ is their product.
$D$ is the modeled A/B GM read traffic in bytes for FP16/BF16 inputs under
the present no-cache traffic assumption. $O$ is the number of output
elements and $F$ is the total floating-point work.

The full discrete decisions are a geometry $g$, $b=\text{baseK}/16$, and
occupied core count $c=\text{usedCoreNum}$. BASE fixes
`singleCoreK = K` and `stepK = 1`. In the **actual master MILP**, only
one binary $y_g$ per feasible geometry remains a variable. The code
computes the best $b$ and $c$ for each geometry before constructing that
MILP.

## Objective

Let $R_C$ be FLOP/µs per Cube core, $R_R,R_W$ the modeled GM read/write
byte rates, and $\lambda_R,\lambda_W$ the fixed GM read/write latency per
operation. Define $P_B(b)=\lceil K/(16b)\rceil$. The candidate objective is

$$
\boxed{\displaystyle
T_B(g,b,c)=L+\max\!\left\{
\underbrace{\frac{F}{cR_C}}_{T_{\rm Cube}},
\underbrace{\frac{D(g)}{R_R}+\frac{2O}{R_W}
 +2S(g)P_B(b)\lambda_R+N_o(g)\lambda_W}_{T_{\rm GM}}
\right\}.}
$$

The GM expression represents one modeled operand-read pair per spatial
base packet and K packet, plus an output write per outer tile. The factor
$2O$ is the modeled output byte count. Cube and GM service are combined
with `max` as a throughput lower-envelope approximation; the equation does
not resolve event precedence, transfer overlap, buffering, or shared-device
contention in detail.

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

## Constraints and local elimination

Let $C_{0A},C_{0B},C_{0C},C_1$ be the modeled L0A, L0B, L0C, and L1
capacities in bytes. The BASE feasibility conditions used to form a local
geometry cost are

$$
\begin{aligned}
1\le b&\le K/16, & 1\le c&\le\min(A,N_o(g)),\\
64B_Mb&\le C_{0A}, &64B_Nb&\le C_{0B},\\
64(B_M+B_N)b&\le C_1, &8B_MB_N&\le C_{0C},\\
W_0&\le W_{\max},
&W_0&=20\,\mathrm{MiB}+\mathbf1_{K>65535}(2MK).
\end{aligned}
$$

The L0A/L0B and L1 inequalities are the code's double-buffer footprints
after substituting `baseK = 16b`; L0C stores two FP32 accumulator tiles.
`SolveBase` first rejects the family if $W_0>W_{\max}$. It then discards a
geometry if L0C fails or no positive $b$ or $c$ exists. Geometric alignment,
$B_M\le C_M$, and $B_N\le C_N$ are enforced when the grid is built.

For any fixed feasible $g$, increasing $b$ cannot increase $T_B$: it only
reduces or preserves the packet count $P_B$. Increasing $c$ cannot increase
$T_B$: it only reduces the Cube term. Therefore an optimum exists at

$$
\begin{aligned}
b_g^*&=\min\!\left(
\frac K{16},
\left\lfloor\frac{C_{0A}}{64B_M}\right\rfloor,
\left\lfloor\frac{C_{0B}}{64B_N}\right\rfloor,
\left\lfloor\frac{C_1}{64(B_M+B_N)}\right\rfloor
\right),\\
c_g^*&=\min(A,N_o(g)),\\
V_g&=T_B(g,b_g^*,c_g^*)-L.
\end{aligned}
$$

All divisions in the bound for $b_g^*$ are integer bounds. The C++ `Bound`
helper also caps each memory-derived bound at $10^6$ units. This is
irrelevant for the default 910B3 capacities but is part of the current code
if unusually large custom capacities are passed. The monotonic argument
remains valid when a latency is zero: the upper bound is still *an* optimal
choice, even if smaller choices tie it.

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
$C_{0C}=131{,}072$, and $C_1=524{,}288$ bytes. The primitive parameters
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
P_B&=\lceil1024/128\rceil=8, &c_g^*&=\min(20,4)=4.
\end{aligned}
$$

The L0C footprint is $8(128)(128)=131{,}072$ bytes, exactly its capacity;
the other local-memory and workspace limits also hold. Because $K<65535$,
$L=0.2$ µs. The two service terms are

$$
\begin{aligned}
T_{\rm Cube}&=\frac{134{,}217{,}728}{4(14{,}745{,}600)}
=2.275556\ \mathrm{\mu s},\\
T_{\rm GM}&=\frac{2{,}097{,}152}{1{,}600{,}000}
+\frac{2(65{,}536)}{1{,}600{,}000}
+2(4)(8)(0.005)+4(0.005)\\
&=1.31072+0.08192+0.32+0.02
=1.73264\ \mathrm{\mu s},\\
T_B&=0.2+\max(2.275556,1.73264)
=2.475556\ \mathrm{\mu s}.
\end{aligned}
$$

Selected comparisons show why both outer and base geometry matter:

| $(C_M,C_N,B_M,B_N)$ | $c_g^*$ | `baseK` | $T_{\rm Cube}$ (µs) | $T_{\rm GM}$ (µs) | $T_B$ (µs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| $(128,128,128,128)$ | 4 | 128 | 2.275556 | 1.732640 | **2.475556** |
| $(128,128,64,64)$ | 4 | 256 | 2.275556 | 3.363360 | 3.563360 |
| $(128,256,128,128)$ | 2 | 128 | 4.551111 | 1.722640 | 4.751111 |
| $(256,256,128,128)$ | 1 | 128 | 9.102222 | 1.717640 | 9.302222 |

The first geometry minimizes the reduced BASE objective across all 16
choices. Its emitted BASE decision has `singleCoreM=128`,
`singleCoreN=128`, `baseM=128`, `baseN=128`, `baseK=128`,
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
