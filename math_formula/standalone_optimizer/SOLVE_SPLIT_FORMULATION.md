# SolveSplit: deterministic Split-K reduced formulation

The [SolveSplit implementation](mip_solver.cpp) minimizes a reduced
service-time objective for deterministic multi-core Split-K. It computes
the best K step and occupied AIC count for each spatial geometry, then
solves a one-hot geometry master. The result is optimal within $T_D$
below and its finite geometry grid. It is not an exact MatMulV3 schedule
or a device-time guarantee.

The FP16/BF16 equations describe the checked-in standalone model. The
FP32 substitutions below describe the experimental 910 selector branch,
as captured in its [model snapshot](bench910/fp32_mode2_v1/remote_selector/model/mip_solver.cpp);
the local `main` checkout's `mip_solver.cpp` and `plan_finalizer.cpp`
still accept only FP16/BF16. The product FP32 online gate remains closed.

The general FP16/BF16 geometry grid is broader than the MatMulV3
deterministic Split-K host tiler. At `(M,N,K)=(2560,1024,73728)` in FP16,
the deployed online
selector's earlier reduced solve chose `singleCoreM×N=128×128`, whereas
the source's NK-order branch emits `2560×384`: it starts from
`singleCoreM=M` and `singleCoreN=stepN×baseN`, then may split M for L2.
The [current outer-M diagnostic](bench910/converted_a_probe_v1/PHASE_MEASUREMENT.md#why-the-optimized-singlecorem-can-be-too-small)
shows that the reduced objective gives identical time to M=128 and
M=2560 when N, base tile, K step, and core count are fixed. Thus the
returned M can be an arbitrary tie; it is not a runtime prediction that
smaller M is better. The online selector uses the reduced result to
choose the family, while source tiling emits the actual tile. The
selected geometry needs a source-path feasibility check before being
treated as an executable tiling.

## Variables, counts, and source assignment

The workload is same-type FP16, BF16, or FP32 multiplication of $(M,K)$
by $(K,N)$, with FP32 accumulation and $A\le20$ available AICs. Let
$b_A,b_B,b_C$ be A, B, and final-output bytes per element (2 for each
FP16/BF16 tensor and 4 for each FP32 tensor), and let $b_P=4$ be bytes
per FP32 partial. A geometry is
$g=(C_M,C_N,B_M,B_N)$, where $C$ is a single-core output extent and $B$
is a base Cube tile. Local integer decisions are occupied AIC count
$c\ge2$ and K step $s\ge1$. The dtype path fixes
$B_K=\text{baseK}=128$ for FP16/BF16 and $B_K=64$ for FP32, with
`singleCoreK` $=B_Ks$. Define $q=\lceil K/B_K\rceil$ and
$P(s)=\lceil q/s\rceil$. The supported NN/ND preflight requires K to
be 256-byte aligned, so $q=K/B_K$ for these dtype paths; the last
$s$-packet K slice may be short.

$$
\begin{aligned}
N_o(g)&=\lceil M/C_M\rceil\lceil N/C_N\rceil,\\
N_M(g)&=\lfloor M/C_M\rfloor\lceil C_M/B_M\rceil
       +\lceil(M\bmod C_M)/B_M\rceil,\\
N_N(g)&=\lfloor N/C_N\rfloor\lceil C_N/B_N\rceil
       +\lceil(N\bmod C_N)/B_N\rceil,\\
S(g)&=N_M(g)N_N(g),\qquad O=MN,\\
D(g)&=K\{b_A M N_N(g)+b_B N N_M(g)\}.
\end{aligned}
$$

$D$ is modeled operand bytes without cache reuse; M/N packet counts
include actual output tails. The source assigns whole contiguous K slices
to AICs. A busiest AIC receives
$r(s,c)=\lceil P(s)/c\rceil$ slices, containing at most

$$
q_{\rm crit}(s,c)=\min\{q,sr(s,c)\}
\quad\text{base K packets},\qquad
k_{\rm crit}=\min\{K,B_Kq_{\rm crit}\}.
$$

The cap handles a short final K slice. A source-valid outer tile may
exceed an output axis: $C_M=384>M=256$ still covers only 256 actual rows.
The default long-K grid includes this source 384 extent for axes 128–383;
an explicit grid can also request it. BASE excludes oversized core tiles.

## Panel schedule

`MatMulMultiCoreSplitKDivide` loops over outer panels, `OutCnt`. That
count is $\lceil N/C_N\rceil$ or $\lceil M/C_M\rceil$, whichever axis the
source iterates outside. Call it $N_{\rm panel}$. It is not the output-tile
product $N_o$. A geometry whose single-core extents cover both axes has
$N_{\rm panel}=1$ for either order. One panel's actual element count is
$O_{\rm panel}$; its operand-read bytes are $D_{\rm panel}$. The busiest
AIC's $k_{\rm crit}$ is unchanged, because the K loop sits inside the
panel loop and every panel repeats the same K split.

Each AIC accumulates its own K slices with `IterateAll` and only then
signals the vector cores. The vector loop therefore adds $c$ workspace
partials, not $P(s)$ K slices, and the sequential chain has length $c-1$.
Those counts coincide only when each AIC owns one slice. Two workspace
buffers let Cube fill panel $i+1$ while the vector cores reduce panel $i$.
Cube waits for a buffer only from the third panel onward. Inside one
panel the reduction does not start until that panel's Cube loop has
finished, so those two stages do not overlap.

One AIV's shard of a panel is
$Q=\lceil O_{\rm panel}/(2c)\rceil$, using two AIVs per occupied AIC.
Tail panels keep their own $O_{\rm panel}$. The closed form below uses
one representative panel; an implementation sums the real tail extents.

## Reduced timing objective

Let $R_C$ be Cube FLOP/µs per AIC, $R_R$ shared GM read bytes/µs, and
$R_V$ vector-add ops/µs per AIV. $\lambda_R$ is the fixed GM-to-L1 step-read
cost, $\lambda_{\rm sync}$ the per-panel Cube/vector signal, and
$\lambda_{\rm store}$ the fixed final output-store cost. For FP16/BF16,
the vector path casts its FP32 reduction to the output dtype before
storing; for FP32 it stores the FP32 reduction directly. The current
reduced objective charges the same fixed `gm_write` latency for either
path and does **not** include final-output byte service or a separate
cast primitive. $\lambda_{\rm partial}$ is the fixed `MTE2→V`, add,
`V→MTE2` cost of each extra partial. The
audit profile sets it to 1.28 µs. The reduction probe has not measured
this value. $L$ is common launch and A ND→NZ conversion time, as defined
in the [SolveBase objective](SOLVE_BASE_FORMULATION.md#objective).

Operand reads overlap Cube work of the same panel, so they form the memory
branch of $T_{\rm cube}$. Partial loads do not: `SplitKVectorProcess`
issues them only after `AIC_SYNC_AIV`, and $T_V$ carries that chain.

$$
\begin{aligned}
T_{\rm cube}&=
\max\!\left(
\frac{2\,O_{\rm panel}\,k_{\rm crit}}{R_C},\;
\frac{D_{\rm panel}}{R_R}
+2S_{\rm panel}g(q_{\rm crit},s,K)\lambda_R
\right),\\[4pt]
T_V&=
\lambda_{\rm sync}
+(c-1)\!\left(\lambda_{\rm partial}+\frac{Q}{R_V}\right)
+\lambda_{\rm store},\\[4pt]
T_D&=
L+T_{\rm cube}+(N_{\rm panel}-1)\max(T_{\rm cube},T_V)+T_V.
\end{aligned}
$$

$S_{\rm panel}$ is the spatial base-packet count of that panel. The
startup count is path-specific:

$$
g(q_{\rm crit},s,K)=
\begin{cases}
\lceil q_{\rm crit}/s\rceil, & K>65535\quad\text{(converted A)},\\
q_{\rm crit}, & K\le65535\quad\text{(unconverted A surrogate)}.
\end{cases}
$$

On the measured converted-A path, each operand's L1 K step covers up to
$s$ base-K slices. The unconverted path retains its earlier effective
per-baseK startup; grouping it degraded family selection on held-out
device cases. Byte service still covers all K slices. For one panel
the pipeline collapses to $L+T_{\rm cube}+T_V$. With many panels it is
the two-buffer schedule, not $N_{\rm panel}$ copies of the whole chain
stacked with no overlap. For unequal tail panels, the implementation uses
$T_D=L+T_{\rm cube,1}+
\sum_{i=2}^{N_{\rm panel}}\max(T_{\rm cube,i},T_{V,i-1})+
T_{V,N_{\rm panel}}$ instead of the identical-panel shorthand above.
Cache reuse, the source's NZ row sharding, and
the choice of MK versus NK order are still outside this expression.
Fixed Cube latency must remain zero because the Cube term does not count
Cube calls. The common A-conversion primitive above $K=65535$ stays inside
$L$. For the measured FP16 $(128,256,4096)$ case, both the solver tuple
and the emitted kernel have
$N_{\rm panel}=1$, so $T_D=L+T_{\rm cube}+T_V$ and the new latency appears
as $(c-1)\lambda_{\rm partial}$: fifteen extra partials at $c=16$, ten at
$c=11$. Measured kernel medians for that shape are 15.70 µs for source
BASE and 19.12 µs for deterministic Split-K.

The [converted-A in-context phase sweep](bench910/converted_a_probe_v1/PHASE_MEASUREMENT.md)
shows a long-K limitation of the fixed operand-read term. At
$(M,N,K)=(2560,1024,73728)$ on the source $2560\times384$ Split-K tile,
the old $2S_{\rm panel}q_{\rm crit}\lambda_R$ summed to 1052.35 µs across
three panels. Counting the source's $s=3$ L1 K steps instead gives
350.78 µs. The modeled GM payload term alone is 1627.88 µs, versus the
measured 1715.65 µs post-wait AIC interval. This grouping removes a
known overcount but does not identify the true on-path startup latency:
the source pipelines transfers and may reuse L2 data. The common
converted-A handoff also remains underestimated. Both need validation
across source-emitted tilings before changing the online selector.

## Feasible set and solution

For capacities $C_{0A},C_{0B},C_{0C},C_1$ and workspace limit
$W_{\max}$, the standalone constraints are

$$
\begin{aligned}
&B_M\le C_M,\quad B_N\le C_N,\quad
  1\le s\le\lfloor K/B_K\rfloor,\\
&2\le c\le\min\{A,P(s)\},\\
&2b_AB_MB_K\le C_{0A},\quad2b_BB_NB_K\le C_{0B},\\
&2b_PB_MB_N\le C_{0C},\quad
  2B_Ks(b_AB_M+b_BB_N)\le C_1,\\
&W_0+2b_PcC_MC_N\le W_{\max},\\
&W_0=20\,\mathrm{MiB}+\mathbf1_{K>65535}(b_AMK).
\end{aligned}
$$

The workspace term follows the source's aligned-output/base-fix
ping-pong partial buffers plus 20 MiB RPC reservation. Other source
fix modes require another formula. Full host-family gates, dtype/layout
rules, and CANN legality are not all MILP rows. FinalizePlan cross-checks
the chosen standalone tuple. Since $b_P=4$, the L0C and partial-workspace
terms remain $8B_MB_N$ and $8cC_MC_N$ for all three dtypes. Also
$b_AB_K=b_BB_K=256$ bytes for both FP16/BF16 and FP32, so the displayed
L0A/L0B and L1 capacity coefficients are numerically unchanged after
substitution. The packet count $q$ and the data traffic $D$ do change.
The current FP32 preflight rejects $K>65535$; its converted-A workspace
term is shown for algebraic completeness, not as a supported FP32 path.

For fixed $g$, the local bounds are

$$
\begin{aligned}
s_{\max,g}&=\min\left\{\lfloor K/B_K\rfloor,
\left\lfloor\frac{C_1}{2B_K(b_AB_M+b_BB_N)}\right\rfloor\right\},\\
c_{\max,g}&=\min\left\{A,q,
\left\lfloor\frac{W_{\max}-W_0}{2b_PC_MC_N}\right\rfloor\right\}.
\end{aligned}
$$

For each $c\in[2,c_{\max,g}]$ and integer
$s\in[1,s_{\max,g}]$ with $c\le P(s)$, the code evaluates the closed-form
objective and retains

$$V_g=\min_{(s,c)\in\mathcal L(g)}\{T_D(g,s,c)-L\}.$$

It does not always choose the largest legal step. A larger $s$ changes
$q_{\rm crit}$ and therefore $T_{\rm cube}$. $T_V$ depends on $c$ and $Q$
rather than on $P(s)$. Whole K slices are still assigned to AICs, so
increasing $s$ can raise the busiest AIC's $k_{\rm crit}$. This is a
bounded local scan, not an event-by-event search. On 910B3, $c\le20$ and
L1 usually keeps $s$ small.

The master chooses one locally feasible geometry:

$$\min_y\sum_{g\in\mathcal G_D}V_gy_g,\qquad
\sum_g y_g=1,\qquad y_g\in\{0,1\}.$$

It has one binary column per feasible geometry and one equality row.
SolveSplit adds $L$ after the master and checks the selected coefficient
against ReducedObjectiveUs. The separate event-graph score is reported
only when its event and workspace representations fit; it does not choose
the winner. That graph reduces every partial once, after all Cube slices,
so its reported score is not $T_D$.

### FP32 source-informed domain and model limit

For the evaluated same-type FP32 NN/ND path (FP32 output, no bias, HF32
disabled), the source-informed domain follows MatMulV3's MK/NK 3×3
pattern: $B_M=B_N=128$, $B_K=64$, $s=3$, and hence
`singleCoreK=192`. One outer output axis begins at 384; the other may
be partitioned to fit the source's 70% L2 panel budget. The occupied
AIC count follows the host's K-slice rule, including its small-M/N
special case, rather than being freely optimized over $2,\ldots,20$.
The source-informed FP32 solver fixes this geometry, K step, and AIC
count before scoring. The general finite-domain formulation above can
still study alternative tuples, but their objective alone does not prove
the host can emit them. See the
[FP32 source-domain audit](bench910/fp32_mode2_v1/remote_selector/FP32_SOURCE_DOMAIN.md).

The FP32 profile uses an effective $R_C=3{,}600{,}000$ FLOP/µs per AIC;
its GM and vector rates and the `split_partial` latency are inherited
from the earlier FP16/BF16 profile. These are provisional FP32
coefficients. The current $T_V$ has no $b_CO_{\rm panel}/R_W$ final-store
term and no dtype-specific cast or reduction throughput. Adding those
services would change the implementation and require corresponding
measurements; they must not be silently included when comparing this
document with the current solver.

## Worked long-K example and model boundary

For FP16 $(M,N,K)=(256,256,65536)$, the
[source tiling audit](bench910/k_gt_65535_v1/source_audit.jsonl)
emits $g=(384,256,128,128)$, $s=3$, $c=20$. Its oversized $C_M$
covers only 256 actual rows. Here $q=512$,
$P=\lceil512/3\rceil=171$, $r=\lceil171/20\rceil=9$, and
$q_{\rm crit}=27$. The busiest AIC processes at most 3456 K elements;
the ideal fractional share $K/20=3276.8$ misses this slice imbalance.
There is one outer output tile, $N_M=N_N=2$, $S=4$,
$O=65{,}536$, and $D=134{,}217{,}728$ bytes. The workspace formula gives
20 MiB RPC + 32 MiB converted A + $8(20)(384)(256)=15$ MiB partial
buffers, or 67 MiB total.

This geometry is one panel: $C_M$ covers $M$ and $C_N$ covers $N$, so
$N_{\rm panel}=1$ and $T_D=L+T_{\rm cube}+T_V$. The
[recorded audit](bench910/k_gt_65535_v1/revised_solver_audit.csv) reports
118.565 µs for $384\times384$, $s=3,c=19$, and the same figure for the
source tuple above. That file predates $\lambda_{\rm partial}$ and is not
a substitution of the equations in this document. The tuple has not been
emitted or timed through the MatMulV3 kernel. The measured source default
is 101.578 µs. For the larger 512² and 1024² long-K cases, that same
audit selects BASE while the source chooses Split-K. Recompute those
rankings with this $T_D$ and the audit profile's 1.28 µs
$\lambda_{\rm partial}$.

Source MK/NK 3×3 step/depth patterns, possible L2 splitting, and key
rules still constrain emittable tuples. A source-emittable optimizer
needs those constraints and measured Split-K and conversion primitives
before making a device-performance claim.

### FP32 worked substitution

Take $(M,N,K)=(128,128,32768)$ with the source-informed FP32 tuple
$g=(384,128,128,128)$, $B_K=64$, $s=3$, and $c=20$. The oversized
$C_M=384$ covers the 128 real rows. There is one panel and one spatial
base packet. The K allocation and modeled traffic are

$$
\begin{aligned}
q&=32768/64=512,&P&=\lceil512/3\rceil=171,\\
r&=\lceil171/20\rceil=9,&q_{\rm crit}&=27,\\
k_{\rm crit}&=27(64)=1728,&D&=4(32768)(128+128)
 =33{,}554{,}432\ \mathrm{bytes},\\
Q&=\lceil16384/40\rceil=410,&
W&=20\ \mathrm{MiB}+8(20)(384)(128)=27.5\ \mathrm{MiB}.
\end{aligned}
$$

Using the experimental profile's $R_C=3{,}600{,}000$ FLOP/µs,
$R_R=3{,}710{,}212$ bytes/µs, $R_V=1{,}000{,}000$ ops/µs,
$\lambda_R=0.109620$ µs, $\lambda_{\rm partial}=1.28$ µs,
$\lambda_{\rm sync}=0.1$ µs, $\lambda_{\rm store}=0.162586$ µs, and
$L=2.68$ µs gives

$$
\begin{aligned}
T_{\rm cube}&=\max\left\{
  \frac{2(16384)(1728)}{3{,}600{,}000},
  \frac{33{,}554{,}432}{3{,}710{,}212}+2(27)(0.109620)
  \right\}=15.728640\ \mathrm{\mu s},\\
T_V&=0.1+19\left(1.28+\frac{410}{1{,}000{,}000}\right)
 +0.162586=24.590376\ \mathrm{\mu s},\\
T_D&=2.68+15.728640+24.590376=42.999016\ \mathrm{\mu s}.
\end{aligned}
$$

This is a reproducible substitution into the reduced objective, not a
measured FP32 kernel runtime. In particular, it inherits the uncalibrated
FP32 vector parameters and omits final-output byte service as noted
above.
