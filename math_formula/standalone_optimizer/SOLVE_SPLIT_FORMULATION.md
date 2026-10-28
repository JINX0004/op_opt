# SolveSplit: deterministic Split-K reduced formulation

The [SolveSplit implementation](mip_solver.cpp) minimizes a reduced
service-time objective for deterministic multi-core Split-K. It computes
the best K step and occupied AIC count for each spatial geometry, then
solves a one-hot geometry master. The result is optimal within $T_D$
below and its finite geometry grid. It is not an exact MatMulV3 schedule
or a device-time guarantee.

## Variables, counts, and source assignment

The workload is FP16/BF16 multiplication of $(M,K)$ by $(K,N)$ on
$A\le20$ available AICs. A geometry is
$g=(C_M,C_N,B_M,B_N)$, where $C$ is a single-core output extent and $B$
is a base Cube tile. Local integer decisions are occupied AIC count
$c\ge2$ and K step $s\ge1$. This branch fixes baseK $=128$ and
singleCoreK $=128s$. The standalone preflight requires $K$ divisible
by 128. Define $q=K/128$ and $P(s)=\lceil q/s\rceil$.

$$
\begin{aligned}
N_o(g)&=\lceil M/C_M\rceil\lceil N/C_N\rceil,\\
N_M(g)&=\lfloor M/C_M\rfloor\lceil C_M/B_M\rceil
       +\lceil(M\bmod C_M)/B_M\rceil,\\
N_N(g)&=\lfloor N/C_N\rfloor\lceil C_N/B_N\rceil
       +\lceil(N\bmod C_N)/B_N\rceil,\\
S(g)&=N_M(g)N_N(g),\qquad O=MN,\\
D(g)&=2K\{M N_N(g)+N N_M(g)\}.
\end{aligned}
$$

$D$ is modeled operand bytes without cache reuse; M/N packet counts
include actual output tails. The source assigns whole contiguous K slices
to AICs. A busiest AIC receives
$r(s,c)=\lceil P(s)/c\rceil$ slices, containing at most

$$
q_{\rm crit}(s,c)=\min\{q,sr(s,c)\}
\quad\text{base K packets},\qquad
k_{\rm crit}=\min\{K,128q_{\rm crit}\}.
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
$R_V$ vector-add ops/µs per AIV. $\lambda_R$ is the fixed operand-read
cost, $\lambda_{\rm sync}$ the per-panel Cube/vector signal, and
$\lambda_{\rm store}$ the final cast and FP16 store. $\lambda_{\rm partial}$
is the fixed `MTE2→V`, add, `V→MTE2` cost of each extra partial. The
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
\frac{D_{\rm panel}}{R_R}+2S_{\rm panel}\,q_{\rm crit}\,\lambda_R
\right),\\[4pt]
T_V&=
\lambda_{\rm sync}
+(c-1)\!\left(\lambda_{\rm partial}+\frac{Q}{R_V}\right)
+\lambda_{\rm store},\\[4pt]
T_D&=
L+T_{\rm cube}+(N_{\rm panel}-1)\max(T_{\rm cube},T_V)+T_V.
\end{aligned}
$$

$S_{\rm panel}$ is the base-packet count of that panel. For one panel
the pipeline collapses to $L+T_{\rm cube}+T_V$. With many panels it is
the two-buffer schedule, not $N_{\rm panel}$ copies of the whole chain
stacked with no overlap. Cache reuse, the source's NZ row sharding, and
the choice of MK versus NK order are still outside this expression.
Fixed Cube latency must remain zero because the Cube term does not count
Cube calls. The common A-conversion primitive above $K=65535$ stays inside
$L$. For $(128,256,4096)$ both the solver tuple and the emitted kernel have
$N_{\rm panel}=1$, so $T_D=L+T_{\rm cube}+T_V$ and the new latency appears
as $(c-1)\lambda_{\rm partial}$: fifteen extra partials at $c=16$, ten at
$c=11$. Measured kernel medians for that shape are 15.70 µs for source
BASE and 19.12 µs for deterministic Split-K.

## Feasible set and solution

For capacities $C_{0A},C_{0B},C_{0C},C_1$ and workspace limit
$W_{\max}$, the standalone constraints are

$$
\begin{aligned}
&B_M\le C_M,\quad B_N\le C_N,\quad 1\le s\le q,\\
&2\le c\le\min\{A,P(s)\},\\
&4B_M(128)\le C_{0A},\quad4B_N(128)\le C_{0B},\\
&8B_MB_N\le C_{0C},\quad4(B_M+B_N)(128s)\le C_1,\\
&W_0+8cC_MC_N\le W_{\max},\\
&W_0=20\,\mathrm{MiB}+\mathbf1_{K>65535}(2MK).
\end{aligned}
$$

The workspace term follows the source's aligned-output/base-fix
ping-pong partial buffers plus 20 MiB RPC reservation. Other source
fix modes require another formula. Full host-family gates, dtype/layout
rules, and CANN legality are not all MILP rows. FinalizePlan cross-checks
the chosen standalone tuple.

For fixed $g$, the local bounds are

$$
\begin{aligned}
s_{\max,g}&=\min\left\{q,
\left\lfloor\frac{C_1}{512(B_M+B_N)}\right\rfloor\right\},\\
c_{\max,g}&=\min\left\{A,q,
\left\lfloor\frac{W_{\max}-W_0}{8C_MC_N}\right\rfloor\right\}.
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

## Worked long-K example and model boundary

For $(M,N,K)=(256,256,65536)$, the
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
