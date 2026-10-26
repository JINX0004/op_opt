# SolveSplit: deterministic Split-K reduced formulation

The [SolveSplit implementation](mip_solver.cpp) minimizes a reduced
service-time objective for deterministic multi-core Split-K. It computes
the best K step and occupied AIC count for each spatial geometry, then
solves a one-hot geometry master. The result is optimal within this
surrogate and its finite geometry grid. It is not an exact MatMulV3 schedule
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

For output tile $t$, let $O_t$ be its actual element count. The vector
reduction approximation balances elements across up to two AIVs per
occupied AIC, with sequential outer tiles:

$$Q_g(c)=\sum_{t=1}^{N_o(g)}
\left\lceil\frac{O_t}{2c}\right\rceil.$$

## Reduced timing objective

Let $R_C$ be Cube FLOP/µs per AIC, $R_R,R_W$ shared GM read/write
bytes/µs, and $R_V$ vector-add ops/µs per AIV. Fixed read, write,
vector, and sync costs are $\lambda_R,\lambda_W,\lambda_V,\lambda_S$.
$L$ is common launch and A ND→NZ conversion time, as defined in the
[SolveBase objective](SOLVE_BASE_FORMULATION.md#objective).
The exact reduced function implemented in SplitStageUs is

$$
\begin{aligned}
T_C(g,s,c)&=\frac{2O\,k_{\rm crit}(s,c)}{R_C},\\
T_{\rm GM}(g,s,c)&=
\frac{D(g)+4P(s)O}{R_R}
+\frac{(4P(s)+2)O}{R_W}\\
&\quad+\{2S(g)q_{\rm crit}(s,c)+N_o(g)r(s,c)\}\lambda_R\\
&\quad+N_o(g)\{r(s,c)+1\}\lambda_W,\\
T_V(g,c)&=N_o(g)(\lambda_S+\lambda_V)
+\frac{(c-1)Q_g(c)}{R_V},\\
T_D(g,s,c)&=L+\max\{T_C,T_{\rm GM}\}+T_V.
\end{aligned}
$$

The $4PO$ read and write terms represent FP32 partial-result traffic;
$2O$ is the final 2-byte output. Shared bytes are totaled across
occupied AICs, while fixed packet service and Cube work follow a
busiest AIC. The maximum approximates Cube/GM overlap. The vector term
charges the largest balanced AIV shard per outer tile. This omits exact
MK/NK ordering, cache reuse, pipeline stalls, and source AIV sharding.
The common A-conversion primitive above $K=65535$ is provisional.
Fixed Cube latency must be zero because this formula does not count
Cube calls.

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

It does not always choose the largest legal step. A larger $s$ reduces
partial-result traffic but can increase $q_{\rm crit}$ because whole
K slices are assigned to AICs. This is a bounded local scan, not an
event-by-event search. On 910B3, $c\le20$ and L1 usually keeps $s$ small.

The master chooses one locally feasible geometry:

$$\min_y\sum_{g\in\mathcal G_D}V_gy_g,\qquad
\sum_g y_g=1,\qquad y_g\in\{0,1\}.$$

It has one binary column per feasible geometry and one equality row.
SolveSplit adds $L$ after the master and checks the selected coefficient
against ReducedObjectiveUs. The separate event-graph score is reported
only when its event and workspace representations fit; it does not choose
the winner.

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

Substituting these counts into the equations gives the reduced objective
for the source tuple: 118.565 µs with the audit primitive profile.
The optimizer then compares other legal tuples.
The [revised local audit](bench910/k_gt_65535_v1/revised_solver_audit.csv)
selects $384\times384$, $s=3,c=19$, for this shape, with a Split-K
estimate of 118.565 µs. This tuple has not been emitted or timed through
the MatMulV3 kernel. The measured source default is 101.578 µs.
For the larger 512² and 1024² long-K cases, the reduced model still
selects BASE while the source chooses Split-K. This is a remaining
family-selection error, not evidence that BASE is faster on device.

Source MK/NK 3×3 step/depth patterns, possible L2 splitting, and key
rules still constrain emittable tuples. A source-emittable optimizer
needs those constraints and measured Split-K and conversion primitives
before making a device-performance claim.
