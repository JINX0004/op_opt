# `SolveSplit`: mathematical formulation and source-code alignment

This document describes the deterministic multi-core Split-K branch of
[`SolveReducedMip`](mip_solver.h), implemented by [`SolveSplit`](mip_solver.cpp).
It follows the same partial-minimization method as
[`SolveBase`](SOLVE_BASE_FORMULATION.md): compute a best local cost for each
spatial geometry, then select one geometry with a one-hot master problem.
The objective is the current **reduced service-time estimate** in µs. The
source review below identifies which assumptions are and are not yet
enforced by the standalone C++ model.

## Decisions and derived work

For workload $(M,N,K)$, allocated AIC count $A$, and spatial geometry
$g=(C_M,C_N,B_M,B_N)$, the decisions are:

| Mathematical symbol | Tiling field | Current standalone rule |
| --- | --- | --- |
| $g$ | `singleCoreM/N`, `baseM/N` | Independent 16-aligned choices from `MipTileDomain`, with $B_M\le C_M$, $B_N\le C_N$ |
| $s$ | `stepK` | Positive integer; `baseK=128` and `singleCoreK=128s` |
| $c$ | `usedCoreNum` | Integer from 2 through the allocated AIC count, subject to slices and workspace |

The geometry grid and tail-aware counts are the same as in the
[`SolveBase` derivation](SOLVE_BASE_FORMULATION.md#inputs-derived-counts-and-decisions):

$$
\begin{aligned}
N_o(g)&=\left\lceil M/C_M\right\rceil\left\lceil N/C_N\right\rceil,\\
N_M(g)&=\left\lfloor M/C_M\right\rfloor\left\lceil C_M/B_M\right\rceil
       +\left\lceil(M\bmod C_M)/B_M\right\rceil,\\
N_N(g)&=\left\lfloor N/C_N\right\rfloor\left\lceil C_N/B_N\right\rceil
       +\left\lceil(N\bmod C_N)/B_N\right\rceil,\\
S(g)&=N_M(g)N_N(g),\quad
D(g)=2K\bigl(MN_N(g)+NN_M(g)\bigr),\\
O&=MN,\quad F=2MNK,\quad q=K/128,\quad P(s)=\lceil q/s\rceil.
\end{aligned}
$$

$q$ counts 128-element K packets and $P$ counts `singleCoreK` slices.
The supported standalone preflight requires $K$ to be a multiple of 128,
so $q$ is an integer. $D$ is the model's no-cache operand read traffic in
bytes for 2-byte inputs; it is not a measured cache-miss count.
For each outer output tile $t$, let $O_t=m_tn_t$ be its actual element
count, including M/N tails. The source kernel launches up to two AIVs per
occupied AIC. The standalone model balances each tile's elements across
those AIVs and defines its vector critical-path work as

$$
Q_g(c)=\sum_{t=1}^{N_o(g)}
\left\lceil\frac{O_t}{2c}\right\rceil.
$$

Thus $Q_g(c)$ sums the most-loaded AIV's work for each outer tile, assuming
those tiles' reduction stages are sequential in this approximation.
For a tail tile with fewer than $2c$ elements, the ceiling is one and some
AIVs are idle.

## Reduced timing objective

Let $R_C$ be Cube FLOP/µs per AIC, $R_R,R_W$ the modeled GM read/write
byte rates, $R_V$ vector-add ops/µs, and
$\lambda_R,\lambda_W,\lambda_V,\lambda_S$ the fixed read, write,
vector-add, and synchronization latencies. Let $L$ contain the common
launch cost and any A ND→NZ conversion, as defined in the
[`SolveBase` document](SOLVE_BASE_FORMULATION.md#objective). The exact
function evaluated by the present C++ reduced model is

$$
\begin{aligned}
T_D(g,s,c)=L+\max\Bigg\{&
 \underbrace{\frac{F}{cR_C}}_{T_{\rm Cube}},\\
&\underbrace{\frac{D(g)+4P(s)O}{R_R}
 +\frac{(4P(s)+2)O}{R_W}
 +\bigl(2S(g)q+N_o(g)(P(s)-c+1)\bigr)\lambda_R
 +N_o(g)(P(s)+1)\lambda_W}_{T_{\rm GM}}\Bigg\}\\
&+N_o(g)(\lambda_S+\lambda_V)+\frac{(c-1)Q_g(c)}{R_V}.
\end{aligned}
$$

The $4PO$ read and $4PO$ write terms represent FP32 partial-output
traffic over $P$ slices; $2O$ is the final 2-byte output. The fixed read
term includes two operand reads for each spatial base packet and K packet,
plus partial-result reads and one reduction read per outer output tile.
The fixed write term includes $P$ partial writes and one final write per
outer output tile. The final line charges one sync and vector launch per
output tile, plus $(c-1)O$ vector additions **in total**. The vector
term charges the additions on the most-loaded AIV, using a per-AIV rate
$R_V$; it does not multiply aggregate device throughput by $2c$ without
regard to idle lanes or tails. The standalone
[`event_builder.cpp`](event_builder.cpp) now emits parallel AIV reduction
events and joins them before the final output write. Source MK/NK and L2
paths can partition rows or NZ blocks differently, so balanced shards
remain an approximation rather than an exact instruction schedule.
The reduced solver requires zero fixed Cube latency because the formula
does not charge it per K packet.

## Feasibility constraints

Let $C_{0A},C_{0B},C_{0C},C_1$ be the standalone L0A/L0B/L0C/L1
capacities. Let $W_0=20\,\mathrm{MiB}+\mathbf1_{K>65535}(2MK)$ be the RPC
reservation plus any converted A matrix, and let $W_{\max}$ be the
workspace budget. The current Split-K feasibility checks are

$$
\begin{aligned}
&B_M\le C_M,\quad B_N\le C_N,\quad 1\le s\le q,\\
&2\le c\le A,\quad c\le P(s),\\
&4B_M(128)\le C_{0A},\quad 4B_N(128)\le C_{0B},\\
&8B_MB_N\le C_{0C},\\
&4(B_M+B_N)(128s)\le C_1,\\
&W_0+8cC_MC_N\le W_{\max}.
\end{aligned}
$$

The $8cC_MC_N$ workspace term is the source branch's `usedCoreNum ×
singleCoreM × singleCoreN × 2 × sizeof(float)` ping-pong partial buffer,
for the aligned-output/fix-opti-BASE case. The 20 MiB RPC reservation is
also source-derived; see
[`GetDeterministicSplitKWorkspaceSize`](../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp).
For source paths with NZ sequential output or different fix optimization,
the workspace formula changes and is outside this standalone subcase.

The fixed L0 bounds, the L1 step bound, and geometry alignment are checked
before the master is built. The selected plan is checked again by
`FinalizePlan`. Global dtype, layout, key, and backend rules are not all
master rows; passing the standalone preflight is not proof of CANN
emittability.

## Eliminate `stepK`, then optimize the occupied cores

For a fixed geometry, the standalone L1 constraint gives

$$
s_{\max,g}=\min\!\left(q,
\left\lfloor\frac{C_1}{512(B_M+B_N)}\right\rfloor\right).
$$

Workspace and the basic slice count give an initial core upper bound

$$
c_{\max,g}=\min\!\left(A,q,
\left\lfloor\frac{W_{\max}-W_0}{8C_MC_N}\right\rfloor\right).
$$

As in `SolveBase`, the C++ `Bound` helper caps memory-derived integer
bounds at $10^6$; the cap does not affect the default 910B3 capacities or
the $A\le20$ core bound.

For fixed $g$ and $c$, all coefficients multiplying $P(s)$ in $T_{\rm GM}$
are nonnegative. Increasing $s$ reduces or preserves $P(s)$ and changes
neither Cube nor the per-AIV vector work $Q_g(c)$. It cannot increase the
reduced objective.
The slice condition $c\le\lceil q/s\rceil$ is exactly equivalent, for
integer $c\ge2$, to

$$
s\le\left\lfloor\frac{q-1}{c-1}\right\rfloor.
$$

Hence the largest legal step is an optimum for that core count:

$$
s_g^*(c)=\min\!\left(s_{\max,g},
\left\lfloor\frac{q-1}{c-1}\right\rfloor\right),\qquad
P_g^*(c)=\left\lceil\frac{q}{s_g^*(c)}\right\rceil.
$$

Discard $g$ when fixed storage fails, $s_{\max,g}<1$, or
$c_{\max,g}<2$. Otherwise evaluate

$$
V_g=\min_{c=2,\ldots,c_{\max,g}}
\bigl[T_D(g,s_g^*(c),c)-L\bigr].
$$

This checks at most 19 core counts because the standalone 910B3 profile
permits at most 20 AICs. Unlike BASE, Split-K cannot simply choose
$c=c_{\max,g}$: more cores lower $F/(cR_C)$ and reduce a fixed partial-read
term, while $(c-1)Q_g(c)/R_V$ may rise, flatten, or fall as more AIVs
share each tile. The core-count loop remains necessary. Ignoring AIV
parallelism can materially change which core count wins.

Finally, the geometry master is

$$
\min_y\sum_{g\in\mathcal G_D}V_gy_g,
\qquad\sum_{g\in\mathcal G_D}y_g=1,
\qquad y_g\in\{0,1\}.
$$

It has one binary column per locally feasible geometry and one equality
row. `SolveSplit` adds $L$ to the selected local cost, constructs the
decision, checks it against `ReducedObjectiveUs`, and optionally scores the
event graph. The one-hot master is also equivalent to taking the minimum
$V_g$ directly while there is no cross-geometry coupling.

## Mapping to `mat_mul_v3`

| Source behavior | Standalone representation and remaining gap |
| --- | --- |
| [`SupportMultiSplitK`](../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp) and `DoDeterministicMultiCoreSplitKTiling` gate the family using feature, shape, K, transpose, and alignment conditions. | The reduced solver applies its own typed preflight after choosing a plan; it does not encode this host-family gate. |
| `IsNkOrder` chooses MK or NK traversal. `SetBasicBlockOfMK33`/`SetBasicBlockOfNK33` initialize 128×128×128 blocks, 3×3 step/depth patterns, one full spatial axis, and possible later L2 splitting. | `SolveSplit` allows independent $C_M,C_N,B_M,B_N$ and variable $s$. It does not carry the source order, L2 split, `stepM/N`, or actual `depthA1/B1`. |
| The host starts with `usedCoreNum = min(K-slice count, aicNum)` and may apply a small-M/N adjustment. The [kernel](../../mat_mul_v3/op_kernel/mat_mul_deterministic_splitk_kernel.h) divides K slices among occupied cores using a round/remainder rule, then launches `2 × usedCoreNum` AIVs for FP32 reduction. | `SolveSplit` makes $c$ a cost decision and enforces $c\le P(s)$. Its reduced objective and event graph now represent parallel AIV shards, but use ideal element balancing rather than the source's MK/NK row or NZ-block partition. Changing the host-selected $c$ requires backend validation. |
| `GetDeterministicSplitKWorkspaceSize` allocates ping-pong FP32 partial buffers plus RPC workspace; other fix modes can use a different layout. | The formula $W_0+8cC_MC_N$ matches the aligned-output/base-fix subcase only. |

For an optimizer integrated into `mat_mul_v3`, the feasible set should be
constructed from source-valid tuples
$\tau=(\text{order},B_K,\text{stepM/N},\text{stepKa/Kb},
\text{depthA1/B1},\text{buffer/layout mode})$, with the host gate and CANN
finalization applied to each candidate. The local problem would then be
$V_g^{\rm src}=\min_{(c,\tau)\in\mathcal L_{\rm src}(g)}
T_D^{\rm src}(g,c,\tau)$, followed by the same one-hot geometry master.
The current $s_g^*(c)$ shortcut applies to that problem only if the
source-valid step/depth choices retain the monotonic conditions proved
above.

## Worked example and a source-code boundary

Use FP16 $(M,N,K)=(512,512,32768)$, $A=20$, a 128 MiB workspace limit,
and the one-geometry domain
$g=(C_M,C_N,B_M,B_N)=(384,512,128,128)$. This geometry matches the
**initial** MK-order spatial pattern of
[`DoDeterministicMultiCoreSplitKTiling`](../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp):
`SetBasicBlockOfMK33` starts from `baseM=baseN=baseK=128`,
`stepM=3`, `stepKa=stepKb=3`, `singleCoreM=3×128=384`, and
`singleCoreN=N`. The source can subsequently apply L2 splitting and CANN
tiling finalization, so this match is only to that initial pattern.

Use the same **illustrative, unmeasured** primitive rates as
[`mip_example.cpp`](mip_example.cpp): $R_C=14{,}745{,}600$ FLOP/µs,
$R_R=R_W=1{,}600{,}000$ bytes/µs, $R_V=1{,}000{,}000$ ops/µs,
$\lambda_R=\lambda_W=0.005$ µs,
$\lambda_S=0.1$ µs, $\lambda_V=0$, and launch $L=0.2$ µs.
Default `CoreMemory` gives $C_{0A}=C_{0B}=65{,}536$,
$C_{0C}=131{,}072$, and $C_1=524{,}288$ bytes. Then

$$
\begin{aligned}
N_o&=2, &N_M=N_N&=4, &S&=16,\\
D&=268{,}435{,}456\ \mathrm{bytes}, &O&=262{,}144,
&F&=17{,}179{,}869{,}184\ \mathrm{FLOP},\\
q&=256, &s_{\max,g}&=4, &c_{\max,g}&=20.
\end{aligned}
$$

For $c=4$, $s_g^*(4)=4$ and $P=\lceil256/4\rceil=64$. The two output
tiles contain $384\cdot512$ and $128\cdot512$ elements, so
$Q_g(4)=\lceil196{,}608/8\rceil+\lceil65{,}536/8\rceil=32{,}768$.
The cost terms are

$$
\begin{aligned}
T_{\rm Cube}&=291.271111\ \mathrm{\mu s},\\
T_{\rm GM}&=167.77216+41.94304+42.27072+41.57+0.65
             =294.20592\ \mathrm{\mu s},\\
T_{\rm sync+vector}&=2(0.1)+3(32{,}768)/1{,}000{,}000
                    =0.298304\ \mathrm{\mu s},\\
T_D(g,4,4)&=0.2+294.20592+0.298304
          =294.704224\ \mathrm{\mu s}.
\end{aligned}
$$

The local core-count check gives:

| $c$ | $s_g^*(c)$ | $Q_g(c)$ | $T_{\rm Cube}$ (µs) | $T_{\rm GM}$ (µs) | $T_D$ (µs) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 65,536 | 582.542222 | 294.225920 | 583.007758 |
| 3 | 4 | 43,691 | 388.361481 | 294.215920 | 388.848863 |
| 4 | 4 | 32,768 | 291.271111 | 294.205920 | 294.704224 |
| 5 | 4 | 26,215 | 233.016889 | 294.195920 | 294.700780 |
| **20** | **4** | **6,555** | **58.254222** | **294.045920** | **294.570465** |

An actual `SolveReducedMip` call with this one-geometry domain returns
`result.deterministic_split_k->decision.step_k == 4`,
`used_cores == 20`, and `surrogate_us == 294.570465` at six decimals.
With one serial vector resource, the previous approximation selected four
cores at 295.392352 µs. Accounting for parallel AIVs changes the decision:
the smaller per-AIV reduction term lets the GM read-call savings from more
cores outweigh vector work in this example.

```cpp
const mmopt::Workload workload{512, 512, 32768, "fp16"};
const mmopt::MipTileDomain domain{{384}, {512}, {128}, {128}};
// Use the mip_example.cpp primitive profile with a 128 MiB workspace limit.
const auto result = mmopt::SolveReducedMip(workload, hardware, {}, domain);
const auto& split = *result.deterministic_split_k;
// split.decision.step_k == 4; split.decision.used_cores == 20
```

**This result is not currently source-emittable merely because its shape
and geometry resemble the source branch.** In
[`SetBasicBlockOfMK33`](../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp),
the host initializes `stepKa=stepKb=3`, `stepM=3`,
`depthA1=9`, and `depthB1=6`. With 128×128×128 blocks, the implied source
L1 use is $2(128)(128)(9+6)=491{,}520$ bytes. The standalone preflight
instead models two symmetric K-step windows,
$4(B_M+B_N)(128s)$; it allows $s=4$ at exactly 524,288 bytes. Scaling
the source 3×3 depth pattern from $s=3$ to $s=4$ would require 655,360
bytes and exceed the same L1 capacity. The source host also initializes
`usedCoreNum=min(ceil(K/singleCoreK), A)` before its small-M/N adjustment;
for this shape that is 20. The reduced model now selects 20 as well, but
this agreement does not resolve the step/depth mismatch. The
source's [`SupportMultiSplitK`](../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp)
gate, MK/NK order selection, possible L2 split, and final CANN tiling
checks are likewise absent from the standalone feasible set.

To optimize **source-emittable** Split-K plans, first produce the feasible
branch patterns from the host tiler (or an exact reusable legality builder),
including order, `stepM/N`, `stepKa/Kb`, buffer depths, and workspace mode.
For each source-valid geometry and core count, minimize over the source's
allowed K-step/depth tuples. The monotone closed-form step bound above is
valid only when that tuple's other costs and constraints have the same
monotonic structure. Then use the same one-hot geometry master and pass the
selected complete plan through CANN finalization. Until that integration,
`SolveSplit` certifies only the stated standalone surrogate domain.
