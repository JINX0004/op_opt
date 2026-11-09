# Single-core Split-K, mode 5: FP16/BF16/FP32 NKM reduced formulation

Mode 5 is `TilingEnableSplitCore::SINGLE_CORE_NKM_SPLIT_K`. It uses the
same kernel as [mode 2](SOLVE_SC_SPLITK_MODE2_FORMULATION.md), instantiated
with `IS_NKM = true` and `MM_CFG_PRELOAD_NK`. The block type is still
`MatmulSingleCoreSplitKBaseBlock`. FP16/BF16 Cube enters
`ProcessBlockNKM`; FP32 uses the direct-output
`UnAlignedProcess` and its NKM `Exector`. In either case,
the outer panel loop is N, K sits in the middle, and M is the inner loop.
This document is the reduced service time of that schedule.
`SolveReducedMip` evaluates this mode alongside the other
supported families. The result is not an event-graph makespan or a
device-time guarantee.

The scope is same-type FP16, BF16, or FP32 A/B/output, with FP32
accumulation, NN, ND/ND/ND, no bias, on 910B3. This document models the
$K=1536$, large-M, $N=384$ channel. The separate FP32 long-K mode-5
route requires transposed B and remains outside the current NN/ND
selector. The 910 selector branch admits FP32 through the host small-K
dtype gate only when MMV3_FP32_MODE5_EXPERIMENT is set; clean master
still has the FP16/BF16-only gate.

[Mode 6](SOLVE_SC_SPLITK_MODE6_FORMULATION.md) is the other single-core
Split-K path. Deterministic multi-core Split-K is
[SolveSplit](SOLVE_SPLIT_FORMULATION.md).

## Small-K host channel and geometry

`CheckSingleTilingOk` assigns mode 5 only when A and B need no
ND→NZ workspace conversion and `IsSupportSingleCoreSplitSmallK` holds
with the axes swapped: $K=1536$, both axes divisible by 128,
$M\ge 49152$, $N=384$,
and $MN$ at most $0.65$ of the L2 size measured in FP32 elements.
`SetBasicBlockOfNK33` runs, and `calOrder` is 1.

| NK33 field | FP16/BF16 | FP32 |
| --- | ---: | ---: |
| baseM, baseN | 128, 128 | 128, 128 |
| baseK | 128 | 64 |
| depthA1, depthB1 | 6, 9 | 6, 9 |
| stepM, stepN | 1, 3 | 1, 3 |
| stepKa, stepKb | 3, 3 | 3, 3 |
| $I_M$, $I_N$ | 128, 384 | 128, 384 |
| iterateOrder | `ITER_COL_FIRST` | `ITER_COL_FIRST` |
| $C_K$ | 384 | 192 |

`singleCoreK` is $\mathrm{stepKa}\cdot B_K$, where
$B_K=256/b_{\rm in}$ and $b_{\rm in}$ is the input bytes per element.
On this channel the
kernel also disables the L2 hint on A when $\mathrm{stepN}\cdot\mathrm{baseN}=N$.

For the FP32 source-informed search, $B_M=B_N=128$, $B_K=64$,
$C_K=192$, stepM $=1$, stepN $=3$, and stepK $=3$ are fixed.
The source-compatible FP32 solver fixes $C_N=384$ and varies $C_M$
and occupied AIC count, subject to the host shape gate, 8-element M
alignment, L0/L1 capacity, and workspace cap. It draws M extents
from aligned partitions of the output over at most 20 AICs, as in the
[mode-2 FP32 domain](SOLVE_SC_SPLITK_MODE2_FORMULATION.md#fp32-search-domain-and-feasibility).
The source's default is one N tile and a nearly even M partition,
such as $C_M=2464$ for $M=49152$. The solver may score other M
partitions. The restriction to one N tile is evidence based:
[device measurements](bench910/fp32_mode5_v1/README.md) found that
splitting N into three 128-wide tiles was 18% slower at
$73728\times384\times1536$, while the original reduced objective
treated the two choices as nearly tied. The source default also
keeps all three N base tiles within one NKM panel. This restricted
domain does not prove that every proposed M partition is emitted by
the host.

## Shared tile and K counts

Spatial assignment matches mode 2. $N_M=\lceil M/C_M\rceil$,
$N_N=\lceil N/C_N\rceil$, $N_o=N_M N_N$,
$r=\lceil N_o/c\rceil$, and the first $c_{\rm pre}$ AICs own $r$ tiles.
$c_{\rm pre}=c$ when $c$ divides $N_o$, and $N_o\bmod c$ otherwise. M
varies fastest in the tile index. K slices are

$$
P=\left\lceil K/C_K\right\rceil,\qquad
k_p=C_K\ (p<P-1),\qquad
k_{P-1}=K-(P-1)C_K.
$$

### FP16/BF16 slice and cast service

For FP16/BF16, the slice and cast services are the same functions as
mode 2. For a
rectangle of $m$ rows and $n$ columns,

$$
\begin{aligned}
m_b&=\lceil m/128\rceil,\quad
n_b=\lceil n/128\rceil,\quad
q_k=\lceil k/128\rceil,\\
D(m,n,k)&=2k\,(m\,n_b+n\,m_b),\\[4pt]
T_{\rm slice}(m,n,k)&=\max\!\left(
\frac{2mnk}{R_C},\;
\frac{D(m,n,k)}{R_R}+2\,m_b n_b q_k\,\lambda_R
+\frac{4mn}{R_W}+m_b n_b\,\lambda_W
\right),\\[4pt]
T_V(m,n)&=\lambda_{\rm sync}+\frac{4Q}{R_R}+\frac{2Q}{R_W}+\lambda_{\rm store},
\quad Q=\lceil m/2\rceil\,n.
\end{aligned}
$$

Rates and latencies are the profile primitives named in the
[mode 2 formulation](SOLVE_SC_SPLITK_MODE2_FORMULATION.md#one-k-slice).
The FP32 accumulator Fixpipe is a plain store on K slice 0 and an
atomic add afterwards. Optional `sc_fp32_store` replaces the generic
write coefficient for the FP32 partial; `sc_atomic_extra` adds the measured
per-packet and per-byte increment to later slices' memory branch;
optional `sc_cast` adds $\lambda_{\rm cast}+Q/R_{\rm cast}$ to $T_V$.
Both default to zero when omitted. The
FP32 workspace is cast to FP16/BF16 output after each tile's K loop.

## Measured FP16/BF16 mode-5 service option

The 910B3 FP16 and BF16 profiles supply both `sc_mode5_fixpipe`
and `sc_mode5_iterate`. With these paired primitives, the source's MAC,
MTE2, and Fixpipe activity is represented as concurrent service
demands within each K slice, followed by one exposed `IterateAll` setup:

$$
\begin{aligned}
T_C(m,n,k)&=2mnk/R_C,\\
T_R(m,n,k)&=D(m,n,k)/R_R+2m_bn_bq_k\lambda_R,\\
T_F(m,n)&=4mn/R_{F,5}+m_bn_b\lambda_{F,5},\\
T_{5,\mathrm{slice}}(m,n,k)&=
\max\{T_C,T_R,T_F\}+\lambda_{I,5}.
\end{aligned}
$$

The K chain sums this expression over its four 384-K slices. The tile
pipeline and spatial assignment below are unchanged. Both measured
profiles use $R_{F,5}=108{,}000$ bytes/µs and $\lambda_{F,5}=0$.
The per-`IterateAll` setup is 1.8 µs for FP16 and 1.76 µs for BF16.
The Fixpipe term is the total in-context demand for plain and atomic FP32
updates; adding `sc_atomic_extra` again would double-count it. Each
setup coefficient comes from the difference in AIC scalar active time
between a source-sized and a 128×128 tile, divided by the extra
`IterateAll` calls. These are component measurements, not whole-kernel
multipliers. The [FP16 validation](bench910/mode5_tile_experiment_v1/MODEL_VALIDATION.md)
and [BF16 calibration](bench910/bf16_mode5_calibration_v1/README.md) report
the device measurements and held-out checks.

When either primitive is absent, the original $T_{\rm slice}$ definition
above remains in force. These FP16/BF16 rates must not be applied to
FP32, whose packet length and output path differ.

$L$ is the launch term from the
[BASE objective](SOLVE_BASE_FORMULATION.md#objective). Here $K=1536$
and neither operand requires ND→NZ workspace conversion, so the
conversion part of $L$ is inactive.

## Tile pipeline, `calOrder = 1`

`rowOrder = 1` forces `innerLoopM = 1`. The last-block update then
gives each N panel the tile's full M extent. The panel width is
$I_N=384$ and the whole output has $N=384$, hence every spatial tile
has $n_t\le384$ and exactly one N panel. Its M extent is $m_t$.

Within a tile the K slices are serial, and the inner M loop runs once:

$$
T_K^{\rm tile}(m_t,n_t)=\sum_{p=0}^{P-1}T_{\rm slice}(m_t,n_t,k_p).
$$

For FP16/BF16, the cast of that tile is $T_V(m_t,n_t)$, issued after
its K loop. For
each AIC $q$, concatenate the spatial tiles in the order assigned by
`InitBlockIndex`. Let $K_{q,j}=T_K^{\rm tile}(m_{q,j},n_{q,j})$ and
$U_{q,j}=T_V(m_{q,j},n_{q,j})$. The ideal Cube/vector pipeline is

$$
\boxed{\displaystyle
\begin{aligned}
C_{q,-1}&=V_{q,-1}=0,\\
C_{q,j}&=C_{q,j-1}+K_{q,j},\\
V_{q,j}&=\max\!\big(C_{q,j},V_{q,j-1}\big)+U_{q,j},\\
T_5^{16}&=L+\max_q V_{q,J_q-1}.
\end{aligned}}
$$

One FP16/BF16 tile collapses to $L+T_K^{\rm tile}+T_V$.

### FP32 direct-output timeline

For FP32 output, the kernel's AIV path returns before the cast. The
AIC runs the NKM Exector schedule, and each IterateAll writes directly
to output GM. K packet 0 is plain; later packets atomic-add. With
$B_K=64$ and $C_K=192$, there are $P=8$ packets. The current FP32
estimator reuses the
[mode-2 direct-output slice](SOLVE_SC_SPLITK_MODE2_FORMULATION.md#one-k-slice)
with $b_{\rm in}=4$, $q_k=\lceil k/64\rceil$, and its plain/atomic
output terms:

$$
\begin{aligned}
T_{K,5}^{32}(m,n)&=\sum_{p=0}^{7}
     T_{\rm slice}^{32}(m,n,192,p),\\
T_5^{32}&=L+\max_q\sum_{j=0}^{J_q-1}
     T_{K,5}^{32}(m_{q,j},n_{q,j}).
\end{aligned}
$$

There is no vector service in this recurrence. The mode-2
sc_fp32_store_steady rate is a provisional proxy for mode 5, not a
measured FP32 mode-5 parameter. The FP16/BF16 sc_mode5_fixpipe and
sc_mode5_iterate rates also do not transfer to FP32. The source's
inner NKM ordering, prefetch, and L2-hint behavior can expose service
beyond the max-of-Cube-and-memory slice.

### FP32 device check

The [910B3 paired study](bench910/fp32_mode5_v1/README.md) used a
research host package with the small-K FP32 gate enabled. The clean
master selected BASE on both shapes; the research package forced mode 5.
Both outputs passed the FP32 correctness check. Times below are
five-kernel profiler medians.

| $M\times N\times K$ | BASE model / measured, µs | Mode-5 source tile | Mode-5 model / measured, µs |
| --- | ---: | ---: | ---: |
| $49152\times384\times1536$ | 820.09 / 823.34 | $2464\times384\times192$ | 877.03 / 825.42 |
| $73728\times384\times1536$ | 1228.80 / 1216.34 | $3688\times384\times192$ | 1308.76 / 1241.70 |

At the second shape, an isolated $3688\times128\times192$ mode-5
tile measured 1467.98 µs. The reduced objective gives it exactly the
same 1308.76 µs as the 384-wide tile: its byte and Cube terms scale
linearly with three N tiles. The profiler shows substantially more
MTE2 activity on the narrow-N tile. A future broader N-tiling model
needs to charge repeated A transfers and exposed per-IterateAll
service across N panels. The present source-compatible solver fixes
$C_N=384$ while retaining an analytic service objective for M and
occupied-core selection. The online FP32 product gate remains closed.

## Workspace and the host gate

The workspace is the mode 2 buffer: one FP32 image of the aligned
output plus 20 MiB RPC. $N=384$ is divisible by 128, and no ND→NZ
conversion buffer is needed:

$$
W=4MN+20\,\mathrm{MiB}.
$$

The Cube C stride is the real N on this aligned channel. For FP32 the
host still reserves this workspace, but AIC writes to output GM and
the objective charges no workspace cast.

## What stays outside $T_5$

The host predicate above selects this channel. Preload-NK depth, the
A-matrix L2 hint, transpose, and L2 reuse sit outside $T_{\rm slice}$.
The byte count is
the same no-cache count as $T_B$. Bias is loaded on K slice 0 and has
no separate term. For FP16/BF16, the Cube/vector flag is modeled as
a two-stage producer-consumer pipeline; flag reuse and shared GM
contention remain unresolved. FP32 has no vector stage, but atomic
GM writes and their overlap with operand reads remain approximate.

The [910B3 mode-5 tile diagnosis](bench910/MODE5_TILE_DIAGNOSIS_910B3.md)
shows why the original reduced objective reversed the measured tile
ranking at 49152×384×1536. The measured FP16/BF16 option above chooses
the source-sized tile on that shape and two held-out M sizes per dtype.
It still underpredicts kernel times, and the BASE versus mode-5 family
margin is less reliable than the mode-5 tile ranking. These results
apply to FP16/BF16 only.
