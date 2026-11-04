# Single-core Split-K, mode 5: FP16/BF16 NKM reduced formulation

Mode 5 is `TilingEnableSplitCore::SINGLE_CORE_NKM_SPLIT_K`. It uses the
same kernel as [mode 2](SOLVE_SC_SPLITK_MODE2_FORMULATION.md), instantiated
with `IS_NKM = true` and `MM_CFG_PRELOAD_NK`. The block type is still
`MatmulSingleCoreSplitKBaseBlock`. Cube enters `ProcessBlockNKM`: the
outer panel loop is N, K sits in the middle, and M is the inner loop.
This document is the reduced service time of that schedule.
`SolveReducedMip` now evaluates this FP16/BF16 mode alongside the other
supported families. The result is not an event-graph makespan or a
device-time guarantee.

The scope is FP16 or BF16 A, B, and output, with FP32 accumulation. The FP32
long-K channel of this source mode is outside this document.

[Mode 6](SOLVE_SC_SPLITK_MODE6_FORMULATION.md) is the other single-core
Split-K path. Deterministic multi-core Split-K is
[SolveSplit](SOLVE_SPLIT_FORMULATION.md).

## FP16/BF16 host channel

`CheckSingleTilingOk` assigns FP16/BF16 mode 5 only when A and B need no
ND→NZ workspace conversion and `IsSupportSingleCoreSplitSmallK` holds
with the axes swapped: $K=1536$, both axes divisible by 128,
$M\ge 49152$, $N=384$,
and $MN$ at most $0.65$ of the L2 size measured in FP32 elements.
`SetBasicBlockOfNK33` runs, and `calOrder` is 1.

| NK33 field | value |
| --- | --- |
| baseM, baseN | 128 |
| baseK | 128 |
| depthA1, depthB1 | 6, 9 |
| stepM, stepN | 1, 3 |
| stepKa, stepKb | 3, 3 |
| $I_M$, $I_N$ | 128, 384 |
| iterateOrder | `ITER_COL_FIRST` |
| $C_K$ | 384 |

`singleCoreK` is $\mathrm{stepKa}\cdot B_K=384$. On this channel the
kernel also disables the L2 hint on A when $\mathrm{stepN}\cdot\mathrm{baseN}=N$.

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

The slice and cast services are the same functions as mode 2. For a
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
above remains in force.

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

The cast of that tile is $T_V(m_t,n_t)$, issued after its K loop. For
each AIC $q$, concatenate the spatial tiles in the order assigned by
`InitBlockIndex`. Let $K_{q,j}=T_K^{\rm tile}(m_{q,j},n_{q,j})$ and
$U_{q,j}=T_V(m_{q,j},n_{q,j})$. The ideal Cube/vector pipeline is

$$
\boxed{\displaystyle
\begin{aligned}
C_{q,-1}&=V_{q,-1}=0,\\
C_{q,j}&=C_{q,j-1}+K_{q,j},\\
V_{q,j}&=\max\!\big(C_{q,j},V_{q,j-1}\big)+U_{q,j},\\
T_5&=L+\max_q V_{q,J_q-1}.
\end{aligned}}
$$

One tile collapses to $L+T_K^{\rm tile}+T_V$.

## Workspace and the host gate

The workspace is the mode 2 buffer: one FP32 image of the aligned
output plus 20 MiB RPC. $N=384$ is divisible by 128, and no ND→NZ
conversion buffer is needed:

$$
W=4MN+20\,\mathrm{MiB}.
$$

The Cube C stride is the real N on this aligned FP16/BF16 channel.

## What stays outside $T_5$

The host predicate above selects this channel. Preload-NK depth, the
A-matrix L2 hint, transpose, and L2 reuse sit outside $T_{\rm slice}$.
The byte count is
the same no-cache count as $T_B$. Bias is loaded on K slice 0 and has
no separate term. The single Cube/vector flag is modeled as a
two-stage producer-consumer pipeline; flag reuse and shared GM
contention remain unresolved.

The [910B3 mode-5 tile diagnosis](bench910/MODE5_TILE_DIAGNOSIS_910B3.md)
shows why the original reduced objective reversed the measured tile
ranking at 49152×384×1536. The measured FP16/BF16 option above chooses
the source-sized tile on that shape and two held-out M sizes per dtype.
It still underpredicts kernel times, and the BASE versus mode-5 family
margin is less reliable than the mode-5 tile ranking.
