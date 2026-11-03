# Single-core Split-K, mode 2: FP16/BF16 MKN reduced formulation

Mode 2 is `TilingEnableSplitCore::SINGLE_CORE_SPLIT_K`. The kernel is
`MatMulSingleCoreSplitKKernel` with `IS_NKM = false` and
`MM_CFG_PRELOAD_MK`, block `MatmulSingleCoreSplitKBaseBlock`. This
document is the reduced service-time of that schedule. `SolveReducedMip`
now evaluates this FP16/BF16 mode alongside the other supported families and
returns the lowest reduced objective. The result is not an event-graph
makespan or a device-time guarantee.

The scope is FP16 or BF16 A, B, and output, with FP32 accumulation. Unless stated
otherwise, the timing expression describes the aligned-output kernel;
the unaligned output and mixed ND→NZ variants need their own epilogues.

The same family contains [mode 5](SOLVE_SC_SPLITK_MODE5_FORMULATION.md)
and [mode 6](SOLVE_SC_SPLITK_MODE6_FORMULATION.md). Deterministic
multi-core Split-K is a different kernel and is formulated in
[SolveSplit](SOLVE_SPLIT_FORMULATION.md).

## What the schedule does

AICs own disjoint spatial tiles of `singleCoreM` by `singleCoreN`. Each
AIC walks its own tile along K in slices of `singleCoreK`. Slice 0 of a
panel Fixpipes FP32 with atomic add off. Every later slice atomic-adds
into that same FP32 workspace. After the K loop of one M panel, the AIC
signals its two AIVs, and those AIVs cast the panel from FP32 to the
output type. The next M panel's Cube starts while that cast runs.

## Geometry the host emits

`DoSingleCoreSplitKTiling` starts from `SetBasicBlockOfMK33`. With no
bias and with the current M/N tile product below the AIC count, it
switches to `SetBasicBlockOf24`. A side shorter than 384 then shrinks
the M step. `CheckSingleTilingOk` sets

$$
C_K=\mathrm{stepKa}\cdot B_K,\qquad B_K=128.
$$

$B_K$ follows the 256-byte base-K setting for two-byte FP16/BF16 inputs.
Base M and base N stay 128.
`dbL0c` is 2. The inner panel extents are $I_M=\mathrm{stepM}\cdot 128$
and $I_N=\mathrm{stepN}\cdot 128$.

| Block | stepM | stepN | stepKa | stepKb | $I_M$ | $I_N$ | $C_K$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MK33 | 3 | 1 | 3 | 3 | 384 | 128 | 384 |
| 2×4 | 2 | 1 | 4 | 4 | 256 | 128 | 512 |
| short side | 1 | 1 | 4 | 4 | 128 | 128 | 512 |

Mode 2 keeps `l2Info.calOrder = 0`. `UpdateBlockCnt` then forces
`innerLoopN = 1`. `CalcGMOffset` treats that single N iteration as the
last inner block, so one `IterateAll` receives the tile's full N extent
(the N tail, on a tail tile). The M loop remains
$\lceil m/I_M\rceil$ panels. `iterateOrder` inside the matmul engine
stays `ITER_ROW_FIRST` for MK33; that order is inside one `IterateAll`
and is outside the panel count.

## Spatial assignment and K slices

Let $C_M,C_N$ be the host's `singleCoreM/N` and $c$ the occupied AIC
count. Tile tails are $m_{\rm tail}=M-(N_M-1)C_M$ and
$n_{\rm tail}=N-(N_N-1)C_N$, with $N_M=\lceil M/C_M\rceil$ and
$N_N=\lceil N/C_N\rceil$.

$$
N_o=N_M N_N,\qquad
r=\left\lceil\frac{N_o}{c}\right\rceil,\qquad
c_{\rm pre}=
\begin{cases}
c & \text{if }c\mid N_o,\\
N_o\bmod c & \text{otherwise.}
\end{cases}
$$

`InitBlockIndex` gives the first $c_{\rm pre}$ AICs $r$ tiles and the
other AICs $r-1$ tiles. Tile index $i$ has
$m_{\rm index}=i\bmod N_M$ and $n_{\rm index}=\lfloor i/N_M\rfloor$, so
M varies fastest. The last index is the corner tail; an M-edge index
uses $m_{\rm tail}$ and a full $C_N$; an N-edge index uses a full $C_M$
and $n_{\rm tail}$.

K slices, shared by every panel:

$$
P=\left\lceil\frac{K}{C_K}\right\rceil,\qquad
k_p=
\begin{cases}
C_K & 0\le p<P-1,\\
K-(P-1)C_K & p=P-1.
\end{cases}
$$

`loopK` in the block is this $P$. `kCoreTail` is $k_{P-1}$.

## One K slice

For a panel of $m$ rows, $n$ columns, and $k$ along K, write
$m_b=\lceil m/128\rceil$, $n_b=\lceil n/128\rceil$, and
$q_k=\lceil k/B_K\rceil$. The modeled operand traffic follows the same
no-cache count as the [BASE objective](SOLVE_BASE_FORMULATION.md#objective):
A is charged once per N base tile and B once per M base tile.

$$
D(m,n,k)=2k\,(m\,n_b+n\,m_b).
$$

$R_C$ is Cube FLOP/µs per AIC. $R_R$ and $\lambda_R$ are the `gm_read`
byte rate and fixed latency. $R_W$ and $\lambda_W$ are the `gm_write`
byte rate and fixed latency. Every slice Fixpipes $4mn$ bytes of FP32.
Slice 0 is a plain store. Later slices are atomic adds of those same
bytes. The generic profile charges the same store term to both. With
optional `sc_fp32_store`, the FP32 term $4mn/R_W+m_b n_b\lambda_W$
uses its separate rate and startup rather than generic `gm_write`. With
optional `sc_atomic_extra`, slice $p>0$ additionally incurs
$m_b n_b\lambda_A+4mn/R_A$ in the memory branch, where $\lambda_A$
and $R_A$ describe the incremental cost over a plain FP32 Fixpipe store.

$$
T_{\rm slice}(m,n,k)=
\max\!\left(
\frac{2mnk}{R_C},\;
\frac{D(m,n,k)}{R_R}+2\,m_b n_b q_k\,\lambda_R
+\frac{4mn}{R_W}+m_b n_b\,\lambda_W
\right).
$$

The $m_b n_b$ write startups are one Fixpipe per base tile of the panel.
Operand reads and this Fixpipe sit inside one `IterateAll`, so they form
the memory branch of that slice. Successive K slices are successive
`IterateAll` calls and add.

## Cast and the panel pipeline

A panel's Cube chain is the sum of its K slices. Its cast starts at
`NotifyEvent` after that sum. Two AIVs split the panel's rows; the
larger shard has $Q=\lceil m/2\rceil\,n$ elements.

$$
\begin{aligned}
T_K(m,n)&=\sum_{p=0}^{P-1}T_{\rm slice}(m,n,k_p),\\[4pt]
T_V(m,n)&=
\lambda_{\rm sync}+\frac{4Q}{R_R}+\frac{2Q}{R_W}+\lambda_{\rm store}.
\end{aligned}
$$

$\lambda_{\rm sync}$ is `sync.fixed_latency_us`. $\lambda_{\rm store}$ is
`gm_write.fixed_latency_us`, the same epilogue startup used by
deterministic Split-K. The $4Q$ and $2Q$ terms are the FP32 workspace
load and the output store of the larger AIV, charged at the existing GM
rates. With optional `sc_cast`, $T_V$ also includes
$\lambda_{\rm cast}+Q/R_{\rm cast}$. Without it, this estimate can miss
cast compute beyond the charged reads and writes. The [910B3 single-core
probe](bench910/SINGLE_CORE_SPLITK_PRIMITIVES_910B3.md) gives exploratory
values for both optional terms.

On one tile of M extent $m_t$ and N extent $n_t$ there are
$J=\lceil m_t/I_M\rceil$ panels. Panel $j<J-1$ has $I_M$ rows. The last
panel has $m_t-(J-1)I_M$ rows. Every panel has $n_t$ columns. A core's
tiles are concatenated in index order, and the cast of one panel
overlaps the Cube chain of the next panel, including across the tile
boundary inside `realRound`. For each AIC $q$, let $K_{q,j}$ and
$U_{q,j}$ be the Cube-chain and cast service of panel $j$. The ideal
two-stage pipeline, including variable-size tails, obeys

$$
\boxed{\displaystyle
\begin{aligned}
C_{q,-1}&=V_{q,-1}=0,\\
C_{q,j}&=C_{q,j-1}+K_{q,j},\\
V_{q,j}&=\max\!\big(C_{q,j},V_{q,j-1}\big)+U_{q,j},\\
T_2&=L+\max_q V_{q,J_q-1}.
\end{aligned}}
$$

Here $K_{q,j}=T_K(m_{q,j},n_{q,j})$ and
$U_{q,j}=T_V(m_{q,j},n_{q,j})$. One panel collapses to
$L+T_K+T_V$. $L$ is the launch latency from the
[BASE objective](SOLVE_BASE_FORMULATION.md#objective), plus the A ND→NZ
workspace conversion when $K>65535$. The busiest AIC is the maximum of
this pipeline over the $c$ cores, each with its own tile extents. When
$M$ and $N$ are divisible by $C_M$ and $C_N$, every tile is full and the
front cores, which own $r$ tiles, are the busiest.

## Workspace

Host `CalcSplitCoreWorkspace` reserves one FP32 buffer of the whole
output, plus the 20 MiB RPC buffer. FP16/BF16 gives an alignment of 128
elements. Host ND→NZ conversion buffers add $W_{\rm ND2NZ,A}$ and
$W_{\rm ND2NZ,B}$ when their respective flags are set:

$$
W=4M\left\lceil N/128\right\rceil 128+20\,\mathrm{MiB}
  +W_{\rm ND2NZ,A}+W_{\rm ND2NZ,B}.
$$

N divisible by 128 uses the real N as the Cube C stride. Otherwise
the block aligns that stride to 64 FP32 elements, and
`NNot128AlignProcess` runs the Cube loop, then one vector cast or
padding removal after the Cube side has finished. $T_2$ is the aligned
schedule, where each M panel is cast while the next panel's Cube runs.

## What stays outside $T_2$

The host search that proposes $(C_M,C_N)$ is outside the expression.
`DoSingleCoreSplitKTiling` cuts N by the L2 tile length (3072, or 1536
on the small-L2 path), adjusts the tile factors toward the AIC count,
and keeps a candidate only when `CheckSingleTilingOk` accepts it. The
usual rejection, skipped for the K=1536 channel and for the mode 5
shape, is `singleCoreN < 512`, a spatial load ratio below 0.7, or the
band $896\le N\le 2048$ with `singleCoreN \le 640` or load ratio below
0.85. Entry to the family is `IsSupportSingleCoreSplitK`: the K=1536
large-MN channel, $K\ge 27392$, an L2-tile failure on a large enough
MKN shape, or the TLB/MATA shapes.

$T_{\rm slice}$ does not apply L2 reuse, the preload-MK depth, or the
bias load on slice 0. Cache-hint changes and transpose changes of the
physical HBM stream are outside the byte count. The single
Cube/vector flag is modeled as a two-stage producer-consumer pipeline;
the recurrence does not resolve flag reuse or shared GM contention
between active AICs and AIVs.
