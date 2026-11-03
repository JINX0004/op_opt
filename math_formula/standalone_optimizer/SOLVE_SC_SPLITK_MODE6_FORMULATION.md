# Single-core Split-K, mode 6: FP16/BF16 explicit GM→L1 reduced formulation

Mode 6 is `TilingEnableSplitCore::SINGLE_CORE_SPLIT_K_GM_TO_L1`. The
kernel is `MatMulSingleCoreSplitKKernelGmToL1` with
`MM_CFG_PRELOAD_MK`. The block type is still
`MatmulSingleCoreSplitKBaseBlock`, and the panel axis matches
[mode 2](SOLVE_SC_SPLITK_MODE2_FORMULATION.md): `calOrder = 0` forces
one N iteration, so each M panel's `Iterate` covers the tile's full N
extent. The difference is the memory path. The host sets
`shareL1Size = 0` and `shareL0CSize = 0`, and the kernel copies A and B
into L1 itself, then runs `MatmulImpl` from those L1 tensors into L0C
and Fixpipe. This document is the reduced service time of that schedule.
`SolveReducedMip` now evaluates this FP16/BF16 mode alongside the other
supported families. The result is not an event-graph makespan or a
device-time guarantee.

The scope is FP16 or BF16 A, B, and output, with FP32 accumulation. The timing
expression describes the aligned-output kernel; mixed ND→NZ conversion
and other layout-specific copy costs require separate parameters.

[Mode 5](SOLVE_SC_SPLITK_MODE5_FORMULATION.md) is the NKM member of the
same family. Deterministic multi-core Split-K is
[SolveSplit](SOLVE_SPLIT_FORMULATION.md).

## When the host selects it

`IsGmToL1ByShape` runs from `CheckSingleTilingOk` for FP16/BF16 A with $N$
divisible by 128, outside the K=1536 MKN channel (which stays on mode
2). The shape test requires no bias, and both
inner axes at most `SHAPE_LIMIT` $=65535$. With mix ND→NZ disabled those
axes are A's inner axis ($M$ if A is transposed, otherwise $K$) and B's
inner axis ($K$ if B is transposed, otherwise $N$). With mix ND→NZ
enabled they are the outer axes. B in NZ with mix ND→NZ disabled refuses
the mode. The basic block is the MK33, 2×4, or short-side block already
chosen by `DoSingleCoreSplitKTiling`; the table is in the
[mode 2 formulation](SOLVE_SC_SPLITK_MODE2_FORMULATION.md#geometry-the-host-emits).

## K packet the kernel copies

The kernel's L1 K length is $\kappa=512$, and `stepM == 3` sets
$\kappa=384$. On the blocks this mode emits, that length equals
`singleCoreK`:

| Block | stepM | $C_K=\mathrm{stepKa}\cdot B_K$ | $\kappa$ |
| --- | ---: | ---: | ---: |
| MK33 | 3 | $3\cdot 128=384$ | 384 |
| 2×4 | 2 | $4\cdot 128=512$ | 512 |
| short side | 1 | $4\cdot 128=512$ | 512 |

$B_K=128$ for FP16/BF16. `loopK` is
$P=\lceil K/\kappa\rceil$, and the copy offset of the next packet is one
$\kappa$ past the current slice. With $\kappa=C_K$ those two agree, and the
inner `nkloop` of a slice is 1, including a tail shorter than $\kappa$.
The objective is written at that grain. Slice $p$ has length

$$
k_p=
\begin{cases}
\kappa & 0\le p<P-1,\\
K-(P-1)\kappa & p=P-1.
\end{cases}
$$

## One K packet

A of the current packet is copied once per 128-row slot and stays in L1
for the whole N loop. B is copied once per 128-column slot. The slots are the ping, pong, and, when `stepM == 3`, peng buffers.
Each slot holds one M or N piece of the current K packet.
Modeled GM bytes of one packet are therefore one copy of each operand,

$$
D_6(m,n,k)=2k\,(m+n),
$$

and the total DataCopy service, one call per slot, is

$$
T_{\rm mte2}(m,n,k)=
\frac{D_6(m,n,k)}{R_R}
+\left(\left\lceil\frac{m}{128}\right\rceil
      +\left\lceil\frac{n}{128}\right\rceil\right)\lambda_R.
$$

$R_R$ and $\lambda_R$ are the generic `gm_read` rate and fixed latency.
When both optional `sc_gm_to_l1_a` and `sc_gm_to_l1_b` are supplied,
the solver replaces this expression by A service
$2mk/R_{R,A}+\lceil m/128\rceil\lambda_{R,A}$ plus B service
$2nk/R_{R,B}+\lceil n/128\rceil\lambda_{R,B}$. The [910B3
source-shaped copy probe](bench910/SINGLE_CORE_SPLITK_PRIMITIVES_910B3.md)
measures those paths separately. Each
$(128\times 128)$ output piece Fixpipes FP32. Over the panel that is
$4mn$ bytes and $\lceil m/128\rceil\lceil n/128\rceil$ Fixpipe startups.
$R_W$ and $\lambda_W$ are the `gm_write` rate and fixed latency. Slice 0
stores; later slices atomic-add to the same FP32 workspace. Optional
`sc_fp32_store` applies to the first plain-store slice (and to modes
2/5). Optional `sc_fp32_store_steady` is the **total** FP32 Fixpipe and
atomic-output pipeline demand for later mode-6 K packets. Its 910B3
value is measured from the source kernel's `aic_fixpipe_time(us)` counter
in the [device-4 output study](bench910/MODE6_OUTPUT_MEASUREMENT_910B3.md).
When supplied, it supersedes `sc_atomic_extra` for those packets.
When absent, the model uses generic `gm_write` **plus** optional
`sc_atomic_extra` as an exploratory proxy. The serial `sc_fp32_store`
probe writes distinct destinations and does not measure this source
pipeline demand.

Define $S_p(m,n)$ as the measured or proxy output demand for packet $p$:

$$
\begin{aligned}
S_p(m,n)&=
\begin{cases}
4mn/R_0+P_{mn}\lambda_0, & p=0,\\
4mn/R_S+P_{mn}\lambda_S, & p>0\text{ and steady profile supplied},\\
4mn/R_W+P_{mn}\lambda_W+A(m,n), & p>0\text{ otherwise},
\end{cases}\\[4pt]
P_{mn}&=\lceil m/128\rceil\lceil n/128\rceil,\\[4pt]
A(m,n)&=4mn/R_A+P_{mn}\lambda_A\quad
  (\text{zero when the atomic-extra primitive is absent}),\\[4pt]
T_{{\rm mac},p}^{\rm red}(m,n,k)&=
\max\!\left(\frac{2mnk}{R_C},\;S_p(m,n),\;
  \max_{u,v}\left[\frac{2m_u n_v k}{R_C}
      +S_p(m_u,n_v)\right]\right).
\end{aligned}
$$

$R_C$ is Cube FLOP/µs per AIC. Cube reads L1, and the operand bytes are
charged in $T_{\rm mte2}$. Here $m_u,n_v$ are the actual extents of each
128×128 output piece, including tails. The last term retains the
`Iterate`-before-`Fixpipe` dependency for one piece. The first two terms
are throughput bounds for all pieces; their maximum is still a reduced
estimate of the packet's ordered Cube/Fixpipe schedule.
Here $(R_0,\lambda_0)$ is `sc_fp32_store` or `gm_write`, while
$(R_S,\lambda_S)$ is `sc_fp32_store_steady`. The measured steady demand
is charged in the output-pipeline throughput bound and one-piece
dependency bound; it is not a full-kernel correction. The generic write
proxy remains uncalibrated for FP32 atomic output.

The kernel starts the first packet's A and B copies before the K loop,
but also issues some slot copies inside that loop. During the last N
slot of a packet that is not the last, `ProcessBaseMNK` issues the next
packet's A and B copies. The following single overlap window is an
approximation of those staggered copy and compute operations.
The last slot has width

$$
n_{\rm tailtile}=n-\big(\lceil n/128\rceil-1\big)\,128,
$$

and Cube time $T_{\rm tail}(m,n,k)=2m\,n_{\rm tailtile}\,k/R_C$. A later
packet therefore adds only the part of its MTE2 that exceeds that slot:

$$
\begin{aligned}
T_K^{\rm red}(m,n)=&\;T_{\rm mte2}(m,n,k_0)\\
&+\sum_{p=0}^{P-2}\Big(
    T_{{\rm mac},p}^{\rm red}(m,n,k_p)
    +\max\!\big(0,\;T_{\rm mte2}(m,n,k_{p+1})-T_{\rm tail}(m,n,k_p)\big)
  \Big)\\
&+T_{{\rm mac},P-1}^{\rm red}(m,n,k_{P-1}).
\end{aligned}
$$

For one K packet the approximation is $T_{\rm mte2}+T_{{\rm mac},0}^{\rm red}$.
The placement of individual N-slot copies, MTE1 transfers, and Fixpipe
calls needs an operation-level dependency graph for a more precise
timeline. In particular, $T_{\rm mte2}$ is total copy service, not a
claim that every operand slot is fully copied before the first MAC.

## Cast and the M-panel pipeline

The outer loops match mode 2. A tile of M extent $m_t$ and N extent
$n_t$ has $J=\lceil m_t/I_M\rceil$ panels, $I_M=\mathrm{stepM}\cdot 128$.
Full panels have $I_M$ rows; the last keeps the remainder. Every panel
has $n_t$ columns. After the panel's K loop the AIC notifies, and two
AIVs cast the panel. The larger shard has $Q=\lceil m/2\rceil\,n_t$
elements.

$$
T_V(m,n)=
\lambda_{\rm sync}+\frac{4Q}{R_R}+\frac{2Q}{R_W}+\lambda_{\rm store}.
$$

$\lambda_{\rm sync}$ is `sync.fixed_latency_us`. $\lambda_{\rm store}$ is
`gm_write.fixed_latency_us`. The byte terms are the FP32 load and the
FP16/BF16 store at the existing GM rates. Optional `sc_cast` adds
$\lambda_{\rm cast}+Q/R_{\rm cast}$ to this service. The cast is
present in this FP16/BF16 scope.

Panels of all tiles on one AIC form one sequence. The cast of panel $j$
overlaps the K chain of panel $j+1$. For each AIC $q$, let
$K_{q,j}=T_K^{\rm red}(m_{q,j},n_{q,j})$ and
$U_{q,j}=T_V(m_{q,j},n_{q,j})$. The ideal two-stage pipeline is

$$
\boxed{\displaystyle
\begin{aligned}
C_{q,-1}&=V_{q,-1}=0,\\
C_{q,j}&=C_{q,j-1}+K_{q,j},\\
V_{q,j}&=\max\!\big(C_{q,j},V_{q,j-1}\big)+U_{q,j},\\
T_6&=L+\max_q V_{q,J_q-1}.
\end{aligned}}
$$

$L$ is the launch term from the
[BASE objective](SOLVE_BASE_FORMULATION.md#objective), plus A ND→NZ when
$K>65535$. One panel collapses to $L+T_K^{\rm red}+T_V$. Spatial counts
$N_o$, $r$, and $c_{\rm pre}$ are the ones in the mode 2 formulation:
the first $c_{\rm pre}$ AICs own
$r=\lceil N_o/c\rceil$ tiles.

## L1 resident bytes

The kernel cap with no bias is 512 KiB. Each A slot holds $128\cdot k_0$
FP16/BF16 elements, and two B slots hold the same for one N slot. The
emitted blocks resident:

| Block | A slots | bytes |
| --- | ---: | --- |
| MK33, $\kappa=384$ | 3 | $3\cdot 128\cdot 384\cdot 2+2\cdot 128\cdot 384\cdot 2=480\,\mathrm{KiB}$ |
| 2×4, $\kappa=512$ | 2 | $2\cdot 128\cdot 512\cdot 2+2\cdot 128\cdot 512\cdot 2=512\,\mathrm{KiB}$ |
| short side, $\kappa=512$ | 1 | $1\cdot 128\cdot 512\cdot 2+2\cdot 128\cdot 512\cdot 2=384\,\mathrm{KiB}$ |

$T_6$ assumes the packet fits this cap. The host's short-side tiling
still declares `depthA1=depthB1=8`, reserving 512 KiB of L1 even
though the kernel's modeled active slots total 384 KiB; the standalone
preflight uses the declared 512 KiB. L0C is double-buffered at
128 KiB in the kernel constants; its turnover is abstracted in
$T_{\rm mac}^{\rm red}$.

## Workspace

The host buffer is the same single FP32 image as modes 2 and 5, plus
20 MiB RPC. Mode 6 requires $N$ divisible by 128, which for FP16/BF16
is a 256-byte alignment, and the aligned kernel uses the real N as the
C stride. Host ND→NZ conversion buffers are additional when enabled:

$$
W=4MN+20\,\mathrm{MiB}+W_{\rm ND2NZ,A}+W_{\rm ND2NZ,B}.
$$

## What stays outside $T_6$

The host predicates that accept the mode are outside the expression, as
is the search that chose $(C_M,C_N)$ and the basic block. $D_6$ counts
each operand once per K packet, with no further L2-hit discount. The
tail-slot overlap approximates the prefetch `ProcessBaseMNK` issues
across its last N slot. Transpose selects the ND or NZ copy and the
inner-axis test. $D_6$ is the same either way. The single Cube/vector
flag is modeled as a two-stage producer-consumer pipeline. Flag reuse,
shared GM contention, and an ND versus NZ copy-rate difference remain
unresolved.
