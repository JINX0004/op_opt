# Single-core Split-K, mode 2: FP16/BF16/FP32 MKN reduced formulation

Mode 2 is `TilingEnableSplitCore::SINGLE_CORE_SPLIT_K`. The kernel is
`MatMulSingleCoreSplitKKernel` with `IS_NKM = false` and
`MM_CFG_PRELOAD_MK`, block `MatmulSingleCoreSplitKBaseBlock`. This
document gives the reduced service-time objective for that schedule.
`SolveReducedMip` evaluates source-admissible mode-2 candidates alongside
BASE and deterministic Split-K, then selects the smallest reduced objective.
The result is not an event-graph makespan or a device-time guarantee.

The scope is same-type FP16, BF16, or FP32 A/B/output, with FP32
accumulation, NN, ND/ND/ND, no bias, on the modeled 910B3 path. HF32 and
mixed-output types are outside this formulation. The implemented FP32
search currently excludes the converted-A path $K>65535$. The
FP16/BF16 timing expression describes N divisible by 128; its
unaligned and mixed ND→NZ variants need separate epilogues. The FP32
direct-output expression also covers the searched N multiples of 64.

The same family contains [mode 5](SOLVE_SC_SPLITK_MODE5_FORMULATION.md)
and [mode 6](SOLVE_SC_SPLITK_MODE6_FORMULATION.md). Deterministic
multi-core Split-K is a different kernel and is formulated in
[SolveSplit](SOLVE_SPLIT_FORMULATION.md).

## What the schedule does

AICs own disjoint spatial tiles of `singleCoreM` by `singleCoreN`. Each
AIC walks its tile along K in slices of `singleCoreK`. Slice 0 of a
panel Fixpipes FP32 with atomic add off; later slices atomic-add into
the same output accumulation. With FP16/BF16 output, that accumulation
resides in FP32 workspace. The AIC then signals two AIVs to cast the
panel to the output type, overlapping the next M panel's Cube work.
With FP32 output, `MatMulSingleCoreSplitKKernel::Process` calls the AIC
`UnAlignedProcess` path and returns on AIV: every `IterateAll` writes
directly to output GM, with no AIV cast stage. The host may still reserve
an output-sized workspace allocation; the modeled FP32 data path does
not charge a workspace read/cast/write.

## Geometry the host emits

`DoSingleCoreSplitKTiling` starts from `SetBasicBlockOfMK33`. With no
bias and with the current M/N tile product below the AIC count, it
switches to `SetBasicBlockOf24`. A side shorter than 384 then shrinks
the M step. `CheckSingleTilingOk` sets

$$
C_K=\mathrm{stepKa}\cdot B_K,\qquad
B_K=\frac{256\ \mathrm{bytes}}{b_{\rm in}}=
\begin{cases}
128,&\mathrm{FP16/BF16},\\
64,&\mathrm{FP32}.
\end{cases}
$$

$b_{\rm in}$ is the input element width. Base M and base N stay 128.
`dbL0c` is 2. The inner panel extents are $I_M=\mathrm{stepM}\cdot 128$
and $I_N=\mathrm{stepN}\cdot 128$.

| Block | stepM | stepN | stepKa | stepKb | $I_M$ | $I_N$ | FP16/BF16 $C_K$ | FP32 $C_K$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MK33 | 3 | 1 | 3 | 3 | 384 | 128 | 384 | 192 |
| 2×4 | 2 | 1 | 4 | 4 | 256 | 128 | 512 | 256 |
| short side | 1 | 1 | 4 | 4 | 128 | 128 | 512 | 256 |

Mode 2 keeps `l2Info.calOrder = 0`. `UpdateBlockCnt` then forces
`innerLoopN = 1`. `CalcGMOffset` treats that single N iteration as the
last inner block, so one `IterateAll` receives the tile's full N extent
(the N tail, on a tail tile). The M loop remains
$\lceil m/I_M\rceil$ panels. `iterateOrder` inside the matmul engine
stays `ITER_ROW_FIRST` for MK33; that order is inside one `IterateAll`
and is outside the panel count.

### FP32 search domain and feasibility

The current FP32 solver searches mode 2 with $B_M=B_N=128$ and $B_K=64$.
For 20 available AICs it generates source-aligned core extents from
partition counts $p_M,p_N\in\{1,\ldots,20\}$:

$$
\begin{aligned}
C_M(p_M)&=\min\!\left(M,\;8\left\lceil
       \frac{\lceil M/p_M\rceil}{8}\right\rceil\right),\\
C_N(p_N)&=\min\!\left(N,\;64\left\lceil
       \frac{\lceil N/p_N\rceil}{64}\right\rceil\right).
\end{aligned}
$$

The 8-element M alignment follows the source's 32-byte outer alignment
for FP32; the 64-element N alignment follows its 256-byte inner
alignment. This admits the measured $824\times2048\times192$ source
tile. It is a bounded source-informed search domain, not a claim that
the source host always chooses the minimizing partition. For each
geometry the solver evaluates occupied cores
$1\le c\le\min(20,N_o)$ and the applicable MK33, 2×4, or short-side
block pattern. $C_K=B_K\,\mathrm{stepKa}$ and the actual L0/L1 and
workspace capacities are checked before scoring.

The current preflight requires same-type FP32, NN, ND/ND/ND with no bias,
$M,N$ positive multiples of 16, $K$ a multiple of 64, $N$ a multiple
of 64, and $K\le65535$. The regular entry gate is $K\ge27392$;
the source's $K=1536$, $M=384$, very-large-N channel is also admitted
when its output fits the stated L2 fraction. Outside that special
channel, the source spatial gate requires $C_N\ge512$ and

$$
\rho=\frac{MN}{C_M C_N\cdot20}\ge0.7.
$$

For $896\le N\le2048$, it additionally requires $C_N>640$ and
$\rho\ge0.85$. These are source feasibility heuristics, not terms in
the objective. The solver's supported set is narrower than all host
entry routes: the source can also enter single-core Split-K through
L2-cache or transposed-layout conditions that this NN/ND model does
not represent. FP32 modes 5 and 6 are outside this search domain.

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
$q_k=\lceil k/B_K\rceil$. Let $r_A,r_B$ count actual GM→L1 reloads of
each operand within this panel. The source-shaped input count is

$$
D(m,n,k)=b_{\rm in}k\,(r_A m+r_B n).
$$

The previous no-reuse assumption $r_A=n_b,r_B=m_b$ remains in the
current C++ scorer. FP32 source-built mode-2 profiles of 128×768,
128×1024, 160×1024, and 208×1024 panels instead support
$r_A=r_B=1$ for the `MM_CFG_PRELOAD_MK` route: A and B are each read
once per inner-M panel and K slice, even when there are multiple base
pieces. Across the full shape, A repeats for distinct N core tiles and
B repeats for distinct M core tiles. The measured total differs from
this count by a fixed 320 KiB in the profiled full kernels; see the
[A/B traffic experiment](bench910/fp32_mode2_copy_route_v1/README.md).
Treat $r_A,r_B$ as source-path parameters until this reuse rule has
been checked for the other data types and tiling modes. The profiler
reports combined traffic, so the separate operand counts are an
inference from the source route and total, rather than individual
counter measurements. The equations below express the analytical
correction; numerical scores quoted from the current C++ implementation
still use its earlier no-reuse count.

$R_C$ is Cube FLOP/µs per AIC. $R_R$ and $\lambda_R$ are the GM-read
byte rate and fixed latency. Let $S_0$ be the plain FP32 Fixpipe
service for the first K slice. A profiled FP32-store service replaces
the generic GM-write rate and latency when present:

$$
S_0(m,n)=m_b n_b\lambda_0+\frac{4mn}{R_0}.
$$

For FP16/BF16, later slices retain the plain-store charge and add the
measured incremental atomic service. For direct-output FP32, the
measured steady-store primitive is a total atomic Fixpipe service;
it replaces, rather than supplements, the plain-store service:

$$
\begin{aligned}
S_p^{16}(m,n)&=S_0(m,n)+m_b n_b\lambda_A+\frac{4mn}{R_A},
&&p>0,\\
S_p^{32}(m,n)&=m_b n_b\lambda_{\rm steady}
                         +\frac{4mn}{R_{\rm steady}},
&&p>0.
\end{aligned}
$$

Here $\lambda_A,R_A$ describe the optional incremental
`sc_atomic_extra` service; without it, $S_p^{16}=S_0$. The
FP32 $\lambda_{\rm steady},R_{\rm steady}$ are
`sc_fp32_store_steady`; without that primitive, the model falls
back to the generic plain-store plus atomic-extra service. In both
paths $S_0$ uses `sc_fp32_store` when profiled, otherwise
`gm_write`. Define $S_p^{(d)}=S_0$ for $p=0$.

$$
\boxed{\displaystyle
T_{\rm slice}^{(d)}(m,n,k_p,p)=
\max\!\left(
\frac{2mnk_p}{R_C},\;
\frac{D(m,n,k_p)}{R_R}
 +2m_b n_b\left\lceil\frac{k_p}{B_K}\right\rceil\lambda_R
 +S_p^{(d)}(m,n)
\right).}
$$

The $m_b n_b$ write startups represent one Fixpipe per base tile.
Operand reads and Fixpipe sit inside one `IterateAll`, forming
the memory branch of that slice. Successive K slices are successive
`IterateAll` calls and add. FP32 output is already in GM after
each call; FP16/BF16 accumulates in FP32 workspace for the later cast.
The maximum assumes that the Cube branch can fully overlap the combined
read/output branch. It does not explicitly schedule L1→L0 transfers,
scalar setup, or the dependency between one `IterateAll` and the next.

## Panel timeline by output dtype

Each panel's K chain adds its sequential `IterateAll` slices:

$$
T_K^{(d)}(m,n)=
\sum_{p=0}^{P-1}T_{\rm slice}^{(d)}(m,n,k_p,p).
$$

On a tile of M extent $m_t$ and N extent $n_t$, there are
$J=\lceil m_t/I_M\rceil$ panels. Panel $j<J-1$ has $I_M$ rows; the last
has $m_t-(J-1)I_M$ rows. Every panel has $n_t$ columns. Concatenate a
core's panels in tile-index order. Let $J_q$ be core $q$'s panel count
and $(m_{q,j},n_{q,j})$ its panel extents.

### FP16/BF16: Cube and AIV cast pipeline

The panel's cast starts after its K chain at `NotifyEvent`.
Two AIVs split its rows; the larger shard has
$Q=\lceil m/2\rceil n$ elements. Its service is

$$
T_V^{16}(m,n)=
\lambda_{\rm sync}+\frac{4Q}{R_R}
 +\frac{2Q}{R_W}+\lambda_{\rm store}
 +\lambda_{\rm cast}+\frac{Q}{R_{\rm cast}}.
$$

$\lambda_{\rm sync}$ is `sync.fixed_latency_us` and
$\lambda_{\rm store}$ is `gm_write.fixed_latency_us`.
The $4Q$ and $2Q$ terms charge the FP32 workspace load and FP16/BF16
output store. The last two terms come from the optional `sc_cast`
primitive and are zero when it is unavailable. That fallback can miss
cast compute. The [910B3 single-core probe](bench910/SINGLE_CORE_SPLITK_PRIMITIVES_910B3.md)
reports the exploratory cast and atomic primitives.

The cast of one panel overlaps the next panel's Cube chain, including
across tile boundaries inside `realRound`. Set
$K_{q,j}=T_K^{16}(m_{q,j},n_{q,j})$ and
$U_{q,j}=T_V^{16}(m_{q,j},n_{q,j})$. The ideal two-stage recurrence is

$$
\boxed{\displaystyle
\begin{aligned}
C_{q,-1}&=V_{q,-1}=0,\\
C_{q,j}&=C_{q,j-1}+K_{q,j},\\
V_{q,j}&=\max\!\big(C_{q,j},V_{q,j-1}\big)+U_{q,j},\\
T_2^{16}&=L+\max_q V_{q,J_q-1}.
\end{aligned}}
$$

### FP32: direct AIC output

For FP32, `Process` returns on AIV and each AIC writes
directly to output GM. There is no cast or vector pipeline; the
per-slice output service is already in $T_K^{32}$. The reduced
makespan is

$$
\boxed{\displaystyle
T_2^{32}=L+\max_q\sum_{j=0}^{J_q-1}
T_K^{32}(m_{q,j},n_{q,j}).}
$$

In both cases $L$ includes launch latency from the
[BASE objective](SOLVE_BASE_FORMULATION.md#objective).
The FP16/BF16 path additionally charges A ND→NZ workspace conversion
when $K>65535$; the current FP32 solver rejects that converted-A
route. The maximum takes the busiest of the occupied AICs, including
M/N tail panels. For full spatial tiles, the front cores owning $r$
tiles are the busiest.

## Workspace

Host `CalcSplitCoreWorkspace` reserves one output-sized FP32 buffer
and the 20 MiB RPC buffer for this family, even when FP32 output bypasses
the workspace in the kernel. With input element width $b_{\rm in}$,
the host's 256-byte alignment in N is
$a_N=256/b_{\rm in}$: 128 elements for FP16/BF16 and 64 for FP32.
Host ND→NZ conversion buffers add $W_{\rm ND2NZ,A}$ and
$W_{\rm ND2NZ,B}$ when their flags are set:

$$
W=4M a_N\left\lceil N/a_N\right\rceil+20\,\mathrm{MiB}
  +W_{\rm ND2NZ,A}+W_{\rm ND2NZ,B}.
$$

For FP16/BF16, N divisible by 128 uses the real N as the Cube C stride.
Otherwise the block aligns that stride to 64 FP32 elements, and
`NNot128AlignProcess` runs the Cube loop followed by vector cast
or padding removal. The FP16/BF16 $T_2^{16}$ above represents the
aligned, overlapped schedule. For FP32, `SetOrgShape` uses real N
and both the N-divisible-by-128 and other N-multiple-of-64 branches
write directly from AIC. Thus $T_2^{32}$ has no vector epilogue for
either branch.

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

$T_{\rm slice}^{(d)}$ does not apply L2 reuse, the preload-MK depth, or
the bias load on slice 0. Cache-hint changes and transpose changes of
the physical HBM stream are outside the byte count. The FP16/BF16
Cube/vector flag is modeled as a two-stage producer-consumer pipeline;
the recurrence does not resolve flag reuse or shared GM contention
between active AICs and AIVs. The FP32 objective uses a mode-6
source-in-context atomic-output rate as a provisional mode-2 proxy.
It does not yet capture all source mode-2 setup or contention costs.
In particular, the `max` in $T_{\rm slice}^{32}$ should be treated as
an optimistic service bound when the hardware cannot overlap those
branches across the short K slices.

### Full FP32 mode-2 event graph (diagnostic prototype)

Keep the current byte counts, Cube work, source geometry, and AIC
assignment. Add a **source-level exposed service** $H_{j,p}^{32}\ge0$
for panel $j$ and K slice $p$ on an AIC. It represents only time on the
critical path beyond the overlap already credited by the current
slice expression: `SetSingleShape`/tensor setup, `IterateAll` handoff,
L1→L0 or Fixpipe waits, and any read service missed by the generic
traffic model. A first diagnostic extension is

$$
\begin{aligned}
\widetilde T_{\rm slice}^{32}(m,n,k_p,p;\mathcal W)
  &=T_{\rm slice}^{32}(m,n,k_p,p)
    +H^{32}(m,n,k_p,p;\mathcal W),\\
\widetilde T_2^{32}
  &=L+\max_q\sum_{j=0}^{J_q-1}\sum_{p=0}^{P-1}
       \widetilde T_{\rm slice}^{32}(m_{q,j},n_{q,j},k_p,p;\mathcal W).
\end{aligned}
$$

$\mathcal W$ includes full K, physical A/B row strides, and source L2
cache hints; these can affect transfer time even when the slice size
and tile are unchanged.
$H^{32}$ is **not currently a measured primitive or a fitted constant**;
both implemented objectives omit this correction. The intended full model
should instead resolve the actual GM→L1, L1→L0, MAC, Fixpipe, and
inter-`IterateAll` precedence with measured primitive times. A
geometry-independent per-slice correction is not justified by the
available profiles. Nor can the existing mode-5
`sc_mode5_iterate` coefficient be transferred unchanged to mode 2.
The profiled `aic_mac_time`, `aic_mte1_time`, `aic_mte2_time`,
`aic_fixpipe_time`, and `aic_scalar_time` overlap; their sum must not
be used as $H^{32}$. The fixed-tile K sweep below establishes geometry
dependence, but source-shaped primitive measurements are still needed
to isolate handoff time from transfer and output stalls.

### Opt-in reduced FP32 mode-2 pipeline

An [experimental C++ profile](bench910/fp32_mode2_sweep_v1/README.md#modeling-implication)
adds explicit L1→L0 A/B services and an output drain without fitting a
whole-kernel residual. It is enabled only when both optional
`sc_mode2_l0a` and `sc_mode2_l0b` primitives are supplied. The default
profile and online family selector retain $T_{\rm slice}^{32}$ above.
For one slice, define

$$
\begin{aligned}
R_p &= \frac{D(m,n,k_p)}{R_R}
       +2m_bn_bq_k\lambda_R,\\
L_{A,p} &= \frac{b_{\rm in}k_pmn_b}{R_{0A}}
          +m_bn_bq_k\lambda_{0A},\\
L_{B,p} &= \frac{b_{\rm in}k_pnm_b}{R_{0B}}
          +m_bn_bq_k\lambda_{0B},\\
C_p &= \frac{2mnk_p}{R_C},\\
F_p &= S_p^{32}(m,n),\\
F_{1,p} &= S_p^{32}\!\left(\min(m,128),\min(n,128)
                   \right)\text{ with one output packet},\\
\boxed{T_{\rm slice,pipe}^{32}}
    &= \max\!\big(C_p+F_{1,p},\;
                   R_p+L_{A,p}+L_{B,p}+F_{1,p},\;
                   F_p\big).
\end{aligned}
$$

The three paths are Cube followed by one output-piece drain,
GM→L1 followed by the A/B L1→L0 loads and the same drain, and total
output throughput. The A/B load services are serial on this
conservative source-level path; input and Cube can overlap across
base pieces. The output primitive still comes from mode 6 and is
provisional for mode 2. This is a compact max-plus **schedule
surrogate**, not the hidden Matmul engine's exact base-piece event
graph. It preserves the existing closed-form repetition of full K
slices and does not add constraints proportional to K.

The opt-in profile uses Huawei's nominal 910B3 $R_{0A}=460{,}800$
and $R_{0B}=230{,}400$ bytes/µs per AIC, with zero nominal fixed
latencies. These are hardware link ceilings, not independently
measured mode-2 primitives. On the 32 audited source tiles, mean
absolute timing error falls from 18.21% to 3.22%; on the 28 cases
excluding the K=49152 transfer anomaly, from 17.76% to 2.59%.
Seven additional source-emitted tiles outside those K series improve
from 9.88% to 5.68% mean absolute error. The result is encouraging
but still underestimates several wide and multi-panel tiles, so it
remains opt-in.

A [standalone packet-event evaluator](bench910/fp32_mode2_timeline_v1/README.md)
now schedules GM→L1 copy, L1→L0A/B loads, Cube work, and first/atomic
Fixpipe writes with explicit engine and double-buffer dependencies. It
uses a separately measured packet-barrier copy probe for the input
duration, not a full-kernel fitted correction. On four fixed-work
192-KiB-pitch holdouts, its mean absolute center-pitch error is 219 µs
versus 578 µs for the copy probe alone, but it still misses the
geometry-dependent pitch response. This is a diagnostic prototype,
not the online objective or a validated full mode-2 event graph.

### Next formulation change: route-aware MTE2 service

The [fixed-tile FP32 mode-2 experiment](bench910/fp32_base_formula_v2/MODE2_GAP_DIAGNOSIS.md)
holds the 128×N×256 tile and 20 AICs fixed at K=32512, 32768, and
33024. At K=32768, kernel time exceeds the neighbors' interpolation
by 574–578 µs and MTE2 busy time by 582–593 µs, while MAC time does
not spike. Requested GM→L1 bytes and aggregate L2 read hit fraction
change smoothly. All six tilings have `nd2nzA=nd2nzB=0` and the same
tiling key; those flags exclude a separate vector conversion pass,
but the ND-input Matmul path may still convert inline on MTE2. The
previous A-only copy probe also used inline ND→NZ, so it did not
separate transfer from conversion.

The [matched A-only copy probe](bench910/fp32_mode2_copy_route_v1/README.md)
varied 127/128/129-KiB physical A pitch at fixed logical K and byte
count. At 20 AICs, the center pitch adds 38 µs with inline ND→NZ and
65 µs with plain 2D copy, whereas one AIC has no center penalty. The
additional A-only sweep shows that this effect depends on occupancy:
inline conversion first shows a positive center excess at 12 AICs;
plain copy does so at two AICs. These are measured service differences,
not a K-specific term in the objective.

A source-shaped probe then interleaved one A load with each panel's B
base-piece loads. At 20 AICs, the 128-KiB center penalty was 545–551 µs
for either A route, compared with the full kernel's 574–578 µs. Probe
input-service time at the center was 2084/2496 µs for N=768/1024,
against full-kernel MTE2 busy times of 2081/2476 µs. Its GM→L1 byte
count exactly matches $b_{\rm in}K(M+cN)$ at $c=20$, and its aggregate
L2 hit fraction is unchanged by the pitch. The A-only experiment thus
understates the penalty in the actual A/B working set; adding an
independent A-only correction to the kernel would be wrong.

Replace the single generic $D/R_R$ term with a measured, route- and
context-aware input service. For a panel/slice, let

$$
\begin{aligned}
D_A&=b_{\rm in}k_p r_A m,
&s_A&=b_{\rm in}K,\\
D_B&=b_{\rm in}k_p r_B n,
&s_B&=b_{\rm in}N,\\
R_{AB}&=\Phi\!\left(
  \{(X_i,d_i,s_i,\tau_i,h_i)\}_{i=1}^{N_{AB}},
  g,c,w\right).
\end{aligned}
$$

$X_i\in\{A,B\}$ labels the $i$th source-ordered input packet,
$d_i$ its requested bytes, $s_i$ its physical row pitch, $\tau_i$ its
copy route, and $h_i$ its cache hint. Geometry $g$ includes panel/base
dimensions and packet order; $c$ is the simultaneously occupied AIC
count; $w$ describes concurrent working-set reuse. $\Phi$ is the
elapsed input service under this schedule, not an independent A-rate
plus B-rate. A compact implementation can use a measured effective
rate for the entire packet stream, with packet startup terms, provided
the rate is calibrated under the same occupancy and working-set
context. It must use **per-AIC traffic with a per-AIC rate** or total
traffic with an aggregate rate. The current C++ scorer's generic
aggregate read rate applied inside each core's slice violates this
unit convention. This service must pass held-out pitch and core-geometry
checks before $\Phi$ can safely drive family selection.

The [192-KiB held-out experiment](bench910/fp32_mode2_transfer_holdout_v1/README.md)
has now tested that condition. It held the probe's logical K and
requested bytes fixed while varying A's pitch through 191/192/193 KiB,
and paired five audited source tiles, including an 824×2048 tile with
three inner-M panels. The probe's center excess was 698–1116 µs on
the four one-panel geometries, whereas the full kernel's MTE2 busy
excess was only 48–477 µs. On the three-panel tile the probe added
4230 µs, the kernel added 707 µs, and reported MTE2 busy time
**decreased** by 3024 µs. Requested GM→L1 bytes match between source
and probe after scaling for logical K, within the same 320-KiB fixed
overhead, and each arm's aggregate L2 hit fraction is nearly unchanged
across the pitch pair. Thus the $r_A,r_B$ traffic correction survives
this holdout, but a $\Phi$ calibrated from **serialized copy time**
does not. Reported MTE2 busy cycles can also move opposite to elapsed
kernel time, so they cannot be treated as $R_{AB}$ without modeling
stalls and dependencies. Packet- and panel-synchronized copy schedules
were therefore tested before changing the online scorer.

The [synchronization follow-up](bench910/fp32_mode2_sync_holdout_v1/README.md)
did that copy-only comparison at every held-out geometry. Moving the
MTE2 barrier from every copy to each K packet or each inner-M panel
reduced probe baseline time, but changed the 192-KiB center excess
by at most 34 µs per geometry. The probe still overpredicted kernel
center excess by 0.5–3.5 ms. Profiling representative one- and
three-panel tiles confirmed identical requested GM→L1 bytes and
aggregate L2 hit fractions in all three modes. Hence barrier
granularity alone is not the missing element of $\Phi$; the next
model must represent operand consumption and the waits/overlap among
MTE2, L1→L0, Cube, and Fixpipe rather than fitting copy-only time.

The later [A/B producer-consumer follow-up](bench910/fp32_mode2_consumer_probe_v1/README.md)
separates A-only, B-only, and combined copies, then adds an explicit
L1→L0 handoff and L1-reuse wait. Across four fixed-work geometries,
B-only has near-zero center-pitch excess, while the combined A+B
excess remains 670–1082 µs with consumption. On the three valid
A-only geometries, its 142–162 µs excess plus B-only excess is far
smaller than the combined excess. This rules out the byte-proportional
allocation of one A+B probe time as a measured A/B primitive. A
separate profile of the real fixed-work source kernel finds 53–462 µs
MTE2 busy excess and essentially flat MAC work. The stage busy
counters overlap, and the controlled handoff still omits Cube and
Fixpipe, so neither provides a complete event precedence graph.

The earlier copy probes also used an A-cache-disable heuristic when
`singleCoreN == N`. The actual mode-2 device source does not disable
either operand's cache for these four tiles: its A condition requires
the NKM path and `stepN * baseN == N`, while its B condition requires
the MKN path and `stepM * baseM == M`. A
[matched-hint rerun](bench910/fp32_mode2_cache_policy_v1/README.md)
of the packet-barrier probe changes the 192-KiB center excess by only
7–17 µs across the four geometries. The several-hundred-microsecond
probe-versus-kernel gap therefore persists with matching cache hints.
The subsequent [actual-kernel 2×2 A/B hint experiment](bench910/fp32_mode2_cache_policy_v1/README.md)
keeps the emitted tile and logical work fixed. Across four geometries,
the four-round-median 192-KiB excess remains 73–672 µs under the four
hint combinations; no policy consistently removes it. Median A/B
effects are small relative to the earlier copy-probe versus kernel
gap and have broad, often sign-changing paired-round ranges. Moreover,
the research binary's policy-0 behavior does not always match the
interleaved original package at this sensitive pitch. Cache hints are
therefore a workload input to preserve in $\Phi$, not a measured
standalone correction that can replace the missing event dependencies.

The [source-shaped stage ladder](bench910/fp32_mode2_stage_ladder_v1/README.md)
then added L1→L0, Cube, and FP32 Fixpipe work to the same A/B copy
sequence at fixed logical K and tile. Its requested GM→L1 input count
is identical across modes, and aggregate L2 read-hit fractions stay
constant to roughly 0.0022 percentage points. Nevertheless, adding
Cube changes the MTE2 busy-time 192-KiB center excess from 1089 to
446 µs for 3072×1024, but only from 664 to 639 µs for 2048×2048;
adding Fixpipe changes them to 27 and 484 µs. The paired actual-kernel
excesses are 739 and 27 µs. Thus no stage in this **serialized,
synthetic** ladder reproduces both source geometries. More
importantly, input-service time is not determined by bytes, pitch,
cache hint, and aggregate L2 hit rate alone: the request issue/wait
schedule changes it. The exact queue or bank mechanism remains
unidentified by these counters.

In $R_{AB}=\Phi(\cdot)$, represent each request's issue time and the
buffer/queue state as endogenous schedule variables rather than
assigning one duration before scheduling. For example, extend each
packet tuple to $(X_i,d_i,s_i,\tau_i,h_i,t_i^{\rm issue})$ and let its
service depend on the recent inter-request gaps and concurrent output
activity. This is a modeling requirement inferred from the holdout,
not a calibrated service law. The ladder's explicit per-fragment drain
and replayed L0 operands prevent using its measured rates directly in
the product selector.

Jointly refine that input service and replace the optimistic
`one-piece` drain in $T_{\rm slice,pipe}^{32}$ with a compact max-plus
schedule for source-ordered base pieces: MTE2, L1→L0, MAC, and
Fixpipe completion each have their own recurrence, with dependencies
for buffer reuse and atomic updates across K slices. The objective
remains the maximum completion time over AICs. This separates a
roughly 575-µs exact-pitch penalty from the 220–240-µs baseline gap
observed at neighboring K. Validate on the earlier 32 source tiles,
the fixed-tile K neighbors, and held-out pitch/core-geometry cases
before enabling an FP32 family decision.

## FP32 worked example and measured check

For $M=2048$, $N=1536$, $K=32768$, the source and the FP32 search both
use mode 2 with $(C_M,C_N,C_K)=(208,768,256)$,
$(B_M,B_N,B_K)=(128,128,64)$, stepM $=2$, stepN $=1$, and 20 AICs.
There are $\lceil2048/208\rceil\lceil1536/768\rceil=10\cdot2=20$
spatial tiles, one per AIC. The last M tile has 176 rows. Since
$I_M=256$, each tile is one panel, and $P=32768/256=128$ K slices.
The largest panel, $(m,n)=(208,768)$, sets the makespan.

For that panel $m_b=2$, $n_b=6$, $q_k=4$. The **current no-reuse scorer
charges** $D=2{,}850{,}816$ input bytes per full K slice; this is not
the profiled GM→L1 request count. The slice performs
$2mnk=81{,}788{,}928$ Cube FLOPs and writes $4mn=638{,}976$
output bytes. Using the
measured profile $R_C=3{,}600{,}000$ FLOP/µs,
$R_R=3{,}710{,}212$ bytes/µs, $\lambda_R=0.109620$ µs,
$(\lambda_0,R_0)=(0.2432\ {\rm µs},34{,}190\ {\rm bytes/µs})$,
and $(\lambda_{\rm steady},R_{\rm steady})=
(0,91{,}000\ {\rm bytes/µs})$ gives

$$
\begin{aligned}
T_C&=81{,}788{,}928/3{,}600{,}000\approx22.719\ {\rm µs},\\
T_R&=2{,}850{,}816/3{,}710{,}212
       +2(2)(6)(4)(0.109620)\approx11.292\ {\rm µs},\\
S_0&=12(0.2432)+638{,}976/34{,}190\approx21.607\ {\rm µs},\\
S_{p>0}^{32}&=638{,}976/91{,}000\approx7.022\ {\rm µs},\\
T_{\rm slice}^{32}(p=0)&=\max(22.719,11.292+21.607)
                         \approx32.899\ {\rm µs},\\
T_{\rm slice}^{32}(p>0)&=\max(22.719,11.292+7.022)
                         \approx22.719\ {\rm µs},\\
T_2^{32}&=2.68+32.899263+127(22.719147)
                         \approx2920.911\ {\rm µs}.
\end{aligned}
$$

The [FP32 mode-2 experiment](bench910/fp32_mode2_v1/README.md)
measured 3274.14 µs for this same master family and tile on device 4.
The reduced objective underestimates it by 10.8%. Across 11 master
mode-2 source tiles, its mean absolute relative error is 13.0%.
Those measurements validate family and source-tile feasibility, while
the score still needs a dedicated mode-2 schedule calibration. The
100-case comparison forced the selected family and let MatMulV3 emit
the tile; it did not benchmark every tile proposed by this search.

### 2816×1024×36864: a wrong family ranking

The [fresh paired experiment](bench910/fp32_base_calls_v1/README.md)
compares the exact source-emitted families. The source BASE score is
3358.12 µs and its forced measurement is 3321.22 µs. Mode 2 scores
3098.38 µs but measures 3928.70 µs, so the objective selects the
slower family. Both master and the forced mode-2 run emit the same tile:
$(C_M,C_N,C_K)=(144,1024,256)$, $(B_M,B_N,B_K)=(128,128,64)$,
`stepM=2`, and 20 AICs. There are exactly 20 spatial tiles and
$P=36864/256=144$ K slices on the critical AIC. Its panel is
144×1024, with $m_b=2$, $n_b=8$, and $q_k=4$.

For each full slice, the default formulation gives

$$
\begin{aligned}
T_C &=\frac{2(144)(1024)(256)}{3{,}600{,}000}
       \approx20.972\ {\rm µs},\\
T_R &=\frac{4(256)(144\cdot8+1024\cdot2)}{3{,}710{,}212}
       +2(2)(8)(4)(0.109620)
       \approx14.915\ {\rm µs},\\
S_0 &=16(0.2432)+\frac{4(144)(1024)}{34{,}190}
       \approx21.142\ {\rm µs},\\
S_{p>0}^{32} &=\frac{4(144)(1024)}{91{,}000}
       \approx6.482\ {\rm µs},\\
T_{\rm slice}^{32}(0)&=\max(20.972,14.915+21.142)
       \approx36.057\ {\rm µs},\\
T_{\rm slice}^{32}(p>0)&=\max(20.972,14.915+6.482)
       \approx21.396\ {\rm µs},\\
T_2^{32}&=2.68+36.057+143(21.396)
       \approx3098.38\ {\rm µs}.
\end{aligned}
$$

The profiler reports about 3026 µs MAC time, close to the modeled
3020 µs Cube service, but about 3835 µs active AIC time and 3929 µs
kernel task time. The active-AIC excess over MAC is approximately
809 µs, or 5.62 µs per K slice; task time exceeds active-AIC time by
about 94 µs. These are **not additive corrections** to the current
score, which already includes some non-MAC service. They show that
incorrect Cube throughput or FP32 element size is not the dominant
issue. The missing critical-path overlap and handoff behavior is the
likely model defect; aggregate profiler counters cannot identify the
exact wait edge.

The 2048×2048×27392 mode-2 tile has 107 K slices and a smaller
measured-minus-scored gap of about 2.34 µs per slice. Thus using the
5.62 µs residual above as a universal $H^{32}$ would be unjustified.
The opt-in pipeline instead scores the 2816×1024×36864 source tile
at 3934.88 µs, above the 3358.12 µs source BASE score, correcting
that family ranking. It still scores the 2048×2048×27392 source mode-2
tile at 3322.44 µs, below its 3493.27 µs source BASE score, although
the measured BASE is slightly faster. The opt-in model therefore does
not yet support enabling FP32 mode-2 family decisions online.

### Fixed-tile K sweep on 910B3

The [32-shape source-kernel sweep](bench910/fp32_mode2_sweep_v1/README.md)
holds each emitted mode-2 M/N tile fixed while varying K. All four
series use 20 AICs, $C_K=256$, and one output tile per AIC. Excluding
K=49152, their median measured-minus-scored residuals per K slice are
5.96 µs for 144×1024, 5.08 µs for 160×1024, 2.33 µs for
208×1024, and 7.35 µs for 144×1536. This confirms that a constant
`IterateAll` correction would not generalize across geometry.

At K=49152, the measured kernel exceeds linear interpolation between
nearby K values by about 413, 649, and 506 µs for the first, second,
and fourth geometries; MTE2 time rises at the same point. The
208×1024 geometry has a smaller 60 µs excess. The source tile remains
fixed across each comparison. On this device, $K\cdot4=192$ KiB is the
FP32 A-row stride at that K and L2 capacity is 192 MiB; however, B
footprint differs among the series, and the measured counters cannot
identify a unique cache mechanism. Therefore $H^{32}$, or its eventual
event-graph replacement, needs workload-stride and cache-policy inputs
as well as tile geometry. The K=49152 anomaly should be isolated with
the same concurrency and route controls at 192 KiB before a transfer
rate class is assigned. The later [K=32768 fixed-tile experiment](bench910/fp32_base_formula_v2/MODE2_GAP_DIAGNOSIS.md#fixed-tile-k-neighbor-experiment)
shows a corresponding MTE2 spike with smooth aggregate L2 hit rate.
The [matched A-copy probe](bench910/fp32_mode2_copy_route_v1/README.md)
shows that the 128-KiB pitch penalty also occurs without ND→NZ
conversion when 20 AICs read concurrently; it does not identify the
physical source of the stall.

The [fixed-work in-context pitch experiment](bench910/fp32_mode2_in_context_pitch_v1/README.md)
keeps logical K and B fixed while varying A's physical pitch through
an isolated research build of the actual mode-2 kernel. For four
correct, one-panel tiles, its 192-KiB center excess is 417, 597, 78,
and 415 µs, within 9–28 µs of the earlier full-kernel sweep's
439, 585, 69, and 442 µs. The serialized copy-only probe gives
987, 1159, 706, and 967 µs for the same geometries. Thus the prior
source-kernel pitch excess is not mainly an artifact of changing
logical K or B footprint. The event model must account for the
`MatmulImpl::IterateAll` producer/consumer overlap and waits before
using an A/B copy rate on the critical path. A fifth, wide-N,
three-panel tile faults with the experimental `Ka != Kb` stride path,
so the fixed-work conclusion has not been validated for that tile.
