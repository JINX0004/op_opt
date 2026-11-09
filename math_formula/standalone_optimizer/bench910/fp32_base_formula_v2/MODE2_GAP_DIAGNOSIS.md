# FP32 mode-2 gap for one-base-row core tiles

The standalone FP32 mode-2 objective selected two exact tiles that ran
slower than deterministic Split-K. Both tiles were accepted by the 910B3
CANN host tiler, emitted the requested fields, and produced correct FP32
output. [Paired profiles](exact_mode2_tile_records.json) used two reversed
arm orders on devices 6 and 7; [AIC counters](exact_mode2_tile_counters.json)
are medians of the same five kernel tasks in each round, then medians
across rounds. Stage busy times overlap and are not additive runtimes.

| M×N×K | Exact core M×N×K | Model µs | Kernel µs | MAC busy µs | MTE2 busy µs | Fixpipe busy µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2560×768×32768 | 128×768×256 | 1885.9 | 2703.9 | 1752.1 | **2076.8** | 730.7 |
| 2560×1024×32768 | 128×1024×256 | 2482.4 | 3277.7 | 2334.6 | **2471.8** | 925.8 |

Both execute 128 sequential K packets per AIC. The current
[`ScMode2PipelineSliceUs`](../fp32_mode2_v1/remote_selector/model/mip_solver.cpp)
uses

$$
T_\text{slice}=\max\{C+F_1,\;R_{\rm GM\to L1}+R_{\rm L1\to L0}+F_1,\;F\},
$$

where $F_1$ is the output drain for just one 128×128 piece and $F$ is
the entire slice's output service. It assumes the remaining output
service overlaps the input/Cube path. The model's per-slice components
for the two exact tiles are:

| Core M×N | GM→L1 µs | L1→L0 µs | Cube µs | Full output µs | One-piece drain µs | Chosen path µs | Observed task µs/slice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128×768 | 5.69 | 5.12 | 13.98 | 4.32 | 0.72 | 14.70 | 21.12 |
| 128×1024 | 7.58 | 6.83 | 18.64 | 5.76 | 0.72 | 19.36 | 25.61 |

The Cube term is close to the measured MAC busy time: 1790 versus
1752 µs and 2386 versus 2335 µs over the full kernel. The score is
wrong because it classifies both slices as Cube-dominant, while
measured MTE2 busy time exceeds MAC by **19%** and **6%**. The earlier
[32-case FP32 mode-2 sweep](../fp32_mode2_sweep_v1/README.md) contained
only `singleCoreM≥144` tiles; MTE2/MAC was **0.29–0.67** in all 32.
This `singleCoreM=128` regime was outside that validation set.

The generic `gm_read` primitive is **3.71 MB/µs aggregate across 20
AICs**, as recorded in [the primitive measurements](../PRIMITIVE_910B3_CANN9.md).
Mode 2 currently divides one core's modeled input bytes by that
aggregate rate inside a per-core slice. This is a unit/occupancy
mismatch. Its modeled GM→L1 term is just 5.69 or 7.58 µs per K packet,
whereas the profiler reports MTE2 busy for about 16.23 or 19.31 µs
per packet. These are different quantities: MTE2 busy includes source
access geometry and stalls, so the ratio is diagnostic, not a measured
read-rate conversion.

The output overlap assumption is also too optimistic in this regime.
The source [loops over K packets and invokes `IterateAll` with atomic
output](../../../../mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel.h) for
each packet. The measured Fixpipe busy time is 731 or 926 µs, while
the current critical Cube path exposes only one 128×128 drain per
packet, about 92 µs in total. A diagnostic lower-bound combination
`max(measured MAC, measured MTE2) + measured Fixpipe` is 2807 and
3398 µs, within 4% of the actual 2704 and 3278 µs. This combination
uses measured counters and is **not** a deployable predictor; the
counter groups can overlap.

Two simple global changes were checked offline against all 32 prior
source-tile profiles. Replacing the mode-2 GM read rate with the
aggregate rate divided by 20 predicts the two exact tiles within
about 4% mean absolute error but raises the old sweep's error from
**3.22% to 57.47%**. Fully serializing the modeled output term leaves
8.61% error on the exact pair but raises the sweep's error to **20.87%**.
The source's effective input bytes, cache behavior, and overlap therefore
need a geometry-aware model; a uniform correction would damage existing
cases.

## Fixed-tile K-neighbor experiment

The [neighbor benchmark](measure_exact_mode2_k_neighbors.py) forced the
same mode-2 tile, `singleCoreM=128`, `singleCoreN=N`,
`singleCoreK=256`, `baseM=baseN=128`, `baseK=64`, and 20 AICs, at
K=32512, 32768, and 33024. It ran two rounds in opposite K order on
devices 6 and 7. All six tilings were audited and outputs passed the
correctness check. These are medians from the [raw records](exact_mode2_k_neighbor_records.json):

| N | K | Kernel µs | MAC busy µs | MTE2 busy µs | Fixpipe busy µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 768 | 32512 | 2109.3 | 1738.4 | 1485.7 | 541.9 |
| 768 | 32768 | **2702.9** | 1752.1 | **2081.3** | 731.2 |
| 768 | 33024 | 2147.5 | 1765.8 | 1512.8 | 552.0 |
| 1024 | 32512 | 2685.2 | 2316.4 | 1869.4 | 739.3 |
| 1024 | 32768 | **3283.1** | 2334.6 | **2475.8** | 924.7 |
| 1024 | 33024 | 2724.2 | 2352.9 | 1895.9 | 746.6 |

Relative to interpolation of the two neighbors, the K=32768 kernel
adds **574.5/578.4 µs** at N=768/1024. MTE2 adds **582.1/593.1 µs**;
MAC has no corresponding excess. A's FP32 row pitch is exactly
`32768 * 4 = 131072` bytes (128 KiB) at the center, versus 127/129 KiB
at the neighbors. Thus, the fixed-tile slowdown is tied to this exact
row pitch, not merely to increasing K or the one-base-row schedule.
The smooth model predicts 1885.9/2482.4 µs at the center, so this
isolated pitch penalty explains about **70%/72%** of its observed
817/801 µs underestimation. The remaining roughly 220–240 µs is a
baseline mode-2 modeling gap even at the adjacent pitches.

The separate [L2 and memory profiles](exact_mode2_l2_metrics.json),
collected with [this driver](profile_exact_mode2_l2.py), show that
reported GM→L1 traffic grows linearly with K. Aggregate AIC L2 read
hit fractions remain **81.429%** for N=768 and **84.445%** for N=1024
at all three pitches (rounded to three decimals). The L2 counters
combine A and B; they do not reveal a per-operand latency or bandwidth.
They do, however, rule out a large *aggregate L2 hit-rate collapse* as
the explanation for the 575 µs center penalty.

## Inline ND-to-NZ versus plain GM-to-L1 copy

All six audited mode-2 tilings have `nd2nzA=nd2nzB=0` and the same
tiling key, 65568. This rules out a *separate vector/GM preprocessing
path changing at K=32768*. It does **not** rule out inline ND-to-NZ
conversion during the Matmul API's GM→L1 MTE2 load: the kernel passes
ND GM tensors to `MatmulImpl::SetTensorA/B` in
[`mat_mul_sc_splitk_kernel.h`](../../../../mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel.h).
[Huawei's Matmul API example](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_10014.html)
describes its inline ND-to-NZ MTE2 path. All three K
values and both N values are 256-byte aligned in FP32, so the documented
non-aligned-input case does not explain this particular spike.

The earlier [BASE A-only copy probe](source_stream_probe/probe_kernel.cpp)
used `DataCopy` with `Nd2NzParams`, so it measured *combined A transfer
and inline conversion*, not raw GM read time. A later
[matched copy-route experiment](../fp32_mode2_copy_route_v1/README.md)
held logical K=32256 and requested bytes fixed while varying physical
A row pitch through 127, 128, and 129 KiB. On 20 AICs, the 128-KiB
pitch adds **38 µs** to the inline ND→NZ route and **65 µs** to a plain
2D GM→L1 copy. With one AIC, the same pitch is slightly faster on both
routes. Both request exactly 315 MiB from GM→L1, and neither loses L2
hits at the center. Thus inline conversion is **not necessary** for
the multicore pitch slowdown, although the route changes its size.
The probe does not distinguish GM address mapping, interconnect
contention, or MTE2 arbitration, and its A-only penalty must not be
used directly as a full MatMulV3 correction.
The estimator's generic GM-read coefficient was itself measured with an
inline ND-to-NZ copy, but with FP16, a contiguous 128-element source
pitch, 4–32 KiB packets, and L2-hot data. It does not represent the
FP32 128-KiB-pitch A stream here. The model can therefore miss a
pitch- and concurrency-dependent transfer service penalty even though it
includes a generic ND-to-NZ read rate.

The [extended probe](../fp32_mode2_copy_route_v1/README.md) swept
1–20 AICs, then interleaved source-shaped A and B loads at 8/12/20
AICs. With 20 AICs, the A/B input service adds **545–551 µs** at the
128-KiB A pitch, compared with the full kernel's **582–593 µs MTE2
busy-time excess** and **574–578 µs runtime excess**. The A/B probe's
center service is 2084/2496 µs for N=768/1024, close to the full
kernel's 2081/2476 µs MTE2 busy time. Both inline-ND→NZ and plain-A
routes show the same large penalty once B traffic is present. The
probe's logical K remains fixed at 32256, so this is a matched input
service check, not a fit to the full kernel's exact byte volume.

The A/B profiler reports exactly its predicted $4K(M+20N)$ requested
bytes and an unchanged aggregate L2 hit fraction across 127/128-KiB
pitch. Full MatMulV3 profiles at 128×N, 160×1024, and 208×1024 cores
all support one A plus one B load per inner-M panel and K slice, within
a fixed 320 KiB. The present `ScMode2PipelineSliceUs` input-byte
formula instead recharges A for each N base piece and B for each M
base piece. That traffic count and its use of an aggregate read rate
inside a per-core slice need a source-shaped, occupancy-aware input
service. The measured A/B rate has not yet been generalized over
held-out pitches and geometries, so no online scorer change follows
from this probe.

The subsequent [192-KiB holdout](../fp32_mode2_transfer_holdout_v1/README.md)
tested five source-emitted geometries, including a three-inner-M-panel
tile. The generalized serialized A/B probe overpredicted the full
kernel's center-pitch penalty in every case. On the three-panel tile,
probe time increased 4230 µs, kernel time increased 707 µs, and the
reported MTE2 busy counter decreased 3024 µs. Source and probe
requested the same GM→L1 bytes after K scaling, within the recurring
320-KiB overhead, and aggregate L2 hit fraction stayed nearly fixed
across 191/192-KiB pitch. This rules out using the serialized probe
time as a general $R_{AB}$ measurement. The traffic-count correction
remains supported; exposed service and overlap require a
dependency-aware timeline.

The [copy-synchronization follow-up](../fp32_mode2_sync_holdout_v1/README.md)
tested less serialized probes. Barriers after every copy, every K
packet, or every inner-M panel produced nearly identical 192-KiB
center penalties at all five geometries, although baseline probe
time fell when barriers were relaxed. Representative counter profiles
show identical requested GM→L1 bytes and aggregate L2 hit fractions
across the three modes. Copy-issue throttling alone therefore does
not explain the source/probe timing gap; a useful runtime surrogate
must include operand consumption, buffer dependencies, and overlap
with Cube and Fixpipe.

The source's `SetL2CacheFlagSingleCoreSplitK` disables A caching when
`singleCoreN == N` and enables B caching when `singleCoreM < M`. The
audited K=32768 tiling log gives `l2CacheFlag: 2`, consistent with the
A-disable bit. At K=32768, A alone is 320 MiB and B is 96/128 MiB;
these capacities do not cross a threshold between adjacent K values.
L2 capacity can influence the baseline traffic, but capacity alone
does not explain the isolated center spike. The multicore plain-copy
result points to a shared transfer/address effect; the present counters
do not distinguish GM partition conflicts, interconnect contention,
or MTE2 request scheduling. No product estimator change was made.
