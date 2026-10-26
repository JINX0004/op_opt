# Huawei memory-path diagram versus the 910B3 primitive profile

The [user-supplied diagram](huawei-hardware-infr.png) shows 20 AICs at
1.8 GHz, 192 MB of L2, nominal 512/64/64/128 KB L1/L0A/L0B/L0C, and
several labeled bytes-per-cycle rates. The diagram is a useful sanity
check, but it does not define the packet shape, cache residency, direction
of every link, or whether every number is a hard peak. The calculations
below interpret the path labels as **per AIC**, multiplied by 20 AICs:

$$R_{20}=r_{\rm B/cycle/AIC}\,(1800\ {\rm cycle/\mu s})\,(20\ {\rm AIC}).$$

Decimal MB/µs below means one million bytes per microsecond, numerically
the same as TB/s. The measurements are effective *requested-byte* rates,
not observed physical-link traffic.

| Diagram path | Diagram rate | 20-AIC equivalent | Relevant probe | Assessment |
| --- | ---: | ---: | ---: | --- |
| GM read/write | 32 B/cycle/AIC | 1.152 MB/µs | Above-L2 ND→NZ read: 1.063 MB/µs | Close (92%); appropriate comparison for a working set larger than the 192 MiB L2. |
| L2 read | 110 B/cycle/AIC | 3.960 MB/µs | L2-hot generic ND→NZ read: 3.710 MB/µs | Close (94%). This is the regime of the current illustrative `gm_read` value. |
| L2 read | 110 B/cycle/AIC | 3.960 MB/µs | L2-hot B-shaped read: 3.323 MB/µs | Below the diagram value (84%). |
| GM read/write | 32 B/cycle/AIC | 1.152 MB/µs | L2-hot A-shaped read: 1.117 MB/µs | Similar numerically (97%), but the probe's 16 MiB source is cache-fitting; this does **not** establish that A traffic came from HBM. |
| L0C/Fixpipe path | 128 B/cycle/AIC | 4.608 MB/µs | FP16 Fixpipe output fit: 5.230 MB/µs | The fit is 14% higher; these are not safely interchangeable without resolving the diagram's link definition and the probe's requested-byte accounting. |
| Cube, if `8192` denotes FP16 FLOP/cycle | 8192 FLOP/cycle/AIC | 14.746 million FLOP/µs/AIC | Repeated Cube: 14.719 million FLOP/µs/AIC | Excellent numerical agreement (99.8%); the diagram actually prints `B/cycle`, so the FLOP interpretation is an inference. |

The screenshot additionally labels **MTE2 43.7 B/cycle**, **L2 write
86 B/cycle**, **MTE3 64 B/cycle**, and **L1→L0A/L0B 256/128 B/cycle**.
If interpreted as hard per-AIC ceilings, the 43.7 B/cycle MTE2 label
would yield only 1.573 MB/µs for 20 AICs, below both our L2-hot generic
and B-shaped requested-byte rates. This indicates a difference in path,
workload, byte convention, or meaning of the red 43.7 annotation; the
image alone cannot resolve it. Similarly, the 86 B/cycle L2-write and
64 B/cycle MTE3 labels are below the fitted Fixpipe requested-byte rate.
The fitted Fixpipe rate should therefore not be called a physical GM or
L2 write bandwidth.

The memory capacities align closely with the [CANN platform query](hardware_910b3.json):
20 AICs, 192 MiB L2, 64 KiB L0A/L0B, and 128 KiB L0C. Queried L1 and UB
are each 256 bytes below the diagram's nominal 512/192 KiB values, so
the solver correctly uses queried capacities for feasibility.
The diagram's 32 B/cycle/AIC calculation gives 1.152 TB/s, while the
[general 910B3 reference profile](../../README.md#23-ascend-910b3-hardware-profile)
records a 1.6 TB/s nominal HBM figure from a different source. The
1.063 TB/s above-L2 probe is closer to the diagram's number. Treat the
two published figures as different path/peak definitions until their
test conditions are reconciled.

**Model consequence.** The generic `gm_read=3.710212e6 bytes/µs` in the
[exploratory profile](PRIMITIVE_910B3_CANN9.md) is consistent with an
L2-hot read regime, but it should not be used as an above-L2 HBM rate.
The separate A/B fits reflect packet geometry and cache behavior; they
are not intrinsic A1/B1 link specifications. The current
`gm_write=5.229502e6 bytes/µs` is a Fixpipe probe slope, not a validated
GM write primitive. The diagram also highlights distinct L1→L0A and
L1→L0B rates, which the reduced `SolveBase` objective does not yet model.
Do not replace the profile with raw diagram peaks: first identify the
actual transfer path and memory residency of each modeled event, then
validate against whole-kernel timings.

## L1→L0A and L1→L0B asymmetry

The diagram labels the A-side MTE1 path **256 B/cycle/AIC** and the B-side
path **128 B/cycle/AIC**. At 1.8 GHz these correspond to nominal per-AIC
service rates of 460,800 and 230,400 bytes/µs respectively. They are
distinct from the GM/L2→L1 MTE2 transfers that our `gm_read_a` and
`gm_read_b` probes measure. The current reduced `SolveBase` objective has
no L1→L0 service term.

For the [checked 2048×2048×4096 BASE comparison](FULL_BASE_OPTIMIZER_VS_DEFAULT_910B3.md),
both plans use `baseK=64`, 20 AICs, and 128 output tiles. If each
base-K MMAD consumes one A and one B L1→L0 tile, then one packet's
nominal service demand is:

| `(baseM,baseN)` | A bytes / A MTE1 time | B bytes / B MTE1 time | Serial sum | Parallel maximum |
| --- | ---: | ---: | ---: | ---: |
| Source 128×256 | 16 KiB / 0.0356 µs | 32 KiB / 0.1422 µs | 0.1778 µs | 0.1422 µs |
| Swapped 256×128 | 32 KiB / 0.0711 µs | 16 KiB / 0.0711 µs | 0.1422 µs | 0.0711 µs |

Across 128 output tiles and 64 K packets, the corresponding whole-device
service lower bounds are 72.82 versus 58.25 µs if A/B MTE1 services
serialize, or 58.25 versus 29.13 µs if their paths run independently and
the slower one sets the stage. These are *nominal stage-service amounts*,
not added kernel latencies. They favor the **swapped 256×128** tile by
14.56–29.13 µs, opposite the measured result: the source 128×256 tile
was about 21.4 µs faster. Thus MTE1 asymmetry cannot alone explain the
source preference. It may offset part of the GM→L1 asymmetry, whose
isolated-read model favored the source tile by 79.7 µs; simply adding the
two stage differences would still overpredict the observed gap.
Putting these MTE1 amounts under one extra `max` in the generic reduced
objective would not change this pair's ranking: its existing memory branch
already exceeds the nominal MTE1 service bounds. Their timing effect can
only be assessed with per-wave dependencies and overlap, not an isolated
global throughput ceiling. The earlier 79.7 µs GM→L1 directional gap came
from isolated `baseK=64` transfers and did not reproduce the source's
larger L1-step packets.

The [emitted tiling records](source_compatible_base_repeat_raw.jsonl)
also change L1 buffering: source 128×256 uses `stepKa=8`, `stepKb=4`,
`depthA1=16`, `depthB1=8`; swapped 256×128 uses `stepKa=4`, `stepKb=8`,
`depthA1=8`, `depthB1=16`. Both have `stepM=stepN=1`. Those choices can
change when A/B packets arrive and how MTE1 overlaps Cube. The BASE kernel
passes that tiling to the Ascend C Matmul engine in
[`mm_.Init`](../../../mat_mul_v3/op_kernel/mat_mul_base_kernel.h), so its
internal MTE1 issue order is not explicit in this repository. A subsequent
[source-step MTE2 probe](l1_step_read/README.md) measured the actual A/B
GM/L2→L1 packet geometries. Their busiest-core serial service is about
133.39 versus 149.64 µs for the two orientations, which gives the observed
preference's sign without a whole-kernel fitted correction. The optional
source-step objective ranks the tested orientation pair correctly on seven
whole-kernel shapes, but its compact law misses one held-out primitive
packet shape by 18.3%, so it remains opt-in.
A later [source-backed selector test](PACKET_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
found one tail-heavy shape where that objective chose a tile 15.94% slower
than the default; the source's L2/core assignment needs to be modeled.
Before adding a nominal MTE1 term, measure A1→L0A and B1→L0B load service
with matched layouts and test an interleaved A/B/Cube timeline on held-out
kernel shapes.
