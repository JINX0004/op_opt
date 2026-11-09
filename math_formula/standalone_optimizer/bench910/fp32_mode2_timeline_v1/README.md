# FP32 mode-2 packet event timeline diagnostic

This directory tests an explicit event schedule for the FP32 MatMulV3
single-core Split-K mode-2 path on 910B3. It uses **no full-kernel runtime
to fit a correction**. It does not change the online selector.

## Inputs and source correspondence

The four accepted fixed-work shapes and their emitted tiles come from the
[in-context pitch experiment](../fp32_mode2_in_context_pitch_v1/README.md).
Each has logical K=48384, 189 packets of K=256, 20 occupied AICs,
FP32 NN input/output, no ND→NZ conversion, and one inner-M panel per
AIC. Only A's physical row pitch changes: 48896, 49152, or 49408 FP32
elements. The measured output is correct at all three pitches.

The [source kernel](../../../../mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel.h)
iterates `innerMIndex`, `kIndex`, then `innerNIndex` for its MKN path;
`SetParamAndExec` calls `MatmulImpl::IterateAll` for each K packet.
The tested source tiles set `stepM=2`, `stepN=1`, `baseM=baseN=128`,
`baseK=64`, so their inner-M panel is the entire per-core M extent.
The internal order of base pieces within `MatmulImpl` is not exposed
by this source. The evaluator therefore reports two order hypotheses:
M-block then N-block (`mn`) and the reverse (`nm`).

The event input service is the median of two independent
[packet-barrier copy-probe](../fp32_mode2_sync_holdout_v1/README.md)
rounds at the same pitch and geometry (`sync_mode=1`). It is divided
by 189 to obtain a per-packet duration. The copy probe itself is an
aggregate copy schedule, **not** a measured in-kernel MTE2 primitive;
its time includes its own issue and synchronization costs. Pitch enters
this model only through that measured probe duration. The probe gives
one duration for A and B together, so the evaluator allocates it among
the A panel and B base-column copies by requested bytes. This is an
explicit approximation, not a measurement of A/B service separately.
The historical probe disables A caching for three of these shapes,
whereas the source leaves A and B at normal caching; a subsequent
[matched-hint control](../fp32_mode2_cache_policy_v1/README.md) found
that this changes the probe's center-pitch excess by at most 17 µs.

## Event formulation

For one packet with core extent `(m,n,k)`, let
`q_m=ceil(m/baseM)`, `q_n=ceil(n/baseN)`, and
`q_k=ceil(k/baseK)`. For each base piece `(i,j,h)`:

| Event | Duration, µs | Dependency |
| --- | --- | --- |
| MTE2 A panel | `probe_packet × m/(m+n)` | previous MTE2 copy |
| MTE2 B column piece `j` | `probe_packet × n_j/(m+n)` | previous MTE2 copy |
| L1→L0A | `4 m_i k_h / 460800` | A ready, L0A reuse after two Cube pieces |
| L1→L0B | `4 n_j k_h / 230400` | B piece ready, L0B reuse after two Cube pieces |
| Cube | `2 m_i n_j k_h / 3600000` | both L0 loads, preceding Cube, L0C reuse after two output pieces |
| First-packet Fixpipe | `0.2432 + 4 m_i n_j / 34190` | final Cube for output piece |
| Later-packet atomic Fixpipe | `4 m_i n_j / 91000` | final Cube for output piece |

Every stage has a separate exclusive engine. For event `v`,

`start(v) = max(available(engine(v)), finish(predecessors(v)))`,
`finish(v) = start(v) + duration(v)`.

The packet makespan is the maximum finish time. The evaluator adds the
189 packet makespans in source K order plus a 2.68-µs launch term. The
packet boundary is a conservative `IterateAll` return dependency;
overlap *within* each packet is scheduled explicitly. The graph keeps
two L0A/L0B slots and two L0C output slots. It does not enforce the
source tile's L1 `depthA1`/`depthB1` occupancy, so B copies can be
issued ahead of operand consumption; this matters especially for the
1536-column case with 12 base-N pieces. Its critical-path stage
durations sum to predicted runtime minus launch, which the analysis
script checks on every prediction.

The Cube rate, launch time, and FP32 plain/atomic Fixpipe rates come
from the [experimental FP32 hardware profile](../fp32_mode2_v1/remote_selector/model/hardware_profile.h).
The L1→L0A/B rates are Huawei nominal ceilings used in the opt-in
reduced pipeline; they are **not** measured mode-2 services. In
particular, the atomic Fixpipe rate originated from a mode-6 proxy.

## Reproduce

From the repository root:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  math_formula/standalone_optimizer/bench910/fp32_mode2_timeline_v1/timeline.cpp \
  -o /tmp/fp32_mode2_timeline
python3 math_formula/standalone_optimizer/bench910/fp32_mode2_timeline_v1/analyze.py \
  /tmp/fp32_mode2_timeline
```

The C++ program accepts whitespace-separated rows:

`id pitch coreM coreN K packetK baseM baseN baseK probe_us order`

It emits per-row runtime, first/steady packet durations, critical-path
stage durations, stage busy durations, and event count. The driver
writes [per-pitch predictions](comparison.csv) and [summaries](analysis.json).
The driver passes only the tile and independent probe time into C++;
it loads full-kernel times for comparison *after* prediction.

## Held-out comparison

The following uses `mn` base-piece order. Times are µs. Center excess
is 192-KiB runtime minus the mean of the 191- and 193-KiB runtimes.

| M×N | Measured 191/192/193 KiB | Predicted 191/192/193 KiB | Measured excess | Predicted excess | Copy-probe excess |
| --- | --- | --- | ---: | ---: | ---: |
| 2816×1024 | 5233 / 5678 / 5290 | 5078 / 5306 / 5074 | 417 | 230 | 987 |
| 3072×1024 | 5438 / 6063 / 5493 | 5122 / 5403 / 5121 | 597 | 282 | 1159 |
| 2048×2048 | 6241 / 6351 / 6306 | 6545 / 6737 / 6545 | 77 | 192 | 706 |
| 2816×1536 | 7568 / 8021 / 7645 | 7245 / 7401 / 7245 | 415 | 157 | 967 |

The mean absolute error of the four center excesses is **219 µs** for
the event schedule versus **578 µs** for the copy probe used directly.
The schedule's 12 absolute runtimes have **5.7% mean absolute percentage
error** with `mn`. With `nm`, absolute error is **3.1%** while every
center-excess prediction stays the same; choosing `nm` from these
measurements would be fitting an unverified internal order.

The event graph therefore captures substantial overlap but **fails the
pitch-response check**: it underpredicts the excess for three shapes
and overpredicts it for 2048×2048. The result is not sufficient to
replace the online mode-2 score or select a family. The missing
information is the in-context A/B service split, `MatmulImpl`'s actual
base-piece and L1 buffer-wait order, and mode-2-specific output and setup
services. Earlier source profiles report large overlapping scalar and
MTE1 busy time; these counters cannot simply be added to the graph's
critical path. The subsequent [A/B producer-consumer probe](../fp32_mode2_consumer_probe_v1/README.md)
finds a large nonadditive A+B pitch interaction that survives a
controlled L1→L0 handoff. In-context source profiling confirms a much
smaller, geometry-dependent MTE2 busy excess. The next required
observation is the actual `MatmulImpl` issue and wait order or an
in-context primitive that includes Cube and output.
