# FP32 mode-2 input-service holdout at 192 KiB on 910B3

## Purpose and setup

This is the held-out check proposed after the
[128-KiB A/B probe](../fp32_mode2_copy_route_v1/README.md). It tests
whether a serialized, source-shaped A/B copy probe predicts the
full FP32 mode-2 kernel's input-service penalty when A's physical row
pitch is 192 KiB, and whether the relation survives different core
M/N geometries and multiple inner-M panels.

The [audit driver](audit_source_tiles.py) forced the single-core
Split-K family in the source-built MatMulV3 package on 910B3 devices
4–7 with CANN 9.0.0. It checked output correctness and the emitted
tile at K=48896/49152/49408 FP32 elements, corresponding to
191/192/193-KiB A row pitch. All 15 audits passed and each geometry's
tile stayed fixed across the three K values. The
[audit records](tile_audits.json) include the exact emitted fields.

The [generalized A/B probe](panel_ab_kernel.cpp) follows the source's
MKN loop shape: inner-M panel, K packet, then 128-column B pieces. It
uses the audited core M/N, `stepM`, and single-core K. The probe's
historical cache-hint heuristic disables A caching when
`singleCoreN == N`; this differs from the device source, which only
disables A for the NKM path when `stepN * baseN == N`, and disables B
for the MKN path when `stepM * baseM == M`. Neither source condition
holds for the four fixed-work geometries later studied in the
[matched-hint follow-up](../fp32_mode2_cache_policy_v1/README.md).
Each A and B copy has an MTE2 barrier.
It therefore measures a **serialized input-copy schedule**, not
MatmulV3's hidden overlap or full execution timeline. Probe logical K
is fixed at 48384 (divisible by both 192 and 256); only A's physical
pitch changes. The source kernels use their actual K values, which
are 1.1–2.1% larger than the probe's logical K. Probe and source use
the same requested-byte pattern after scaling for that K difference.
For N=1024, this also makes the probe's B tensor 189 MiB while the
source's B tensor reaches 192 MiB at K=49152, so their cache working
sets are not identical at that threshold.

The [probe driver](run_panel_probe.py) ran two opposite pitch orders,
three warmups and five timed launches for each of the 15 settings.
The [source profiler](profile_source_tiles.py) ran two opposite pitch
orders, five warmups and five profiled MatMulV3 tasks per setting.
The [binary fingerprint](binary_sha256.txt), [probe records](probe_records.json),
[source profiles](source_profiles.json), and [analysis](summary.json)
preserve the measurements. One timed probe launch failed its output
token check and was discarded; rerunning that exact case passed, and
the accepted record includes only checked launches.

## Result: the serialized probe does not generalize

Center excess is the 192-KiB timing minus the mean of the 191/193-KiB
neighbors. Each value below is computed from the median of two rounds
at each pitch. Stage busy counters can overlap and are not elapsed
kernel stages.

| M×N | Core M×N×K | Inner-M panels | Probe excess µs | Full MTE2 busy excess µs | Full kernel excess µs |
| --- | --- | ---: | ---: | ---: | ---: |
| 2048×2048 | 208×1024×256 | 1 | **+698** | +48 | +69 |
| 2816×1024 | 144×1024×256 | 1 | **+996** | +409 | +439 |
| 2816×1536 | 144×1536×256 | 1 | **+969** | +287 | +442 |
| 3072×1024 | 160×1024×256 | 1 | **+1116** | +477 | +585 |
| 8192×4096 | 824×2048×192 | 3 | **+4230** | **−3024** | +707 |

The old 128×N test matched the full MTE2 penalty within roughly 5–8%.
At 192 KiB, the new probe overstates the kernel's runtime excess in
all five geometries. The three-panel case is especially instructive:
the serialized probe slows by 4.23 ms, while the full kernel slows by
0.71 ms and its reported MTE2 busy time *decreases* by 3.02 ms. Its
MAC busy time stays essentially smooth across the K neighbors. A stage
busy counter is therefore not a reliable elapsed input-service time
for this geometry; stalls and overlap can change the kernel timeline
without a matching increase in MTE2 busy cycles. The exact hidden
dependency responsible is not identified by these counters.
The two individual probe rounds give a 4.226–4.234-ms center excess
for the three-panel case; the full-kernel rounds give 0.682–0.732 ms,
and both MTE2 rounds have a negative excess. For the other geometries,
both rounds also preserve the probe-versus-kernel ordering. These
ranges are repeat observations, not confidence intervals.

## Traffic and L2 checks

The [counter profiler](profile_traffic.py) checked the 191/192-KiB
pair for three representative geometries. In each, the source's
requested GM→L1 bytes equal the probe's requested bytes scaled by
`source K / 48384`, plus the same fixed 320 KiB seen in the previous
full-kernel profiles:

| M×N | Source − scaled probe at 191 KiB | At 192 KiB | Source L2 read-hit fraction, 191→192 KiB | Probe fraction |
| --- | ---: | ---: | ---: | ---: |
| 3072×1024 | 320 KiB | 320 KiB | 79.75→79.77% | 82.61→82.61% |
| 2048×2048 | 320 KiB | 320 KiB | 83.33→83.33% | 83.33→83.33% |
| 8192×4096 | 320 KiB | 320 KiB | 86.27→86.27% | 86.27→86.27% |

See the raw [MemoryAccess](traffic_MemoryAccess.json) and
[L2Cache](traffic_L2Cache.json) counters. Requested-byte mismatch does
not explain the probe's timing error. Aggregate L2 hit fraction is
stable across pitch within each arm, even for the multi-panel case;
it cannot by itself explain the 192-KiB timing reversal. The source
and probe have a 2.9-point hit-fraction difference for 3072×1024,
which may contribute to that case's absolute service gap. The
189-versus-192-MiB B footprint is one possible reason. The counters
do not separate A from B latency or reveal all queue stalls.

## Consequence for the formulation

The source-shaped **traffic count** remains supported, including a
multi-panel tile. The serialized A/B probe is **not** a general
estimator for the exposed GM→L1 path or the kernel makespan. In
particular, do not fit an effective rate from its 192-KiB center
penalties and insert it into the online selector. A next experiment
should preserve the measured byte stream but remove the per-copy
barriers, then compare packet-level and panel-level synchronization
against the full kernel. The mathematical model also needs explicit
overlap and buffer/wait dependencies before a measured input rate can
drive family selection. No product scorer or family gate was changed
by this experiment.
