# FP32 mode-2 A-load copy-route experiment on 910B3

## Question

Does the K=32768 slowdown require the inline ND→NZ conversion instruction,
or does it also occur on a plain GM→L1 copy with the same A access pattern?

## Controlled probe

[`copy_route_kernel.cpp`](copy_route_kernel.cpp) issues 128×256 FP32 A
copies from GM to a 128-KiB L1 buffer. Route `inline` uses `Nd2NzParams`;
route `plain` uses one 128-row, 256-element-per-row 2D `DataCopyParams`
copy. Both routes read the same 315-MiB logical A data per launch, in
the same row/panel/K-packet order, and both set the A L2 hint to
`CACHE_MODE_DISABLE`, as in the exact mode-2 tile. The plain route
does not rearrange data; this read-only probe does not compute Matmul.
The probe was built in Release mode against the machine's CANN 9.0.0;
[`binary_sha256.txt`](binary_sha256.txt) fingerprints the executable.

Logical K is held at 32256. Physical row pitch is varied among 32512,
32768, and 33024 FP32 elements (127, 128, and 129 KiB). Thus byte
count and packet count stay fixed when pitch changes. With 20 AICs each
gets one disjoint 128-row panel. With one AIC it traverses all 20
panels. The [driver](run.py) ran both routes on devices 4 and 6, two
rounds with reversed pitch and route order, three warmups and 12 timed
kernel launches per case. Each launch writes and checks one token per
AIC. The [raw timings](records.json) and [analysis](summary.json) are
preserved.

| AICs | Route | 127-KiB pitch µs | 128-KiB pitch µs | 129-KiB pitch µs | Center excess µs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20 | Inline ND→NZ | 259.30 | **297.15** | 260.10 | **37.86** |
| 20 | Plain 2D copy | 233.34 | **298.60** | 231.99 | **65.33** |
| 1 | Inline ND→NZ | 4006.80 | 3753.77 | 4004.11 | −251.24 |
| 1 | Plain 2D copy | 3003.30 | 2966.38 | 3000.85 | −36.93 |

Center excess subtracts the mean of the two neighboring pitches within
each device/round. Its four-sample ranges are 34.41–42.66 µs (20-AIC
inline), 63.28–72.20 µs (20-AIC plain), −279.85 to −244.34 µs
(one-AIC inline), and −38.47 to −14.83 µs (one-AIC plain). These
are replicate ranges, not confidence intervals.

The separate [`msprof` checks](profile_metrics.json), collected by
[`profile.py`](profile.py) on device 4, report exactly 322560 KiB of
GM→L1 requested data in all four profiled route/pitch combinations.
The MTE2 busy fraction is 99.4–99.5%. L2 read-hit fraction is about
0.003% at 127 KiB and rises to 0.75%/1.16% at 128 KiB for inline/plain;
the slow pitch does **not** have fewer aggregate L2 hits.

## Interpretation for the formulation

Inline conversion is **not necessary** for the multicore 128-KiB
slowdown: the plain copy has a larger center penalty. A single AIC
does not show a slowdown at that pitch. The evidence points to a
concurrency-dependent A transfer/address service effect, with the
copy route affecting its magnitude. The counters do not distinguish
GM partition mapping, interconnect arbitration, or MTE2 issue behavior.

This A-only probe cannot directly supply a mode-2 kernel correction:
MatMulV3 interleaves A and B loads, Cube work, and Fixpipe; its
measured center excess is 575–578 µs. The extended A/B experiment
below tests that input-service context. The subsequent timeline change
still needs to address the residual 220–240-µs baseline gap.

## A-only occupancy sweep

The [sweep driver](run_core_sweep.py) repeated the A-only probe at
2, 4, 8, 12, and 16 AICs on devices 4 and 6. Each device used the
opposite core/pitch/route order. The 1- and 20-AIC rows above had two
rounds per device; these intermediate rows had one. All use the same
20 spatial panels, so a core can own more than one panel when occupancy
is below 20. Values below are the median timing across devices at each
pitch; center excess is center minus the mean of those two neighbor
medians. See [raw records](core_sweep_records.json) and the
[derived summary](extended_summary.json).

| AICs | Inline center excess µs | Plain center excess µs |
| ---: | ---: | ---: |
| 1 | −251.7 | −35.7 |
| 2 | −125.4 | +11.5 |
| 4 | −40.4 | +74.1 |
| 8 | −0.7 | +66.5 |
| 12 | +23.1 | +80.2 |
| 16 | +20.9 | +60.7 |
| 20 | +37.5 | +65.9 |

The sign and size depend on both route and concurrency. These are
descriptive measurements, not a monotone fitted law; unequal numbers
of panels per active core also change the measured makespan.

## Source-shaped A/B input-service probe

The [A/B kernel](ab_route_kernel.cpp) issues one 128×256 A load,
followed by six or eight 256×128 B loads, per K packet. It repeats
this for each assigned spatial panel. A has the same disabled L2 hint
as the exact mode-2 case; B uses its normal hint. B always uses inline
ND→NZ; A is either inline ND→NZ or plain 2D. Each copy is followed by
an MTE2 barrier, so this probe measures serialized **input service**
under the source-shaped working set, not the full Matmul timeline. It
uses logical K=32256 and varies only A's physical pitch among
32512/32768/33024 FP32 elements. Devices 4 and 6 ran N=768/1024,
respectively, with 20/12/8 AICs, two reversed-order rounds, three
warmups and five timed launches per case. Output tokens were checked
after device synchronization. The [runner](run_ab.py),
[raw records](ab_records.json), and [binary fingerprint](ab_binary_sha256.txt)
preserve the experiment.

| N | AICs | A route | 127-KiB µs | 128-KiB µs | 129-KiB µs | Center excess µs |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 768 | 8 | Inline | 3804.1 | 4092.6 | 3857.8 | +261.7 |
| 768 | 8 | Plain | 3649.2 | 4089.3 | 3711.6 | +408.9 |
| 768 | 12 | Inline | 2751.9 | 3155.1 | 2764.4 | +397.0 |
| 768 | 12 | Plain | 2665.6 | 3175.6 | 2672.1 | +506.8 |
| 768 | 20 | Inline | 1531.1 | 2084.5 | 1535.5 | **+551.2** |
| 768 | 20 | Plain | 1526.3 | 2076.2 | 1529.1 | **+548.5** |
| 1024 | 8 | Inline | 4815.7 | 5149.8 | 4869.2 | +307.4 |
| 1024 | 8 | Plain | 4666.5 | 5144.1 | 4726.5 | +447.6 |
| 1024 | 12 | Inline | 3433.6 | 3849.8 | 3437.9 | +414.0 |
| 1024 | 12 | Plain | 3337.6 | 3859.4 | 3342.1 | +519.6 |
| 1024 | 20 | Inline | 1949.8 | 2495.5 | 1951.5 | **+544.9** |
| 1024 | 20 | Plain | 1949.1 | 2495.4 | 1946.6 | **+547.6** |

At 20 AICs, the corresponding full mode-2 kernel's center excess is
574.5/578.4 µs for N=768/1024. Its MTE2 busy time at the center is
2081.3/2475.8 µs, versus 2084.5/2495.5 µs in the inline A/B probe.
The probe's center excess is within 5–8% of the full kernel's MTE2
center excess (582.1/593.2 µs). It is a useful matched service
measurement; equality of MTE2 busy and elapsed probe time at this
geometry does not imply identical schedules for other tiles. The probe
holds logical K at 32256, while the full kernel uses actual K values
32512/32768/33024, so its input byte volume is 0.8–2.3% smaller.

The [A/B profiler](profile_ab.py) checked four 20-AIC cases. Its
[metrics](ab_profile_metrics.json) report exactly
2,257,920/2,903,040 KiB of GM→L1 requested data for N=768/1024,
equal to $4K_{\rm logical}(M+20N)/1024$. The count is identical at
127- and 128-KiB pitch. Aggregate L2 read hit fraction is likewise
unchanged: 81.429% for N=768 and 84.444% for N=1024. MTE2 is
99.74–99.80% busy. The large center penalty therefore occurs at fixed
logical bytes and essentially fixed aggregate L2 hits. The plain-A
route has almost the same center penalty as inline A in this A/B
context, so conversion alone cannot explain the full-kernel effect.

## Full-kernel traffic check at other geometries

The earlier full-kernel profiles at 128×N showed one A and one B GM→L1
read per inner-M panel and K slice, within a fixed 320 KiB of measured
traffic. The [source geometry profiler](profile_source_geometry.py)
checked two more FP32 mode-2 tilings in the source-built MatMulV3 on
devices 5 and 7:

| Shape M×N×K | Core M×N×K | Measured GM→L1 KiB | A/B once per panel KiB | Difference KiB |
| --- | --- | ---: | ---: | ---: |
| 3072×1024×32768 | 160×1024×256 | 3,014,976 | 3,014,656 | 320 |
| 2048×2048×32768 | 208×1024×256 | 3,146,048 | 3,145,728 | 320 |

The [audit and counters](source_geometry_metrics.json) support the
same traffic rule beyond 128-row cores. For these shapes, the
predicted input bytes are
$4K(M\lceil N/C_N\rceil+N\lceil M/C_M\rceil)$: A is repeated across
distinct N core tiles and B across distinct M core tiles. Within one
inner-M panel, the current mode-2 formula instead charges A once per
128-column base piece and B once per 128-row base piece. This is an
overcount for the tested `MM_CFG_PRELOAD_MK` source path. Tail tiles,
different preload configurations, and other data types still need
their own traffic check.

## Model implication

Use a measured A/B service with physical pitch, route, occupied AICs,
cache hints, working set, and source packet order as inputs. The A-only
rate cannot be added independently to a B rate: adding B changes the
A-pitch penalty from roughly 38–65 µs to roughly 545–551 µs at 20 AICs.
The corrected traffic count and matched A/B service are separate
inputs to the future timeline recurrence. Neither is a hard-coded
K=32768 penalty. More held-out pitches and core M/N geometries are
needed before switching the FP32 online family's score to this model.
