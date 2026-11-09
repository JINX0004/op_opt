# FP32 mode-2 copy synchronization holdout on 910B3

## Question

The [192-KiB source-versus-probe holdout](../fp32_mode2_transfer_holdout_v1/README.md)
showed that a serialized A/B copy probe overstates the full MatMulV3
kernel's 192-KiB A-row-pitch penalty. Does relaxing copy barriers to
K-packet or inner-M-panel boundaries explain that mismatch?

## Controlled experiment

The [kernel](panel_ab_kernel.cpp) follows the same audited MKN input
packet pattern as the earlier probe and has three runtime modes:

| Mode | Synchronization |
| ---: | --- |
| 0 | MTE2 barrier after every A or B copy |
| 1 | MTE2 barrier after each K packet's A and B copies |
| 2 | MTE2 barrier after each inner-M panel |

All modes use the same GM addresses, copy parameters, L1 destination
buffers, A/B cache hints, AIC count, and logical K=48384. The
destinations are never consumed; successive MTE2 writes reuse them
in issue order. The probe tests copy-issue throttling, **not** the
Matmul engine's operand-consumption or buffer dependency graph.
The [C++ runner](panel_ab_runner.cpp) checks an output token from
each AIC after device synchronization.

The [sweep driver](run.py) used the 15 previously audited source tiles
at 191/192/193-KiB A pitch. Devices 4–7 ran two rounds with reversed
pitch and mode order, three warmups and five timed launches per case.
All **90 accepted probe measurements** passed token checks. The
[binary fingerprint](binary_sha256.txt), [raw timings](probe_records.json),
and [analysis](summary.json) are preserved. The full-kernel comparison
uses the two-round forced MatMulV3 profiles from the
[prior experiment](../fp32_mode2_transfer_holdout_v1/source_profiles.json),
so its tile and output audits are unchanged.

## Center-pitch result

Center excess is time at 192 KiB minus the mean of the 191/193-KiB
neighbors. Each table entry uses the median of the two timing rounds
at each pitch. Stage busy counters can overlap and need not track
elapsed kernel time.

| M×N | Core M×N×K | Full kernel excess µs | Copy-barrier probe µs | Packet-barrier probe µs | Panel-barrier probe µs |
| --- | --- | ---: | ---: | ---: | ---: |
| 2048×2048 | 208×1024×256 | **69** | 698 | 706 | 712 |
| 2816×1024 | 144×1024×256 | **439** | 988 | 987 | 989 |
| 2816×1536 | 144×1536×256 | **442** | 974 | 967 | 961 |
| 3072×1024 | 160×1024×256 | **585** | 1125 | 1159 | 1134 |
| 8192×4096 | 824×2048×192, three inner-M panels | **707** | 4217 | 4223 | 4200 |

Moving barriers to panel boundaries reduces the neighbor-pitch
probe's **absolute** time by 198–382 µs for the one-panel cases and
2149 µs for the three-panel case. Yet the 192-KiB center excess
changes by at most **34 µs** among the modes for any geometry, a size
comparable with the round-to-round spread. The mean absolute
difference between probe and full-kernel center excess across the
five geometries is 1152, 1160, and 1151 µs for modes 0, 1, and 2.
Thus neither relaxed barrier schedule improves the held-out runtime
prediction. The full kernel's reported MTE2 busy excess ranges from
+48 to +477 µs on the one-panel tiles and is **−3024 µs** on the
three-panel tile; it is not a direct elapsed copy-service target.

## Traffic and cache check

The [counter driver](profile_modes.py) profiled all three modes at
192-KiB pitch for a one-panel tile with A caching disabled and a
three-panel tile with A caching enabled. All modes request the same
GM→L1 bytes and have the same aggregate L2 read-hit fraction:

| M×N | Each mode GM→L1 requested KiB | Each mode L2 read-hit fraction |
| --- | ---: | ---: |
| 2816×1024 | 4,402,944 | 83.517% |
| 8192×4096 | 26,320,896 | 86.275% |

The [raw counter records](mode_profile_metrics.json) support the
controlled comparison. This result does not identify the physical
cause of the pitch penalty. It shows that, for this copy-only probe,
the penalty persists when outstanding copy issue is allowed across
an entire panel. The earlier source/probe byte checks still match
after K scaling, within a fixed 320-KiB source overhead. The
one-panel N=1024 probe has a 189-MiB B tensor versus the source's
192-MiB B tensor at K=49152, so those working sets are not identical.

## Modeling decision

Keep the validated mode-2 GM→L1 traffic count, but do **not** fit the
online input-service rate from any of these copy-only schedules.
Changing MTE2 barrier granularity alone leaves the same systematic
error. The next useful measurement is an in-context producer/consumer
probe or a compact event schedule that includes L1→L0 operand use,
Cube, Fixpipe, buffer reuse, and waits across K packets. Its predictions
should be checked against the five held-out kernel excesses above
before enabling FP32 mode-2 family selection. No product selector or
primitive parameter was changed by this experiment.
