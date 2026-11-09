# FP32 mode-2 source-shaped stage ladder on 910B3

## Question

Earlier A/B copy probes overpredicted the actual MatMulV3 kernel's
192-KiB A-row-pitch penalty. Adding a fixed L1→L0 handoff and changing
A/B L2 cache hints did not close the gap. This experiment adds Cube and
then Fixpipe work to a source-ordered A/B probe, asking which stage
first changes its center-pitch response toward the real kernel.

## Fixed work and modes

The two [source-audited tiles](fixed_work_summary.json) are
3072×1024×48384, with a large pitch penalty, and 2048×2048×48384,
with a small one. Both use 20 AICs, mode 2, `baseM=baseN=128`,
`baseK=64`, `singleCoreK=256`, and tiling key 65568. A's logical K,
all B data, tile, and physical work stay fixed while A's row pitch
varies among 48896, 49152, and 49408 FP32 elements (191/192/193 KiB).
The source package and host tiler match the earlier
[fixed-work experiment](../fp32_mode2_in_context_pitch_v1/README.md).

The [AscendC probe](probe_kernel.cpp) assigns source-shaped M×N tiles
to the 20 AICs and loops over inner-M panel, K packet, then 128-column
B pieces. Each mode copies one A panel and the same B pieces from GM
to L1. A and B use the source's normal cache policy. Destination-reuse
waits are identical across modes.

| Mode | Additional work |
| ---: | --- |
| 0 | GM→L1 A/B copies and waits |
| 1 | Four 128×64 FP32 L1→L0A/B loads per B piece |
| 2 | Mode 1 plus four `Mmad` accumulations per 128×128 M block |
| 3 | Mode 2 plus FP32 Fixpipe to disjoint scratch tiles, plain on the first K packet and atomic afterwards |

The probe replays one L1 prefix for its four K fragments. It matches
the amount of L0 and Cube work but **does not reproduce source operand
values or `MatmulImpl`'s internal buffer schedule**. It drains Cube
before rewriting the same L0A/L0B addresses; an earlier build without
that wait produced a confirmed L0B read/write conflict. This is a
serialized diagnostic schedule, not a claim of source-equivalent
overlap. The [runner](probe_runner.cpp) checks a completion token from
every AIC; the token's scalar GM store is flushed from device cache.
Mode 3 also checks that Fixpipe wrote a finite positive scratch value.
The binary is built with [build.sh](build.sh) against CANN 9.0.0.

The [run driver](run.py) interleaves the actual source-built MatMulV3
kernel at every pitch. Four rounds reverse both pitch and mode order
on 910B3 devices 5 and 6. It checks source output and saves all
accepted timings. [analyze.py](analyze.py) computes center-pitch
excess as `T(49152) − [T(48896)+T(49408)]/2`, from per-pitch medians
and separately for each round. No selector or product kernel is changed.

## Measured center-pitch excess

The four-round [probe records](records.json) contain 96 checked runs;
the [source controls](source_controls.json) contain 24 output-checked
MatMulV3 runs. [Analysis](analysis.json) preserves every per-pitch
median and round-level excess. The [binary hashes](binary_hashes.json)
and [run log](run.log) preserve provenance and order. Values below are
the 192-KiB center excess in µs, from the median at each pitch:

| M×N | Actual MatMulV3 | 0: copy | 1: +L0 | 2: +Cube | 3: +Fixpipe |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3072×1024 | 739 | 1112 | 1006 | 432 | 30 |
| 2048×2048 | 27 | 672 | 675 | 655 | 469 |

For 3072×1024, adding Cube reduces the probe excess by 574 µs,
crossing from above the actual kernel's 739 µs to below it. Adding
Fixpipe reduces it by another 402 µs. For 2048×2048, Cube changes the
excess by only −20 µs and Fixpipe by −186 µs; the resulting 469 µs
remains far above the actual kernel. The source control itself moves
between rounds: its center excess ranges from 648–872 µs for the
first shape and −38–156 µs for the second. The probe effects are more
stable: mode 2 ranges 406–438 and 647–660 µs respectively. These are
repeat observations, not confidence intervals.

## Pipe and traffic checks

The [24 pitch profiles](profile_pitch_summary.json) confirm that mode
1 adds MTE1 work, mode 2 adds Cube work, and mode 3 adds Fixpipe work.
At 191 KiB, the 3072×1024 MTE1 busy time rises from approximately
zero to 1368 µs, Cube busy time rises to 4469 µs in mode 2, and
Fixpipe busy time rises to 1350 µs in mode 3. The corresponding
2048×2048 Cube and Fixpipe busy times are 5694 and 1660 µs. Stage
busy times overlap and are not additive elapsed stages.

The [MemoryAccess profiles](traffic_summary.json) report the same
requested GM→L1 amount for **all four modes** at 192 KiB: 4,451,328
counter units for 3072×1024 and 4,644,864 for 2048×2048. Mode 3
adds L0C→GM output traffic as intended. The
[counter analysis](profile_analysis.json) shows that each mode's
elapsed center-pitch excess closely follows its **MTE2 busy-time**
excess:

| M×N | Mode 0 MTE2 | Mode 1 MTE2 | Mode 2 MTE2 | Mode 3 MTE2 |
| --- | ---: | ---: | ---: | ---: |
| 3072×1024 | 1089 | 1002 | 446 | 27 |
| 2048×2048 | 664 | 662 | 639 | 484 |

MTE1, Cube, and Fixpipe busy-time center excesses are approximately
zero. Thus Cube and Fixpipe change the observed input-service time
itself despite identical requested GM→L1 volume; the decrease cannot
be explained solely as a constant compute/output cost hiding an
unchanged copy penalty. These counters do not identify whether L2
residency, request pacing, or a lower-level queue stall causes the
change.

## L2 check

The [18 L2Cache profiles](l2_summary.json) cover modes 0, 2, and 3
at all three pitches on both shapes. Aggregate AIC read-hit fractions
are 82.6087% for 3072×1024 and 83.3334% for 2048×2048 in modes 0
and 2 at every pitch. Mode 3 is the same within 0.0022 percentage
points. Thus the large mode-0-to-mode-2 MTE2 excess change does **not**
come with an aggregate L2 hit/miss-rate change. These counters do not
separate A/B, cache banks, request queues, or latency distributions,
so they cannot identify the physical stall. They support testing
request spacing and source buffer/queue waits next.

## Model consequence and limits

No stage in this serialized ladder reproduces both actual-kernel
geometries. The event model should not use a copy duration indexed
only by bytes, physical pitch, and tile. Input service must also
depend on the issue/wait schedule and concurrent output activity.
The probe's explicit per-fragment drain, replayed L0 data, synthetic
input values, and scratch output differ from `MatmulImpl`; they limit
any direct rate calibration. The next useful model refinement is a
schedule-dependent MTE2 service term with explicit request spacing and
buffer-reuse waits, checked against real source profiles and held-out
geometry before entering the online selector.
