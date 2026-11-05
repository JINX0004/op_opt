# BF16 online selector host cost and end-to-end timing (910B3)

Measured 2026-11-05 on device 4, CANN 9.0.0, using the BF16 source-BASE
online package and the clean-master package from the preceding
[`bf16_source_base_v1` comparison](../bf16_source_base_v1/README.md).
The online arm had `MMV3_ONLINE_SELECTOR=1` and
`MMV3_ONLINE_BF16_EXPERIMENT=1`. The benchmark-only BF16 dtype-inference
patch was present in both packages. No other workloads were intentionally
run on device 4 during the paired timing runs.
The online source branch was `codex/matmul-v3-mip-selector` at `aad7d8ce`.
The host tiling library SHA-256 values were `1fd89d86dc5a7b37af3da56cbf0bad74da4be440d940de5b96faa8308d9ee4d1`
(online) and `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`
(master).

## What was measured

`score_latency.cpp` calls the actual `LookupOnlineMipMode` implementation
linked to the branch's `mip_solver.cpp` and `plan_finalizer.cpp`. For each
of the 100 preceding test shapes, it measures one cold call with a
representative source-BASE tile and the median of 101 cache hits. It runs
the 100-shape sequence twice in separate processes. The source-BASE tile
is a proxy for some shapes; the shape-dependent solver work is the same,
but the returned mode can differ from the product decision. Treat these
numbers as scorer timing, not a new selection audit.

`host_wall_bench.cpp` times three representative shapes through the actual
BF16 MatMulV3 packages. `first_wall_us` covers first enqueue (including
CANN setup/compilation) and stream synchronization. For repeated calls,
`wall_median_us` covers event recording, cached enqueue, and synchronization;
`host_median_us` covers cached enqueue alone. Each arm had two independent
processes in reversed order, three warmups and 11 timed calls per process.
The patterned-output correctness check passed in all 12 processes.

## Direct scorer cost

| 100 shapes | Run 1 | Run 2 |
| --- | ---: | ---: |
| Cold median | 50.9 ms | 53.8 ms |
| Cold 90th percentile | 270.3 ms | 301.6 ms |
| Cold 95th percentile | 324.9 ms | 403.3 ms |
| Cold maximum | 584.6 ms | 584.5 ms |
| Cached lookup median | 0.35 µs | 0.35 µs |

The cold cost rises with the shape and the work done by `SolveReducedMip`,
which constructs spatial geometries and solves each implementation family.
The online selector caches a decision by shape, dtype, and source-BASE
geometry. A plog inspection for `16×16×65408` showed only one online
decision across initial compilation, warmup, and repeated calls in one
process; CANN's compiled-tiling cache also bypassed scoring on those
repeated calls. Thus the cold cost is incurred at first use of a distinct
configuration, not on every steady-state invocation.

## MatMulV3 end-to-end sample

Values below are medians of the two process medians. First-call values
include approximately 12 seconds of CANN setup/compilation and are too
variable to isolate 0.1–0.2 seconds of scoring. Repeated-call values are
wall time per synchronized call, not profiler kernel time.

| Shape M×N×K | First master / online | Repeated wall master / online | Repeated host enqueue master / online |
| --- | ---: | ---: | ---: |
| 16×16×65408 | 12.566 / 11.477 s | 501 / 339 µs | 185 / 105 µs |
| 2560×768×32768 | 13.086 / 12.483 s | 1773 / 1132 µs | 211 / 210 µs |
| 65536×384×1536 | 12.807 / 11.863 s | 650 / 706 µs | 96 / 180 µs |

The unchanged-tiling `16×16×65408` sample has a large apparent wall-time
difference despite identical emitted tiles and earlier near-identical
profiler kernel times (31.84 versus 30.98 µs). This shows the host/event
timings fluctuate enough that small differences should not be attributed
to the selector. The changed-family `2560×768×32768` case retains a large
repeated-call improvement in both process orders. The large-M sample does
not establish an improvement: its repeated-call intervals overlap across
orders. Both arms' raw first-call times were dominated by CANN; the online
arm happened to be faster in this small sample, so these data do not
demonstrate a first-call regression or prove its absence.

## Follow-up: unchanged tilings after CANN initialization

To isolate cold scoring from the approximately 12-second process startup,
`run_unchanged.py` first compiles a different BF16 shape (`16×16×128`) in
each process. It then times the first call of the target shape, whose
tiling and online decision are still uncached. The target is warmed three
times before 21 synchronized repeated-call measurements. There are three
independent master/online pairs per shape, with reversed order in the
middle pair. All 18 target runs passed the patterned-output correctness
check. The preceding 100-shape audit verified that **all three selected
shapes emit the same complete tile in both packages**, including mode,
single-core tile, base tile, steps, block dimension, and workspace.

| Unchanged shape M×N×K | Direct cold score | First target call master / online | Paired online penalty | Repeated wall master / online |
| --- | ---: | ---: | ---: | ---: |
| 16×16×65408 | 0.26–0.41 ms | 0.819 / 1.101 ms | 0.187–0.453 ms | 388 / 281 µs |
| 2048×4096×4096 | 120–126 ms | 0.810 / 122.243 ms | 121.1–122.8 ms | 614 / 733 µs |
| 8192×8192×16384 | 584–585 ms | 10.275 / 599.465 ms | 587.3–678.7 ms | 9922 / 9978 µs |

First-call and repeated-call arm columns are medians of three process
medians. The penalty column is the range of the three paired first-call
differences. In each pair, the added first-call time was almost entirely
inside host enqueue: its difference agreed with the wall-time difference
within about 0.2 ms. The medium and large penalties match direct scoring
time. The third large-shape pair had a 679 ms scoring/host-enqueue tail.

**Conclusion:** for an unseen shape with an unchanged tiling, the online
selector measurably lengthens the first target call. The added cost is
approximately the cold scoring duration, while repeated calls do not pay
it again. Repeated wall times vary with host scheduling and show no
reliable selector overhead; identical tiles had near-identical profiler
kernel times in the preceding experiment. The raw, unprewarmed first-call
measurements above obscured this effect with CANN process initialization.

## Amortization against the earlier kernel measurements

If a shape is new to the process, a simple incremental-cost estimate is

`online extra time ≈ cold scoring time − calls × (master kernel time − online kernel time)`.

This omits differences in CANN compilation, tiling, and host dispatch. It
is an optimistic break-even calculation for the 18 changed-family cases
in the prior profiler comparison. Scoring time is the median of the two
direct runs; kernel savings are the median of the two paired profiling
rounds. The break-even count includes the first call.

| Shape M×N×K | Cold score ms | Kernel saved per call µs | Approx. calls to break even |
| --- | ---: | ---: | ---: |
| 1792×768×32768 | 177.8 | 85.1 | 2088 |
| 2048×768×27392 | 127.7 | 71.6 | 1784 |
| 2304×768×32768 | 164.2 | 381.4 | 431 |
| 2560×768×32768 | 132.4 | 525.2 | 253 |
| 2048×1024×32768 | 114.8 | 244.0 | 471 |
| 2304×1024×36864 | 145.6 | 106.7 | 1365 |
| 2816×1024×36864 | 164.5 | 145.0 | 1135 |
| 3072×1024×36864 | 163.3 | 118.3 | 1381 |
| 2560×1024×32768 | 118.9 | 533.2 | 224 |
| 2560×1024×49152 | 125.3 | 51.9 | 2415 |
| 3072×1024×73728 | 168.8 | 423.6 | 399 |
| 768×2048×32768 | 136.3 | 136.8 | 997 |
| 1664×768×32768 | 158.3 | 172.7 | 917 |
| 1920×768×32768 | 158.8 | 148.8 | 1068 |
| 2176×768×32768 | 172.7 | 322.4 | 536 |
| 2432×768×32768 | 147.6 | 443.7 | 333 |
| 2304×1024×32768 | 142.4 | 381.0 | 374 |
| 2816×1024×49152 | 164.1 | 26.7 | 6157 |

The median break-even count is approximately **957 calls** (range 224–6157).
For one-shot or low-reuse shapes, the current scorer can outweigh even a
substantial kernel improvement. For repeatedly used shapes, scoring is
effectively one-time and the earlier kernel speedups apply. The most useful
next change is to reduce cold solving cost, for example by pruning
implementation families and spatial geometries before the detailed model;
then remeasure the cold scorer and first-call MatMulV3 latency. The product
selector's existing cache should be retained.

Raw data: [score pass 1](score_latency.txt),
[score pass 2](score_latency_repeat.txt),
[paired wall records](host_wall_records.json), and
[unchanged-tiling paired records](unchanged_records.json),
[shape list](shapes.txt). Reproduction programs:
[direct scorer](score_latency.cpp),
[MatMulV3 harness](host_wall_bench.cpp),
[paired runner](run_host_wall.py), and
[unchanged-tiling runner](run_unchanged.py).
