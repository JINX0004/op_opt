# FP32 BASE source-tile timeline, 910B3

This experiment separates an optimized, hypothetical BASE tile from the
BASE tile emitted by MatMulV3. The latter is required when comparing BASE
with deterministic or single-core Split-K for an online family decision.
The FP32 product selector remains disabled; this is a diagnostic model.

## Model change

`SourceBaseL1Steps` now uses `ElementBytes(A)` and `ElementBytes(B)` for
both L1 packet capacity and 256/512-byte step alignment. In particular,
same-type FP32 uses four-byte elements. The ordinary optimized BASE score
retains its previous surrogate. A caller explicitly requests the
source-emitted BASE score with
`ReducedObjectiveUs(workload, hardware, decision, memory, true)` or
`mmopt_selector --source-base` followed by
`M N K fp32 usedCores coreM coreN baseM baseN baseK`.

For a source tile, let $h_A,h_B$ be its L1 K steps, $B_K$ its baseK,
$S_q$ the spatial base packets assigned to AIC $q$, and $P_K=\lceil
K/B_K\rceil$. The source BASE memory branch now counts

$$
S_q\left[\left\lceil\frac{K}{h_AB_K}\right\rceil\lambda_A+
          \left\lceil\frac{K}{h_BB_K}\right\rceil\lambda_B+
          P_K\lambda_{\rm packet}^{32}\right]
$$

plus the existing four-byte A/B payload and FP32 output terms. The
`base_fp32_packet_issue` primitive supplies $\lambda_{\rm packet}^{32}$.
The model also computes $\lceil K/8192\rceil$ internal FP32 calls and
accepts optional `base_fp32_call` and `base_fp32_atomic_output` services,
but neither is installed in the profile: boundary measurements below do
not isolate a positive fixed call penalty. These are *exposed* service
terms in the reduced model, not an event-graph schedule.

## Controlled measurements

`run.py` used the existing forced-family package and `msprof` on idle
devices 4–7. Every 16×16 case emitted `baseM=baseN=16`, `baseK=32`,
`stepKa=stepKb=64`, and one occupied output AIC. The `summary.json`
records source tiles, device IDs, timing, and profiler CSV paths. The
calibration used K=8064, 8192, 16384, and 32768. Least squares through
the origin on measured time minus the source-step byte/startup score,
against $P_K$, gives **0.06636304 µs per baseK packet**. This is an
in-context effective service slope; it does not identify MTE1 alone.

| 16×16 K | Source BASE measured µs | Source-step model µs | With packet term µs | Role |
| ---: | ---: | ---: | ---: | --- |
| 8192 | 21.56 | 4.00 | 20.99 | Calibration |
| 8256 | 21.98 | 4.22 | 21.35 | Held out |
| 16384 | 39.28 | 5.16 | 39.14 | Calibration |
| 16448 | 38.48 | 5.38 | 39.49 | Held out |
| 32768 | 75.28 | 7.48 | 75.44 | Calibration |
| 32832 | 74.72 | 7.70 | 75.79 | Held out |
| 65408 | 145.46 | 12.12 | 147.76 | Held out |

The 8192→8256, 16384→16448, and 32768→32832 pairs change the internal
call count with little change in K. They show no consistent positive
runtime jump. The current data support a K-packet service more strongly
than a separate per-call startup. `probe_kernel.cpp` and
`probe_runner.cpp` independently time GM→L1 A and B packets; the source
16×2048 FP32 packet has approximately 1.45 µs/copy for A and 0.52
µs/copy for B after subtracting the one-copy launch baseline. Those
transfers explain part, but not all, of the full-kernel slope. Raw
results and the reproducible sweep are in `probe_results.json` and
`measure_probe.py`.

The same packet coefficient is insufficient for wider output tiles:
source BASE measured 21.88 µs at 64×64×8192 and 34.54 µs at
128×128×8192, versus approximately 18.11 and 25.47 µs with the new
model. The FP32 source path still needs geometry-dependent transfer or
per-baseK service and an output-path check.

## Recorded 100-case comparison

`reselect_100.py` rescored the source BASE tile whenever either measured
arm emitted BASE in the previous paired FP32 study. It covered **81 of
100** shapes. The ordinary hypothetical selector still makes the same
100 family decisions as before; only explicit source-tile scoring uses
the new timeline. `source_reselection.csv` records all scores and paired
measurements.

Five of the 81 family choices change when source BASE replaces the
hypothetical BASE score:

| Shape | New choice | Paired BASE µs | Paired alternative µs | Assessment |
| --- | ---: | ---: | ---: | --- |
| 16×16×65408 | deterministic | 146.40 | 39.98 | Corrects a large regression |
| 2048×768×27392 | deterministic | 1469.12 | 1435.48 | Small improvement |
| 768×2048×32768 | deterministic | 1806.40 | 1646.34 | Improvement |
| 2816×1024×36864 | mode 2 | 3315.91 | 3938.92 | Wrong family change |
| 2048×2048×27392 | mode 2 | 3468.05 | 3507.05 | Wrong family change |

The mode-2 errors prevent enabling this FP32 family selector in the
host tiler. The next calibration should target source-backed mode-2
output and memory service and broader BASE tile geometry, then repeat
the paired comparison against `master`. The 19 shapes without a
source-emitted BASE tile need an audit before claiming full 100-case
coverage.

## Fresh paired rerun of the five changed families

`five_case_rerun.py` reran all five on idle 910B3 devices 4–7, using
clean master, the previous choice forced to source BASE, and the new
choice forced to its source family. Each arm had two reverse-order
`msprof` rounds of five kernel tasks. The new family was the same as
master on every shape, and their emitted tilings matched field for
field. All 15 audit executions had zero reported relative error.
`five_case_summary.json` contains the per-arm medians and emitted tiles;
the raw per-round profiler records remain under
`/home/xianfei/solvebase_bench_20261025/cases/fp32_base_five_family_rerun_v1`.

| Shape | New family | Master µs | Old BASE µs | New choice µs | New vs old BASE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16×16×65408 | 3 | 38.72 | 142.65 | 39.73 | −72.15% |
| 2048×768×27392 | 3 | 1424.63 | 1468.81 | 1420.83 | −3.27% |
| 2816×1024×36864 | 2 | 3924.62 | 3321.22 | 3928.70 | **+18.29%** |
| 768×2048×32768 | 3 | 1648.01 | 1800.77 | 1651.64 | −8.28% |
| 2048×2048×27392 | 2 | 3508.90 | 3463.26 | 3507.99 | +1.29% |

The two mode-2 regressions replicate the direction of the recorded
100-case comparison. In particular, the 2816×1024×36864 mode-2 error
is far larger than the variation between rounds. The host FP32 selector
gate therefore remains closed. These timings compare actual source
family implementations; the new selector decision itself was replayed
as a forced family because the online FP32 host gate is closed.

## Why 2816×1024×36864 is selected incorrectly

The new source BASE score is **3358.12 µs**, close to the forced BASE
measurement of **3321.22 µs**. The source mode-2 score is **3098.38 µs**,
but the same emitted mode-2 tile measures **3928.70 µs**. Thus this is
principally a **mode-2 underestimation**, rather than a BASE feasibility
or tile mismatch. Both master and the new choice emit mode 2 with
`singleCoreM/N/K=144/1024/256`, `baseM/N/K=128/128/64`, and 20 AICs;
the forced BASE arm emits `128/256/36864` with 20 AICs.

There are 144 sequential K slices per occupied mode-2 AIC. For the full
144×1024×256 slice, `ScSliceUs` computes 20.97 µs Cube service, 14.92 µs
generic read service, and 6.48 µs steady atomic output service. Its
`max(cube, read + output)` is therefore 21.40 µs for each later slice.
The first slice costs 36.06 µs because it uses the slower non-atomic
output primitive. Together with the 2.68 µs launch term, this reproduces
the **3098.38 µs** score exactly. In particular, the later-slice score
assumes nearly perfect overlap between MAC, transfers, output, and
per-`IterateAll` control work; `ScSliceUs` has no separate FP32 mode-2
iteration or dependency cost. The steady output primitive is also a
provisional mode-6 proxy, not a mode-2 calibration.

Fresh `msprof` summaries for the identical master/new mode-2 tile show
approximately 3835 µs active AIC time and 3026 µs MAC time, versus a
3929 µs kernel task. The modeled Cube service across all slices is about
3020 µs, which agrees with the measured MAC time. The missing time is
roughly **809 µs inside the AIC** (5.62 µs per K slice beyond MAC), plus
about **94 µs between the task duration and active AIC time**. These
differences are diagnostic, not a decomposition into additive hardware
stages: MTE1, MTE2, Fixpipe, and scalar counters can overlap one another
and MAC. They show that the model's full-overlap assumption is too
optimistic here; the profiler alone cannot identify the exact wait edge.

The comparison shape 2048×2048×27392 uses the same 256-wide K slices
but a 208×1024 output tile. Its 107-slice score is 3257.52 µs and
measurement is 3507.99 µs, a smaller **2.34 µs per-slice** gap. Active
AIC time exceeds MAC time by about 338 µs. This rules out simply adding
one universal per-slice constant as a general correction. An earlier
2816×1024×49152 run with the same 144×1024×256 tile also has a larger
mode-2 gap (4125.40 µs estimated versus 5716.10 µs measured); its
MTE2 counter grows faster than K, so input residency may also matter.
The latter is a hypothesis requiring a controlled sweep.

The next calibration should hold M/N and the emitted tile fixed while
sweeping K, then repeat with several `singleCoreM` and `singleCoreN`
values. Record per-AIC MAC, MTE1, MTE2, Fixpipe, scalar, and active time;
fit an exposed `IterateAll`/handoff term and test a separate input/output
service schedule against held-out tiles. A mode-5 iteration coefficient
cannot be directly reused: its path and geometry differ, and its 1.8 µs
value would explain only 259 µs of this case's 830 µs score error.
