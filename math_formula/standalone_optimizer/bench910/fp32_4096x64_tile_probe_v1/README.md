# FP32 4096×64×27456: execute the estimator's BASE tile

The FP32 selector chooses BASE with 16 occupied AICs,
`singleCoreM/N/K=256/64/27456`, and `baseM/N/K=256/64/32`, scoring
it at **342.705 µs**. The earlier family comparison measured a
source-generated BASE tile of `128×128×64`, so it did not validate
the proposed spatial tile. This experiment tests the proposed tuple
directly on a 910B3.

## Method

- The [patch](patch_source.py) changes only the host tiler in an isolated
  copy of the source-matched research package. It applies only to
  same-type FP32, HF32 disabled, shape 4096×64×27456, forced BASE,
  with `MMV3_FP32_4096X64_TILE_PROBE=1`. It sets the proposed core and
  base dimensions, recomputes L1 steps and L2 tile assignment, and
  leaves the device binary unchanged.
- The experimental host-library SHA-256 is
  `79b1643572379e821c9b75c3b257926a2dec35e1e9ddf84e4f2b3e3f2734a264`.
  The FP32 device-binary SHA-256 in both the experimental and master
  packages is
  `41bb84c6f7363b494ed632c786e96722f2eb4f518a02342f141e764021cf7024`.
- The [runner](run.py) uses the existing `fp32_bench` and `msprof`
  protocol on idle device 4 with CANN 9.0.0. All three arms pass the
  output check with `max_relative_error=0.0`. Two rounds reverse arm
  order; each round reports the median of five profiled kernel tasks
  after five warmups. Timings exclude host scoring.

| Arm | Family | Emitted `baseM×baseN×baseK` | A/B L1 K steps | Core tile | Round 0 µs | Round 1 µs | Median µs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Clean master | deterministic Split-K | 128×128×64 | 3/3 | 4096×384×192 | 486.56 | 486.78 | **486.67** |
| Source BASE | BASE | 128×128×64 | 4/4 | 128×128×27456 | 539.78 | 550.94 | **545.36** |
| Proposed BASE | BASE | 256×64×32 | 4/16 | 256×64×27456 | 521.98 | 523.00 | **522.49** |

The proposed tile reduces BASE time by **4.2%** versus source BASE,
but is **7.4% slower** than master deterministic Split-K. Both rounds
have the same ranking. The proposed tile is valid and numerically
correct, so this wrong family decision is not explained solely by a
model-versus-source tiling mismatch.

The model scores proposed BASE at 342.705 µs and deterministic Split-K
at 377.629 µs. Relative to these measurements, it underestimates
proposed BASE by **34.4%** and deterministic Split-K by **22.4%**.
That unequal timing bias reverses the family ranking even when the
proposed BASE tile is executed. This identifies a remaining BASE
timing-model issue, without isolating its device-stage cause.

The [audited tiling and timings](summary.json) are saved here. Raw
`msprof` records are under
`/home/xianfei/solvebase_bench_20261025/cases/fp32_4096x64_tile_probe_v1`
on 910. The initial audit in that directory records a skipped hook;
the corrected audit and measurements produced the results above.

## Why the BASE score misses a copy-bound kernel

The [counter profiler](profile_counters.py) ran the proposed tile with
`PipeUtilization`, `MemoryAccess`, and `L2Cache` AIC metric groups, plus
master with `PipeUtilization`. The [median counters](counter_summary.json)
are from three profiled kernel tasks per group. The proposed BASE kernel
reports **99.24% MTE2 busy** and **48.07% Cube MAC busy**. Its requested
GM→L1 traffic is **549,120 KiB = 562,298,880 bytes**, exactly the
source-shaped count:

\[
D_A=4(4096)(27456)=449{,}839{,}104,\qquad
D_B=4(27456)(64)\lceil4096/256\rceil=112{,}459{,}776.
\]

Thus the model has the correct traffic *quantity*, while the measured
pipeline is transfer-service-limited. The model's Cube term is about
**249.9 µs** per AIC. Its inherited generic GM rate of 3.710 MB/µs
gives **151.6 µs** for the requested bytes; two `baseK=32` startup
streams add **188.1 µs**. The modeled output write and launch add
about **3.0 µs**, giving exactly the 342.705-µs objective. The measured
522.5-µs kernel has MTE2 active
for nearly its whole duration. Requested bytes divided by that
effective service time are about **1.08 MB/µs**. This 1.08-MB/µs number
includes issue, stalls, and overlap effects and is **not** a standalone
replacement for the generic hardware rate.
Its approximately 48.1% MAC busy share corresponds to about 251 µs,
close to the modeled 249.9-µs Cube term; Cube throughput is not the
leading discrepancy in this case.

The aggregate L2 read-hit fraction is **38.8%**. That observation is
consistent with material memory service, but it does not identify
which operand or cache/queue behavior causes the slower rate. The
master deterministic kernel also has high MTE2 utilization (96.69%).

`BaseStageUs` in the experimental selector takes the maximum of a Cube
term and generic byte-plus-startup term. Its optional
`base_fp32_mte2_service` bound is absent from the normal FP32 profile.
Even the opt-in diagnostic bound would not activate here: the A row
pitch is **109,824 bytes**, which is not divisible by its required
32-KiB pitch quantum. Its existing profile was developed for distinct
aligned-pitch cases, so simply applying its measured rate here would
be unsupported. A general BASE formulation needs a source-schedule
and access-geometry-dependent MTE2 service term, validated on held-out
narrow-N tiles and K pitches.
