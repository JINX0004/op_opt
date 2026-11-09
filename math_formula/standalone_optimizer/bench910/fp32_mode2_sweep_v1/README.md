# FP32 mode-2 fixed-tile K sweep on 910B3

This experiment tests the missing service in the [mode-2 reduced
formulation](../../SOLVE_SC_SPLITK_MODE2_FORMULATION.md). It uses the
source-built forced-family package on idle 910B3 devices 4–7, CANN 9.0.0,
same-type FP32 NN/ND/ND/ND, and `MMV3_RESEARCH_SPLITK_FAMILY=single`.
The source audit checks correctness, kernel provenance, family 2, and
the emitted tile before profiling. Each timed profile has five MatMulV3
tasks after five warmups; K=36864 and 49152 in the 2816×1024 series
have two profiler rounds. All **32 shapes** passed the audit and profile
checks. The source experiment script is [run.py](run.py); raw device
profiles remain in
`/home/xianfei/solvebase_bench_20261025/cases/fp32_mode2_sweep_v1`
on 910.

Within each series, the emitted mode-2 tile is fixed across K:

| Shape M×N | Device | `singleCoreM×singleCoreN×singleCoreK` | AICs | K values |
| --- | ---: | --- | ---: | ---: |
| 2816×1024 | 4 | 144×1024×256 | 20 | 12 |
| 3072×1024 | 5 | 160×1024×256 | 20 | 6 |
| 2048×2048 | 6 | 208×1024×256 | 20 | 6 |
| 2816×1536 | 7 | 144×1536×256 | 20 | 8 |

All have `baseM=baseN=128`, `baseK=64`, `stepM=2`, and 20 output
tiles, one per AIC. Thus each critical AIC runs one M panel and
$\lceil K/256\rceil$ sequential `IterateAll` calls. The
[analysis script](analyze.py) reproduces the implemented FP32 mode-2
objective for these audited tiles, including the K tail. Its
2816×1024×36864 score is **3098.383 µs**, matching the C++ score.
[results.csv](results.csv) gives every score, timing, emitted geometry,
and MAC/MTE1/MTE2/Fixpipe/scalar/AIC counters. The counters use the same
last five timed tasks as the runtime parser, but their stage durations
may overlap and are not additive.

## Fixed 2816×1024 tile as K changes

| K | K slices | Current score µs | Measured µs | Residual per slice µs |
| ---: | ---: | ---: | ---: | ---: |
| 27392 | 107 | 2306.7 | 2901.3 | 5.56 |
| 32768 | 128 | 2756.0 | 3540.0 | 6.13 |
| 36864 | 144 | 3098.4 | 3943.3 | 5.87 |
| 40960 | 160 | 3440.7 | 4458.4 | 6.36 |
| 45056 | 176 | 3783.1 | 4825.0 | 5.92 |
| 47104 | 184 | 3954.2 | 5048.4 | 5.95 |
| 48128 | 188 | 4039.8 | 5160.9 | 5.96 |
| 48896 | 191 | 4104.0 | 5242.3 | 5.96 |
| **49152** | **192** | **4125.4** | **5679.9** | **8.10** |
| 49408 | 193 | 4146.8 | 5291.2 | 5.93 |
| 57344 | 224 | 4810.1 | 6375.6 | 6.99 |
| 65408 | 256 | 5487.3 | 7043.7 | 6.08 |

Most of this series has an exposed residual around **6 µs per K
slice**; this is a diagnostic difference, not a calibrated parameter.
The exceptional K=49152 point is about **413 µs slower** than linear
interpolation between K=48896 and 49408, while its MTE2 counter is
about **349 µs above** the same interpolation. The two K=49152
rounds measured 5630.5 and 5729.4 µs, so the spike is larger than
their observed round spread. The source tile and workspace are identical
at the three adjacent K values.

## Geometry dependence and the K=49152 spike

| M×N | Median residual/slice excluding K=49152, µs | K=49152 excess over adjacent K interpolation, µs | MTE2 excess, µs |
| --- | ---: | ---: | ---: |
| 2816×1024 | 5.96 | 413 | 349 |
| 3072×1024 | 5.08 | 649 | 606 |
| 2048×2048 | 2.33 | 60 | 55 |
| 2816×1536 | 7.35 | 506 | 390 |

The baseline residual changes markedly with spatial tile geometry,
even though `singleCoreK`, base tiling, and core count stay fixed. A
single per-`IterateAll` constant cannot explain all four series. The
K=49152 excess repeats for three geometries and is most visible in
MTE2; it is small for the 208×1024 tile. This separates a general
geometry-dependent overlap/handoff deficit from an additional
K-specific transfer anomaly.

The queried 910B3 L2 capacity is **192 MiB**. At K=49152, each FP32
A row spans **192 KiB**; B spans 192 MiB when N=1024, 288 MiB when
N=1536, and 384 MiB when N=2048. The source enables B's L2 hint in
all four series. It enables A's L2 hint only for 2048×2048 because
that series splits N across cores. These facts make cache mapping,
stride aliasing, or a cache-policy interaction plausible; the spike
also occurs when B is larger than L2, so a simple “B reaches L2
capacity” rule is contradicted by the data. The aggregate profiler
counters do not prove which cache or DMA dependency caused it.

## Modeling implication

The existing score's Cube service tracks the measured MAC counter
reasonably well, but the `max(Cube, read + output)` slice bound hides
exposed work that scales with geometry. A first **opt-in** C++
[pipeline scorer](../fp32_mode2_v1/remote_selector/model/mip_solver.cpp)
now adds L1→L0 A/B demand and a last output-piece drain. Its FP32
mode-2 rates are Huawei's nominal 910B3 links, 460,800 and 230,400
bytes/µs per AIC; these are not fitted runtime corrections or measured
source-path rates. The default hardware profile and online selector do
not supply the new primitives, so their behavior is unchanged.

For a K slice, let $R$ be the existing GM→L1 read service, $L_A,L_B$
the nominal L1→L0 A/B services, $C$ the Cube service, $F$ the total
Fixpipe service, and $F_1$ the service for one full base output piece.
The experimental max-plus path bound is

$$
T_{\rm slice}^{\rm pipe}
=\max\big(C+F_1,\;R+L_A+L_B+F_1,\;F\big).
$$

It serializes the read and two load services on the input path, allows
that path to overlap Cube across base pieces, and retains one output
drain after the last wave. Repeated full K slices still use the
constant-time multiplication in `ScKChainUs`. This is a reduced source
schedule, **not** the Matmul engine's complete internal event graph;
the exact A/B issue order and buffer waits are not visible in the
MatMulV3 source. The output service remains the provisional mode-6
atomic-output proxy.

The [C++ scorer](score_pipeline.cpp) and [32-case comparison](compare_pipeline.py)
use exactly the audited source tiles and existing device measurements.
The [per-case results](pipeline_results.csv) are:

| Cases | Original mean absolute error | Opt-in pipeline error |
| --- | ---: | ---: |
| All 32 | 18.21% | **3.22%** |
| 28 excluding K=49152 | 17.76% | **2.59%** |

For 2816×1024×36864, the source mode-2 score moves from 3098.38 to
**3934.88 µs**, versus **3943.28 µs** measured in this sweep. The
source BASE score is 3358.12 µs, so the revised *source-tile*
comparison ranks BASE ahead of mode 2, matching the paired device
result. The opt-in score does not fix every family ranking:
2048×2048×27392 scores mode 2 at 3322.44 µs versus source BASE at
3493.27 µs, while the measured times are about 3508 and 3468 µs.
The [opt-in solver check](select_pipeline.cpp) chooses BASE for both
shapes when it uses its *optimized hypothetical* BASE score (3022.58
and 3243.94 µs). That is not a source-emitted BASE comparison and
does not resolve the second mismatch.

The separate [11-source-tile comparison](compare_source_tiles.py) checks
seven tiles outside the four fixed-tile K series, including wider N
and a multi-panel large-M tile. Their mean absolute error decreases
from **9.88% to 5.68%**; [individual results](source_tile_pipeline_comparison.csv)
remain mostly underestimated. K=49152 and cache/stride effects are not
modeled. The next measurements should isolate FP32 mode-2 A/B L1→L0
service and the `IterateAll` handoff, then test a full per-base-piece
event graph on held-out tiles. Investigate K=49152 separately by
changing physical row stride or padding and collecting cache or
operand-specific MTE2 counters. The FP32 online family gate remains
closed.
