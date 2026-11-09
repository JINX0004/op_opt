# FP32 BASE formula: K-stride failure and next model structure

This study isolates two large errors in the experimental 910B3 FP32 BASE
source-tile objective. It follows the [100-case family comparison](../fp32_mode2_pipeline_100_v1/README.md)
and its [21 deterministic-to-BASE source-tile scores](../fp32_mode2_pipeline_100_v1/source_base_det_diagnosis.csv).
The product FP32 online gate remains disabled.

## Controlled whole-kernel result

[run_same_device.py](run_same_device.py) forces only the BASE family through
the existing source-built MatMulV3 package. It keeps each M/N series on one
910B3 device, measures K values in ascending then descending order, and
uses five warmups plus five `msprof` kernel samples per round. Audits verify
same-type FP32 NN, correctness (maximum relative error 0), source tiling,
and idle devices. Both series emit the **same 128×256×32 BASE tile** with
`stepKa=8`, `stepKb=4`, and 20 occupied AICs. The [raw paired records](same_device_records.json)
and [analysis script](analyze.py) preserve the protocol. The source score
uses the same `ExperimentalFp32Mode2Pipeline910B3()` hardware profile as
the current experimental selection; that mode-2 addition does not alter
BASE.

| M×N×K | Source BASE score µs | Measured µs | Two rounds µs |
| --- | ---: | ---: | --- |
| 2560×768×32512 | 1778.3 | 1748.7 | 1748.1, 1749.2 |
| **2560×768×32768** | **1792.2** | **2398.4** | 2465.5, 2331.4 |
| 2560×768×33024 | 1806.2 | 1775.8 | 1772.6, 1778.9 |
| 2560×1024×48896 | 3563.2 | 3495.3 | 3498.6, 3492.0 |
| **2560×1024×49152** | **3581.8** | **4728.7** | 4808.6, 4648.8 |
| 2560×1024×49408 | 3600.5 | 3526.8 | 3527.3, 3526.4 |

At K=32768, measured runtime is **36.1%** above the mean of its two
neighbors; at K=49152, it is **34.7%** above. The current score changes
smoothly with K and is within about 2% of all four neighboring timings.
The [first cross-device sweep](k_sweep_records.json) also measured
2560×768×36864 at 2028.2 µs versus a 2015.9 µs score; the repeated
same-device runs establish that the exact-K spikes are not device
assignment artifacts. [comparison.csv](comparison.csv) contains all six
same-device scores and timings.

Across the [earlier 81 cases with an audited source BASE tile](source_base_81.csv), the error
has distinct regimes. The **59 cases with logical N>128 and K outside
{32768,49152}** have 2.3% mean absolute percentage error and near-zero
mean bias. The seven wide-N cases at those exact K values have 19.8%
mean absolute error, while 15 cases with N≤128 have 21.9%. These are
descriptive subsets of the fixed stress set; one exact-K case with a
small enough A matrix is also scored accurately. The result argues
against changing the global FP32 Cube or GM rate to fix the outliers.

## Independent operand probe

[run_stride_probe.py](run_stride_probe.py) reuses the earlier source-shaped
GM→L1 probe on device 4, with 20 AICs and the emitted BASE L1 steps:
A reads 128×256 FP32 elements with row pitch K; B reads 128×256 with
row pitch N. It reports the incremental service `(128-copy time −
1-copy time)/127` after 100 repetitions. The [raw probes](stride_probe.json)
and [derived values](probe_summary.csv) include every measurement.

| M×N | Operand | K below / exact / above | Incremental µs per copy |
| --- | --- | --- | --- |
| 2560×768 | A | 32512 / **32768** / 33024 | 1.988 / **2.304** / 1.967 |
| 2560×768 | B | 32512 / **32768** / 33024 | 0.750 / **0.733** / 0.756 |
| 2560×1024 | A | 48896 / **49152** / 49408 | 2.007 / **2.708** / 2.040 |
| 2560×1024 | B | 48896 / **49152** / 49408 | 0.772 / **0.769** / 0.774 |

A's isolated service rises by about **16.5%** and **33.8%** at the two
exact K values, while B is stable. In FP32, A row pitches are respectively
128 KiB and 192 KiB at those K values. This implicates A-side access
geometry, without identifying a particular cache set or memory bank.

## Source-sized streams and pipeline counters

The [full-tile probe](source_stream_probe/probe_kernel.cpp) traverses the
same source-sized 128×256 output tiles, with A windows of 128×256 and B
windows of 128×256. It runs A alone, B alone, interleaved A+B, and an
approximation of the source's four-M-tile L2 regions. The [driver scripts](run_combined_stream.py)
and [raw outputs](combined_stream.json) are preserved, as are the
[single-operand](source_stream.json) and [region](region_stream.json) runs.
The [analysis](analyze_pipeline.py) reports each probe's extra time at an
exact K relative to its ±256 neighbors:

| M×N, central K | Whole BASE excess | A only | B only | A+B interleaved | A+B region |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2560×768, 32768 | 636 µs | 31 µs | 1 µs | 347 µs | 59 µs |
| 2560×1024, 49152 | 1218 µs | 129 µs | 2 µs | 659 µs | 515 µs |

The region-only probe serializes copies with MTE2 barriers and omits
Cube/MTE1 overlap. Its baseline and spike vary substantially from the
source kernel, so its time is **not** a standalone predictor.

We profiled the source-built kernel on device 4 with `PipeUtilization`,
`MemoryAccess`, and `L2Cache` AIC metric groups. The [profiler driver](profile_pipeline.py),
[extractor](summarize_pipeline.py), and [median counters](pipeline_metrics.json)
record the experiment. `GM→L1` values below are requested bytes; the
L2 hit ratio is derived from the R0 and R1 read counters.

| K, 2560×768 | AIC cycles (20-core aggregate) | MAC busy | MTE2 busy | GM→L1 | L2 read hit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32512 | 62.97 M | 99.2% | 99.2% | 2.996 GB | 56.67% |
| **32768** | **84.91 M** | **74.7%** | **99.7%** | **3.020 GB** | **56.67%** |
| 33024 | 63.77 M | 99.5% | 98.7% | 3.043 GB | 56.67% |

Thus requested transfer volume and L2 hit fraction change smoothly, while
MTE2 remains saturated and Cube utilization falls. The missing quantity
is effective transfer service under this access schedule. The counters
do not identify the physical bottleneck more narrowly than that.

## More K pitches and source grouping

The [fixed-M/N sweep](run_stride_grid.py) and [task-level neighbor profiles](run_stride_neighbors.py)
keep the emitted 128×256×32 tile, `stepKa=8`, `stepKb=4`, and 20 AICs.
The [task-level records](stride_neighbors_records.json) show:

| K center | Source score | Kernel task | Neighbor mean | Extra time |
| ---: | ---: | ---: | ---: | ---: |
| 24576 | 1345 µs | 1475 µs | 1331 µs | 144 µs |
| 32768 | 1792 µs | 2398 µs | 1762 µs | 636 µs |
| 40960 | 2240 µs | 2401 µs | 2207 µs | 194 µs |
| 49152, N=1024 | 3582 µs | 4729 µs | 3511 µs | 1218 µs |
| 57344 | 3134 µs | 3555 µs | 3078 µs | 477 µs |

All five center pitches are multiples of 32 KiB; the two most severe
are multiples of 64 KiB. The source log has no `CalcTile` update for
the 2560×768/1024 cases, so they use the fallback grouping. In contrast,
the 4096×4096×16384 source log records a 1024×512 L2 region and its
original BASE score is accurate. A's whole-matrix footprint is also
relevant: 1536×768×24576 and 768×2048×32768 have A below the queried
192-MiB L2 capacity and do not show the same spike. These are observed
conditions, not proof of a particular hardware alias rule.

## Opt-in BASE transfer bound

For plain BASE with one output base tile per AIC assignment, issued read
volume is

$$
D_A=b_A K M\left\lceil N/B_N\right\rceil,\qquad
D_B=b_B K N\left\lceil M/B_M\right\rceil.
$$

The experimental objective adds a measured MTE2 service lower bound:

$$
T_B=L+\max\left(T_{\rm Cube},\;T_{\rm existing\ read+packet},\;
\frac{D_A+D_B}{R_{\rm MTE2}(p_A,\mathcal L)}\right)
+T_{\rm call}+T_{\rm atomic},\qquad p_A=b_AK.
$$

[`BaseMte2Service`](../fp32_mode2_v1/remote_selector/model/optimizer.h) is an
optional hardware service surface. The [profile](../fp32_mode2_v1/remote_selector/model/hardware_profile.h)
uses 1.52 MB/µs for 32-KiB-aligned A pitch and 1.26 MB/µs for
64-KiB-aligned pitch, measured in the source kernel. It activates this
bound only when A exceeds 192 MiB and a minimum four-by-four base-tile
L2 region would exceed the source's 100-MiB working-set target, or the
output lacks enough tiles to form it. Otherwise the previous objective
applies. These are access-pattern primitives, not a per-shape runtime
offset. The [source scorer](score_base_mte2.cpp) and [evaluation script](evaluate_transfer_profile.py)
make the opt-in result reproducible.

| Audited set | Baseline MAPE | Opt-in MAPE | Changed scores |
| --- | ---: | ---: | ---: |
| Earlier 81 source BASE cases | 7.45% | 6.26% | 5 |
| Six paired K cases | 9.49% | 1.48% | 2 |
| Nine new K-neighbor cases | 4.01% | 1.73% | 3 |

Within the earlier 81, the 66 wide-N cases improve from 4.16% to
2.71% MAPE; the 15 narrow-N cases remain at 21.92%. The activation
classes and rates were inferred from source-kernel measurements, so they
are provisional in-context transfer services rather than independently
isolated hardware constants.

The [per-case comparison](transfer_profile_comparison.csv) shows remaining
13% underestimation at 1664×768×32768 and 2432×768×32768, and 8.8%
overestimation at 2560×1024×32768 in the earlier set. Several points
used to establish the rate classes also appear in these comparisons;
the table is a diagnostic fit check, **not** held-out validation. The
online FP32 gate remains disabled, and the default hardware profile is
unchanged. The next model step is the source's rotating region assignment
and exposed per-core MTE2/Cube overlap. Narrow N≤128 errors need a
separate padded-output transfer path.

## Selector versus master with the opt-in BASE bound

The standalone selector was rebuilt with
`ExperimentalFp32BaseMte2910B3()` and evaluated on the same 100 FP32
shapes as the [earlier mode-2 pipeline comparison](../fp32_mode2_pipeline_100_v1/README.md).
The updated objective changes seven family decisions, all away from BASE.
For those seven, [measure_new_family.py](measure_new_family.py) ran fresh
clean-master versus forced-family pairs on 910B3 devices 4, 6, and 7.
Each arm had two `msprof` rounds in reversed order, each using five warmups
and five measured kernel runs. The script audits package hashes, source
tiling, correctness, and idle devices. The [paired records](new_family_pairs_records.json)
retain the source tiles and timing rounds. The other 93 unchanged decisions
reuse the previously audited pairs. [analyze_selector_comparison.py](analyze_selector_comparison.py)
recomputes the [100-case table](latest_selector_comparison.csv) and
[summary](latest_selector_analysis.json), rejecting any changed decision
without a fresh pair.

The ratio is master kernel time divided by selected-family kernel time.
Values above 1 favor the selector. The selected family uses MatMulV3's
**source-generated tile**; the model's optimized tile is not injected.
Host-side online scoring time is excluded.

| Set | Shapes | Geometric mean ratio | Selector faster | Master faster |
| --- | ---: | ---: | ---: | ---: |
| All | 100 | **0.984×** | 41 | 59 |
| Family changed from master | 25 | 0.968× | 15 | 10 |
| Family unchanged | 75 | 0.989× | 26 | 49 |

The shape-bootstrap 95% interval for all cases is **0.947–1.012×**,
which includes 1. The previous model was **0.981×** on the same set,
with 38 selector-faster cases. The revised BASE bound has a small net
effect on this comparison and does not establish an overall speedup.

| Shape | Master family | New family | Master µs | New family µs | Ratio |
| --- | --- | --- | ---: | ---: | ---: |
| 2560×768×32768 | deterministic | mode 2 | 2035.8 | 4656.9 | 0.437× |
| 2560×1024×32768 | deterministic | mode 2 | 2707.0 | 3343.3 | 0.810× |
| 2560×1024×49152 | deterministic | deterministic | 3910.5 | 3969.9 | 0.985× |
| 2048×1536×32768 | mode 2 | mode 2 | 3274.3 | 3271.9 | 1.001× |
| 8192×4096×32768 | mode 2 | mode 2 | 33932.9 | 33900.7 | 1.001× |
| 1664×768×32768 | deterministic | deterministic | 1336.5 | 1335.1 | 1.001× |
| 2432×768×32768 | deterministic | deterministic | 1914.6 | 1920.0 | 0.997× |

Five changes now match master's family, removing prior BASE regressions.
The other two switch to mode 2 while master uses deterministic Split-K;
the 2560×768×32768 case is a severe regression. Its selected candidate
scores 1885.9 µs but the source-generated mode-2 kernel takes 4656.9 µs.
The remaining 16×16×65408 deterministic-to-BASE failure is unchanged.
These errors make a source-tile-aware family comparison the next priority.
This is an opt-in experimental result; the FP32 online product gate is still
disabled. The rate classes were also inferred from part of this fixed set,
so a held-out comparison is needed before enabling the profile.

### Why the seven choices changed

All seven previously selected BASE. The table separates the optimized
candidate scored by the selector from the tile that family forcing caused
MatMulV3 to emit. The two latter timings are paired kernel measurements;
"previous BASE" is the earlier audited run and is shown for context rather
than as part of the new pair.

| Shape | New family | Optimized `singleCoreM×N×K` | Emitted `singleCoreM×N×K` | Model µs | Previous BASE µs | New family µs | Master µs |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 2560×768×32768 | mode 2 | 128×768×256 | 128×256×256 | 1885.9 | 2367.4 | 4656.9 | 2035.8 |
| 2560×1024×32768 | mode 2 | 128×1024×256 | 256×512×256 | 2482.4 | 2940.4 | 3343.3 | 2707.0 |
| 2560×1024×49152 | deterministic | 2560×384×192 | 2560×384×192 | 3662.6 | 4934.6 | 3969.9 | 3910.5 |
| 2048×1536×32768 | mode 2 | 208×768×256 | 208×768×256 | 3004.4 | 3934.4 | 3271.9 | 3274.3 |
| 8192×4096×32768 | mode 2 | 1640×1024×192 | 824×2048×192 | 31206.3 | 37681.9 | 33900.7 | 33932.9 |
| 1664×768×32768 | deterministic | 1664×384×192 | 1664×384×192 | 1254.4 | 1810.7 | 1335.1 | 1336.5 |
| 2432×768×32768 | deterministic | 2432×384×192 | 2432×384×192 | 1820.8 | 2634.9 | 1920.0 | 1914.6 |

The first two emitted mode-2 tiles would fail the source's ordinary
`CheckSingleCoreSplitKEdgeCases` gate: `singleCoreN=256` is below 512;
for N=1024, `singleCoreN=512` is at most 640. The forced-family route
bypasses that family decision gate and emits a different tile from the
candidate the optimizer scored. [`score_changed_source_tiles.cpp`](score_changed_source_tiles.cpp)
also finds the standalone plan finalizer rejects these two emitted tiles
under its mirrored spatial gate. The model score therefore cannot be
interpreted as an estimate for the measured source tile. The exact share
of the runtime gap due to tiling versus the mode-2 timing formula has not
been isolated.

For the three deterministic changes and the 2048×1536×32768 mode-2
change, the optimized and source tiles match. The 8192×4096×32768
mode-2 change has an orientation mismatch but is essentially tied with
master. Thus family-only scoring still needs source-emitted tile awareness,
especially at the deterministic/mode-2 boundary.

### Exact model-tile device experiment

To distinguish the tile mismatch from the mode-2 timing error,
[`enable_exact_mode2_tile.py`](enable_exact_mode2_tile.py) adds a
shape-guarded environment override to the isolated research host tiler.
[`build_exact_mode2_tile.sh`](build_exact_mode2_tile.sh) builds a separate
operator package and restores the research source byte-for-byte. The
production master and previous forced-family packages are unchanged.

The [measurement script](measure_exact_mode2_tile.py) compares clean
master, source-generated forced mode 2, and mode 2 with the **exact
optimized model tile** on 910B3 devices 6 and 7. Audits verified all tile
fields, mode-2 key, 20 launch blocks, workspace, source provenance, and
zero output error. Each arm had two `msprof` rounds in reversed order,
five warmups and five measured kernel tasks per round. The reported time
is the median of the two round medians. See [raw paired records](exact_mode2_tile_records.json)
and [summary](exact_mode2_tile_summary.json).

| Shape | Exact `singleCoreM×N×K` | Model score µs | Source mode 2 µs | Exact tile µs | Master µs | Exact / score | Exact / master |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2560×768×32768 | 128×768×256 | 1885.9 | 4649.5 | **2703.9** | **2034.4** | 1.434× | 1.329× |
| 2560×1024×32768 | 128×1024×256 | 2482.4 | 3350.8 | **3277.7** | **2710.9** | 1.320× | 1.209× |

The first exact tile reduces source mode-2 time by **41.8%**; the second
reduces it by **2.2%**. Both are slower than master's deterministic path,
and the mode-2 objective remains **32–43% optimistic** even when its
chosen tile is executed exactly. Thus tile mismatch explains part of the
first regression, but the timing formulation also misses device service
or schedule costs. The opt-in FP32 mode-2 pipeline uses nominal L1→L0
throughput ceilings, not calibrated in-context mode-2 transfer rates;
this is a candidate cause rather than an isolated diagnosis. The product
FP32 online gate remains disabled.

The wrong family ranking is **not** caused by deterministic Split-K being
overestimated on these shapes. Scoring the audited master's exact
`2560×384×192` deterministic tile with the same profile gives
1915.2 and 2544.2 µs. The paired master measurements are 2034.4 and
2710.9 µs, respectively: deterministic is **6.2–6.6% underestimated**.
The mode-2 scores are 1885.9 and 2482.4 µs, only 29.3 and 61.8 µs
below the deterministic scores, while the exact mode-2 kernels run
669.5 and 566.8 µs slower than master. The dominant ranking error is
the mode-2 objective's much larger **32–43% underestimation**.

The [mode-2 stage diagnosis](MODE2_GAP_DIAGNOSIS.md) compares the exact
tiles' MAC, MTE2, and Fixpipe counters with the slice formula and the
earlier 32-case validation sweep. It identifies the one-base-row
MTE2-dominant regime and records the still-pending K-neighbor control.
