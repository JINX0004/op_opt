# Deterministic Split-K per-panel measurement on 910B3

This experiment tests the large-shape FP32 ranking error at
`8192×4096×32768`. The source's forced deterministic family emits the
same core `2816×384×192`, base `128×128×64`, step `(1,3,3)`, 20 AICs,
and 185-MiB workspace for every tested N from 384 to 4096. All 11
[tiling and output audits](audits.json) passed. K and M stay fixed;
N=384,768,…,3840 gives one through ten full 384-column outer panels.
N=4096 adds a 256-column tail panel.

[run.py](run.py) used the earlier forced-family host package (SHA-256
`cd074481e4819bffc4e8d85d1d42e9c889d670dc3ecb213264cecb83ebbe5ad3`)
on device 5. The forward and reverse sweeps each used five warmups and
five timed `msprof` tasks per shape with `PipeUtilization`; all outputs
passed. The 11-panel runtime, 36.418 ms, agrees with the latest online
package's earlier paired 36.407 ms. [Raw panel records](panel_records.json),
[model scores](model_scores.txt), [analysis](analyze.py), and the
[machine-readable summary](summary.json) are retained.

| N | Panels | Measured µs | Model µs | Missing µs | MTE2 busy µs | Fixpipe busy µs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 384 | 1 | 3318.6 | 3048.7 | 269.9 | 1642 | 1546 |
| 768 | 2 | 6811.6 | 6068.6 | 743.1 | 3632 | 3441 |
| 1152 | 3 | 10508.7 | 9088.5 | 1420.2 | 6068 | 5929 |
| 1536 | 4 | 13635.9 | 12108.4 | 1527.5 | 7454 | 7114 |
| 1920 | 5 | 16959.4 | 15128.3 | 1831.1 | 9191 | 8890 |
| 2304 | 6 | 20418.0 | 18148.2 | 2269.8 | 11238 | 10726 |
| 2688 | 7 | 23782.2 | 21168.0 | 2614.1 | 13055 | 12530 |
| 3072 | 8 | 27159.9 | 24187.9 | 2971.9 | 14812 | 14173 |
| 3456 | 9 | 30409.9 | 27207.8 | 3202.1 | 16456 | 15852 |
| 3840 | 10 | 34165.3 | 30227.7 | 3937.6 | 19035 | 18372 |
| 4096 | 11, last partial | 36418.3 | 32240.5 | 4177.8 | 20544 | 19312 |

Medians combine the two order-reversed runs. The largest difference
between those two runs at any N is 140 µs; at N=4096 it is 19 µs.
For one through ten full panels, ordinary least squares gives a
**3393.7-µs measured increment per panel** (rounds separately: 3390.2
and 3397.3 µs; $R^2=0.99983$ for the median series). `SplitStageUs`
predicts exactly **3019.9 µs per full panel**. Its **373.8-µs per-panel
shortfall** explains roughly 4.0 ms of the 4.18-ms error at N=4096.
The measured MAC, MTE2, Fixpipe, and scalar busy-time slopes are
2804, 1872, 1804, and 842 µs per full panel respectively. These
pipelines overlap; their busy times cannot be added to make a runtime.

## Requested traffic and cache behavior

[MemoryAccess and L2Cache profiles](memory_records.json) used the same
forced deterministic family at one, ten, and eleven panels. Values
below are profiler-requested KiB converted to GiB. They are not
external-DRAM bytes.

| N | GM→L1 GiB | L0C→GM GiB | L2 read hit fraction |
| ---: | ---: | ---: | ---: |
| 384 | 1.141 | 2.004 | 9.02% |
| 3840 | 11.407 | 20.039 | 9.85% |
| 4096 | 12.500 | 21.375 | 10.06% |

Requested traffic scales roughly with panel count; the L2 read-hit
fraction does not collapse. A separate [mode-2 MemoryAccess profile](mode2_memory_record.json)
at N=4096 reports **17.000 GiB GM→L1**, 36% more than deterministic,
and exactly the same **21.375 GiB L0C→GM**, yet mode 2 is faster.
Data volume by itself therefore does not explain the deterministic
penalty. Its access pattern, stage service, and exposed waits matter.

The source deterministic loop writes partials to workspace on each K
slice. For each outer panel it signals vector reduction, and from the
third panel waits for the ping-pong buffer before reuse. It also uses
`SyncAll` and full pipe barriers. The reduced model currently treats
each full panel as Cube-bound: **3019.9 µs Cube** versus **2873.0 µs
input service**, with only **26.1 µs** modeled vector reduction. It
assumes nearly complete overlap of read, Cube, and preceding-panel
reduction. The measured panel increment is 12.4% longer. The experiment
identifies the missing *exposed per-panel service*; the pipeline
counters do not isolate its exact division among MTE2, Fixpipe,
workspace reduction, and synchronization. A uniform 373.8-µs penalty
should not be applied to other geometries without separate validation.

## Retired workspace timeline trial

The source hands off each inner M block separately in this L2-split NK
case. The prototype therefore schedules 33 handoffs at N=4096. Its
logical GM→L1 volume is 12.500 GiB, within 320 KiB of the requested
traffic counter, and its 21.375 GiB Fixpipe volume matches the counter.
These checks validate the traffic accounting for this tile. At N=4096
the recurrence predicts zero workspace-slot stalls with the current
service proxies, so the 1.91-ms residual cannot be assigned to those
stalls from this result.

| Validation set | Tiles | Legacy mean absolute relative error | Timeline error |
| --- | ---: | ---: | ---: |
| Fixed-tile N sweep | 11 | 11.01% | 4.57% |
| Wider paired comparison | 44 | 10.42% | 19.26% |
| Wider comparison, one handoff | 24 | 14.08% | 30.29% |
| Wider comparison, multiple handoffs | 20 | 6.02% | 6.04% |

The new slope is 3.227 ms per full outer panel versus 3.020 ms before
and 3.394 ms measured. At N=4096 the deterministic score rises from
32.241 to 34.504 ms, above mode 2's existing 32.295-ms score. That
matches the measured family order. The two other mode-2 ranking
regressions, `2048×1024×32512` and `2304×1024×36864`, still rank mode 2
first under the new deterministic scores. The larger error on
one-handoff cases warrants calibrating the Cube-call and vector-tail
services on representative small and narrow tiles before enabling the
new score for family selection.

The [100-shape selector test](../fp32_det_panel_selector_v1/README.md)
led to retiring the scorer. The historical [per-arm results](timeline_comparison.csv)
contain scores, modeled stage demand, exposed buffer waits, and logical traffic.
