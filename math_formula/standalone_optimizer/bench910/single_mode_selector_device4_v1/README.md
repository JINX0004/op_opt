# Measured-output MIP for single-core Split-K mode selection

## Question

Would replacing MatMulV3's source logic among single-core Split-K modes
2, 5, and 6 with the calibrated reduced MIP improve FP16 NN runtime on
910B3? This experiment tests the overlapping **mode-2 versus mode-6**
channel and separately audits the special K=1536 mode-5 channel. It
does not execute MIP-selected tiles; both timed modes use the source
host tiler's own tile.

## Source scope

Mode 2 is the ordinary single-core Split-K kernel. Mode 6 uses the
explicit GM→L1 kernel for aligned FP16/BF16 inputs with no bias and
the relevant inner axes at most 65535. Mode 5 is the NKM kernel on the
K=1536, M-large/N=384 channel. These kernel variants are not freely
interchangeable for every input. In particular, a mode selector runs
*after* the earlier `IsSupportSingleCoreSplitK` and
`CheckSingleCoreSplitKEdgeCases` family/shape gates; replacing only the
2/5/6 choice cannot open a single-core path rejected by those gates.

The [research-only source patch](../enable_single_mode_probe.py) adds
`MMV3_RESEARCH_SC_MODE=2`, which suppresses the final mode-6 conversion
after the original single-core tiler has generated a tile. The unset
path and mode 6 retain source behavior. It was built against the
machine's CANN 9.0.0 in a **separate**
`ops_nn_single_mode_package`; the existing source and build library were
restored to their previous SHA-256 after copying the probe library.

## Frozen protocol

- Hardware: Ascend 910B3 device 4, 20 AICs, CANN 9.0.0.
- Workload: FP16 ND/ND/ND NN, no bias, exact `mod4` input pattern,
  full-output correctness check. The eight shapes, random seed,
  warmups, rounds, and repetitions are in [manifest.json](manifest.json).
- The [model driver](model.cpp) uses the prior 910B3 primitive profile,
  including the measured 91,000 B/µs/AIC mode-6 steady-output service.
  [Predictions](model_predictions.csv) and their
  [source hashes](model_predictions_metadata.json) were frozen before
  the eight-shape audit and timing. It chose mode 6 on all eight.
- The [experiment driver](../single_mode_experiment.py) audits mode 2,
  mode 6, and default independently. All eight passed correctness and
  emitted the requested mode. For each shape, mode 2 and mode 6 had
  identical `singleCoreM/N/K`, base tiles, steps, and block count.
- Two randomized rounds time both modes. Each run has 10 warmups and
  the median of eight `msprof` `MIX_AIC` MatMulV3 task durations.
  Before and after every run, `npu-smi` must show device 4 idle. Each
  run repeats correctness, source, mode, and tile verification.
  [`export_splitk_family_cases.py`](../export_splitk_family_cases.py)
  checked the saved snapshots and profiler records before exporting
  [34 compact records](compact_cases.jsonl): 32 planned mode runs and
  two follow-up default runs on the narrow-N shape.

## Results

All eight mode pairs passed. The maximum reported relative output
error across timed runs was 0.00028. Times are the median of the two
per-run medians, in microseconds. `T₂/T₆ > 1` favors mode 6.

| M × N × K | Default mode | Mode 2 (µs) | Mode 6 (µs) | Measured T₂/T₆ | MIP predicted T₂/T₆ |
| --- | :---: | ---: | ---: | ---: | ---: |
| 1792 × 1792 × 36864 | 6 | 1283.1 | 1112.3 | 1.154 | 2.197 |
| 2048 × 2048 × 40960 | 6 | 1778.1 | 1580.5 | 1.125 | 2.204 |
| 2304 × 2048 × 32768 | 6 | 1465.1 | 1393.3 | 1.052 | 2.213 |
| 2560 × 2560 × 36864 | 6 | 2196.1 | 1953.8 | 1.124 | 2.401 |
| 3072 × 2048 × 32768 | 6 | 2578.4 | 2099.5 | 1.228 | 2.476 |
| 2304 × 3072 × 40960 | 6 | 2482.4 | 2317.6 | 1.071 | 2.312 |
| 2560 × 1024 × 36864 | 3 | 1058.0 | 836.7 | 1.264 | 1.657 |
| 2304 × 2048 × 65408 | 6 | 2813.3 | 2489.4 | 1.130 | 2.233 |

Mode 6 was faster on **8/8** shapes; the geometric mean speedup over
mode 2 was **1.1415×**, with a 95% bootstrap interval of
**1.0975–1.1904×** resampling the eight designed shapes. This
validates the MIP's mode ranking on the tested overlap. Its predicted
ratio has a 2.198× geometric mean, so the **size** of its mode-6
advantage is substantially overstated. The ratio uses each mode's
optimized MIP geometry in the prediction and the common source geometry
in measurement; it is a selection test, not an absolute tile-time
calibration test.

The source default already chose mode 6 on **7/8** shapes. On those
seven, the audited default and forced mode-6 keys, tiles, workspaces,
and block counts match. Replacing the internal mode 2/6 rule with this
MIP selector would make the same choice on all seven. The remaining
shape, 2560 × 1024 × 36864, defaults to deterministic multi-core
Split-K (mode 3). Its two directly timed default runs have a 1022.0 µs
median versus 836.7 µs for forced mode 6, a **1.222× speedup**. That
gain requires changing the earlier **family-selection** gate, not the
choice among modes 2, 5, and 6.

The separate [structural audit](structural/summary.json) passed full
correctness on K=1536: MatMulV3 emits mode 5 at
`49152 × 384 × 1536` and mode 2 at `384 × 49152 × 1536`.
The single-mode MIP reports mode 5 and mode 2 respectively, because
only the matching structural channel is modeled as legal. Mode 5 was
not timed against mode 6; the two are not a supported choice on these
shapes.

## Decision

Do **not** replace the production 2/5/6 selector based on this suite.
The MIP agrees with the source's mode-6 decision wherever the default
actually enters the tested single-core path, and mode 5 serves a
specialized shape channel. The next experiment should relax only the
source's *performance* family gates in the model, preserve capacity and
kernel-key constraints, and compare MIP-selected single-core mode 6
against default deterministic mode 3 on held-out shapes. Any production
selector would also need a source-host integration path and validation
of the MIP-selected tiles, which were not executed here.

The full profiler outputs remain under
`/home/xianfei/solvebase_bench_20261025/cases/single_mode_selector_device4_v1`.
The local [summary](summary.json), [mode audits](audit),
[structural audits](structural/audit), and [analyzer](analyze.py)
reproduce the conclusions above.
