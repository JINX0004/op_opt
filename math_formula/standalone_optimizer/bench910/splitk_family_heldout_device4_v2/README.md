# Held-out measured-output Split-K family experiment

## Question

Does the measured-output version of the reduced MIP objective choose a
faster MatMulV3 Split-K implementation family than MatMulV3's default
host tiler on new FP16 NN shapes? The choice is between single-core
Split-K and deterministic multi-core Split-K. This experiment evaluates
**family selection at the tile emitted by each source branch**; it does
not execute MIP-selected tiles.

## Predeclared protocol

- Hardware: Ascend 910B3, device 4, CANN 9.0.0, 20 cube cores.
- Workload: FP16 `A[M,K] * B[K,N]`, ND/ND/ND, NN, no bias, exact
  `mod4` input pattern; full-output correctness check with maximum
  relative error at most 2%.
- The 16 shapes, two rounds, eight profiler repetitions per run, ten
  warmups, and random seed 20261031 are in [manifest.json](manifest.json).
  None of these shapes appeared in the preceding eight-shape family
  selection experiment or the one-shape source-in-context output
  primitive calibration.
- [model_predictions.csv](model_predictions.csv) was generated and
  frozen **before** the first source audit or timed run on this suite.
  [model_predictions_metadata.json](model_predictions_metadata.json)
  records the freeze timestamp and hashes of the manifest,
  predictions, and C++ model sources. The model includes the measured
  91,000 B/µs/AIC mode-6 FP32 steady-output service parameter.
- Audit the default, forced single-core, and forced deterministic
  source branches independently. Include a shape only if all three
  pass correctness and emit the expected Split-K modes. The forced
  branch enters the corresponding original source tiling function;
  capacity checks, tiling, key, workspace, kernel launch, and output
  verification still use the source implementation.
- If the default and matching forced branch emit identical tile, key,
  block count, and workspace, time the default and opposite branch.
  Otherwise time all three. Each timed run has before/after device-4
  occupancy checks, a fresh correctness check, and an `msprof` median
  of eight `MIX_AIC` kernel task durations.
- Primary comparison: geometric mean of `T_default / T_model` over
  eligible shapes. Report a 95% bootstrap interval resampling `(M,N)`
  pairs and the count of choices matching the faster measured source
  family. This designed suite is not a random production workload sample.

The remote full profiler records live under
`/home/xianfei/solvebase_bench_20261025/cases/splitk_family_heldout_device4_v2`.
The [holdout driver](holdout_experiment.py) consumes the frozen files;
the [analyzer](analyze.py) uses compact verified records exported by
[`export_splitk_family_cases.py`](../export_splitk_family_cases.py).

## Results

All 16 declared shapes passed the 48 source audits. In every audit, the
forced single-core branch emitted mode 6, the forced deterministic branch
emitted mode 3, and default emitted one of those two modes. Default and
the matching forced arm emitted identical keys, tile parameters, block
counts, and workspaces, so each shape needed only default and the
opposite family in each round. The 64 timed tasks passed the full-output,
source-path, mode, geometry, kernel-count, and before/after device-4
occupancy checks. The maximum reported relative output error was zero.

`S` means single-core mode 6 and `D` means deterministic mode 3. The
time for the default family is its measured default run; the other
family is its measured forced run. Times below are medians of the two
per-run `msprof` medians, in microseconds. A speedup above 1 favors the
model; 1 means both selectors chose the same source branch.

| M × N × K | Default | Model | S (µs) | D (µs) | Default/model |
| --- | :---: | :---: | ---: | ---: | ---: |
| 1792 × 896 × 32768 | D | D | 715.4 | 685.6 | 1.000 |
| 2048 × 896 × 32768 | D | D | 955.1 | 842.0 | 1.000 |
| 2304 × 768 × 32768 | D | S | 738.4 | 898.8 | 1.217 |
| 2560 × 768 × 40960 | D | S | 1319.2 | 901.1 | 0.683 |
| 1792 × 1024 × 40960 | D | D | 728.6 | 808.4 | 1.000 |
| 2048 × 1024 × 40960 | D | D | 839.0 | 955.9 | 1.000 |
| 2304 × 1024 × 32768 | D | D | 661.1 | 1089.2 | 1.000 |
| 2560 × 1024 × 32768 | D | S | 797.3 | 1319.8 | 1.655 |
| 1536 × 1536 × 40960 | D | S | 850.6 | 1049.5 | 1.234 |
| 1792 × 1792 × 32768 | S | S | 997.7 | 1258.9 | 1.000 |
| 2304 × 1536 × 36864 | S | S | 1129.8 | 1348.4 | 1.000 |
| 2560 × 2048 × 32768 | S | S | 1638.0 | 2848.7 | 1.000 |
| 3072 × 2048 × 40960 | S | S | 2582.5 | 2829.5 | 1.000 |
| 2048 × 2560 × 36864 | S | S | 1826.5 | 2120.9 | 1.000 |
| 2560 × 3072 × 40960 | S | S | 2524.7 | 3482.7 | 1.000 |
| 3072 × 3584 × 32768 | S | S | 3122.6 | 4288.1 | 1.000 |

The measured-output model selects the faster source family on **12/16**
shapes; the MatMulV3 default does so on **10/16**. The model differs
from default on four shapes, winning three and losing one. Its geometric
mean speedup over default across all 16 shapes is **1.0336×** (about
3.3% less runtime). The 95% shape-bootstrap interval is **0.9535–1.1274×**;
it includes 1. Consequently these 16 shapes do **not** establish a
statistically reliable runtime improvement over default. The suite was
deliberately chosen to probe the family boundary and does not imply a
population-wide gain.

The loss at 2560 × 768 × 40960 is material: model-selected mode 6
takes 1319.2 µs versus 901.1 µs for default mode 3. Conversely, the
model finds substantial wins at 2304 × 768 × 32768,
2560 × 1024 × 32768, and 1536 × 1536 × 40960. It also misses three
large mode-6 wins where it agrees with default (the 1792 × 1024,
2048 × 1024, and 2304 × 1024 rows). On the first two of those, the
model reports no feasible single-core solution even though the forced
source branch executes correctly. This is a decision-domain mismatch,
not a timing-calibration issue alone. The remaining misses point to the
mode-6 versus mode-3 objective near the family boundary.

The full [summary](summary.json), [verified compact timing records](compact_cases.jsonl),
[source audits](audit), and [eligibility record](eligibility.json) are
saved here. The old eight-shape suite helped choose the new output
primitive, so it is not pooled into this held-out confidence interval.
