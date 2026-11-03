# Split-K family selection experiment on 910B3

## Question and scope

Does the current `SolveReducedMip` objective select a faster MatMulV3
implementation family than the source host tiler's default decision when
choosing between single-core Split-K (modes 2/5/6) and deterministic
multi-core Split-K (mode 3)? This experiment measures the **family
decision**. Each family runs with the tile emitted by its source host
branch; it does not claim that an arbitrary MIP tile was executed.
In this suite, the single-core branch emits mode 6 and K stays below
65535, so the result does not cover mode 2/5 or long-K ND→NZ conversion.

## Frozen protocol

- Hardware: device 4, Ascend 910B3, CANN 9.0.0, 20 AICs. Operator source
  commit `b2cdb471de09d93eb79b34812c1ebe4e938ae10c`.
- Workload: FP16 `A[M,K] * B[K,N]`, ND/ND/ND, NN, no bias, exact `mod4`
  input pattern. The benchmark checks every output element against its
  reference and rejects relative error above 2%.
- Predeclared shapes, before timed runs: `(1792,768)` and `(2048,768)`
  at `K=27392,32768` (four L2-resident operand sets), plus
  `(2048,2048)` and `(2048,3072)` at the same K values (four operand
  sets exceeding L2). All eight passed correctness and emitted the
  intended mode in the independent device-4 audit. Every audited tile,
  key, block count, and workspace matched the earlier device-0 audit.
- Three logical arms: default source selection, source single-core
  Split-K, and source deterministic Split-K. The isolated host library
  accepts `MMV3_RESEARCH_SPLITK_FAMILY=single|deterministic`. It invokes
  the original branch and bypasses only its shape/performance selection
  gate. Capacity/platform checks, branch tile calculation, key emission,
  kernel launch, and output verification remain active. The regular
  source checkout and package are not changed.
- The audit confirmed that default and its matching forced arm emit
  identical key, block count, M/N/K tiles, steps, and conversion flags
  on all eight shapes. Timed runs therefore measure default and the
  **opposite** family. The matching arm inherits the default time.
- Two randomly ordered rounds per shape/arm. Each run uses 10 warmup
  launches and the median of eight `msprof` `MIX_AIC` MatMulV3 task
  times. Every timed run repeats correctness/tiling verification; source
  and geometry must match the audit. `npu-smi` snapshots before and after
  each run must show device 4 idle; a busy snapshot flags the record and
  stops the driver. Arm order is randomized with seed
  20261030. The primary statistic is the geometric mean of the
  shape-level speedup `T_default/T_MIP`, with a 95% bootstrap interval
  resampling the four `(M,N)` clusters (both K variants stay together).
  Also report decision accuracy against the faster of the
  two executed families and median regret. These eight shapes are a
  designed suite, not a random workload population.

## Frozen solver decision

[`splitk_family_model.cpp`](splitk_family_model.cpp) calls the existing
`SolveReducedMip` on each shape with the earlier exploratory BASE,
Split-K, and FP16 single-core primitive values. The workspace limit is
512 MiB to encompass the observed source workspaces. The decision is
the minimum reduced objective over deterministic Split-K and all
single-core Split-K modes; BASE is excluded from this two-family
question. Predictions were written to
[`splitk_family_model_predictions.csv`](splitk_family_model_predictions.csv)
before device timing. The model selects deterministic on all eight
shapes. The above-L2 cases test extrapolation: the optional mode-6
copy rates were measured only for L2-resident data, so the wide-case
objective is outside that primitive's validated memory regime.
The linked driver has since gained the measured mode-6 steady-output
primitive; the CSV here remains the **pre-change frozen result**.
[`diagnose_splitk_family_timing.py`](diagnose_splitk_family_timing.py)
reproduces the old source-geometry objective decomposition.
For diagnosis, [`splitk_family_source_geometry_scores.csv`](splitk_family_source_geometry_scores.csv)
recomputes the same reduced objective at each audited source geometry.
The narrow forced single-core tiles fail the model's spatial host gate,
which the research arm bypasses. On wide shapes both source geometries
pass the model preflight, allowing a like-for-like objective ranking.

## Reproduce and inspect

The isolated remote package is
`/home/xianfei/solvebase_bench_20261025/ops_nn_splitk_family_package`.
The remote experiment artifacts are under
`/home/xianfei/solvebase_bench_20261025/cases/splitk_family_selection_device4_v1`.
[`enable_splitk_family_probe.py`](enable_splitk_family_probe.py) applies
the source probe to an isolated checkout;
[`splitk_family_experiment.py`](splitk_family_experiment.py) audits and
times the branches. On the 910 machine, with CANN's `set_env.sh` sourced:

```bash
python3 splitk_family_experiment.py audit --device 4 \
  --experiment-dir /home/xianfei/solvebase_bench_20261025/cases/splitk_family_selection_device4_v1
python3 splitk_family_experiment.py timing --device 4 \
  --experiment-dir /home/xianfei/solvebase_bench_20261025/cases/splitk_family_selection_device4_v1 \
  --rounds 2 --repetitions 8
```

Use [`analyze_splitk_family_experiment.py`](analyze_splitk_family_experiment.py)
with the experiment directory and frozen CSV after timing completes. It
also recomputes the same summary locally from the compact records and
audit files, without the remote profiler directories:

```bash
python3 analyze_splitk_family_experiment.py \
  splitk_family_selection_device4_v1 splitk_family_model_predictions.csv
```

## Results

The device-4 run completed **32 timed tasks**: eight shapes, two arms per
shape, two randomized rounds. Every task passed the full output check
(maximum relative error ≤0.000445), mode/tiling/source verification, and
the before/after device-4 occupancy checks. Each task has eight
`msprof` kernel times. The table uses the median of the two task medians
per arm, in microseconds. `D` means deterministic mode 3; `S` means
single-core mode 6. The solver chose `D` on all eight shapes.

| M × N × K | Default | S (µs) | D (µs) | Faster |
| --- | :---: | ---: | ---: | :---: |
| 1792 × 768 × 27392 | D | 507.2 | 388.5 | D |
| 2048 × 768 × 27392 | D | 511.1 | 464.8 | D |
| 1792 × 768 × 32768 | D | 642.7 | 574.0 | D |
| 2048 × 768 × 32768 | D | 654.2 | 747.2 | S |
| 2048 × 2048 × 27392 | S | 1072.6 | 1251.8 | S |
| 2048 × 3072 × 27392 | S | 1596.9 | 1819.3 | S |
| 2048 × 2048 × 32768 | S | 1449.7 | 1717.5 | S |
| 2048 × 3072 × 32768 | S | 2091.1 | 2460.3 | S |

The MIP decision matches the faster executed family on **3/8** shapes;
the source default matches on **7/8**. The MIP choice has a geometric
mean `T_default/T_MIP` of **0.9258**, equivalent to **8.0% more runtime**
than default on this suite. A cluster bootstrap over the four `(M,N)`
pairs gives a 95% interval of **[0.8571, 1.0000]** for that speedup; it
does not support an improvement claim. On the four above-L2 shapes,
the MIP choice is slower by **13.9–18.5%** per shape. On the four
below-L2 shapes, it makes the same choice as default and therefore has
identical runtime; both selectors miss the 2048 × 768 × 32768 case,
where source mode 6 is 12.4% faster than default mode 3.

The ranking error is present even when the model scores the exact
executed source geometry. For 2048 × 2048 × 32768, the reduced
objective predicts 2972.5 µs for mode 6 and 2026.9 µs for mode 3,
while device 4 measures 1449.7 and 1717.5 µs respectively. Thus the
current objective overestimates the source mode-6 path on this case;
its MIP tile search is not the sole explanation. The optional mode-6
copy coefficients were measured on L2-resident data, and this shape's
operands exceed L2. The broader event recurrence and overlap assumptions
also need validation before using the family decision in production.
The [timing diagnosis](SPLITK_FAMILY_TIMING_DIAGNOSIS.md) decomposes the
source-tile objectives and identifies the FP32 output-service term and
copy overlap as the leading mode-6 assumptions to validate.
The [provisional mode-6 output-cost change](MODE6_OUTPUT_COST_CHANGE_910B3.md)
records revised source-tile estimates; this section retains the frozen
pre-change predictions and experimental result.
The later [source-in-context output measurement](MODE6_OUTPUT_MEASUREMENT_910B3.md)
documents the device-4 Fixpipe parameter and its larger-shape check.

The local evidence is the [device-4 audit](splitk_family_selection_device4_v1/audit),
[summary](splitk_family_selection_device4_v1/summary.json), and
[32 compact timing records](splitk_family_selection_device4_v1/compact_cases.jsonl).
The full `msprof` directories remain in the remote experiment folder.
[`export_splitk_family_cases.py`](export_splitk_family_cases.py) checks
each occupancy snapshot, output result, kernel count, and mode before
writing the compact records. The suite covers only FP16 NN mode 6
versus mode 3 at K below 65535 and was selected to exercise both default
families; it is not a random workload sample.

During a discarded device-0 pilot, an unrelated `spmm_csr_aicore`
process started on device 0 at 11:18:47 local time on 2026-10-30,
using approximately 10 GiB of HBM. One later mode-6 task jumped from
about 1083 to 2839 µs. The driver was stopped and
[`quarantine_splitk_contention.py`](quarantine_splitk_contention.py)
moved overlapping cases into the remote
`quarantine_device_contention_20261030` folder. No device-0 timing enters
the device-4 analysis.

The later [held-out measured-output experiment](splitk_family_heldout_device4_v2/README.md)
uses the measured mode-6 FP32 output primitive and 16 new shapes.
It finds a 1.0336× geometric mean speedup over default, with a 95%
shape-bootstrap interval of 0.9535–1.1274×. That interval includes a
tie, so the new objective has not yet established a reliable overall
runtime advantage.
