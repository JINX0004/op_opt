# FP32 100-case comparison with the opt-in mode-2 pipeline score

This evaluates the experimental FP32 mode-2 pipeline objective on the same
[100 shapes](../fp32_100_rerun_20261106/dataset.json) as the prior
[paired master comparison](../fp32_100_rerun_20261106/README.md). The scorer
uses `ExperimentalFp32Mode2Pipeline910B3()` and
`SourceCompatibleDomain()` from selector branch
`codex/matmul-v3-mip-selector` at `f765fa27`. Its model source hashes were
checked against that remote revision before evaluating the 100 cases. The
FP32 product online gate remains disabled.

The model chooses a family using its optimized candidate tile. The measured
MatMulV3 run uses **source-generated tiling within that family**; the model's
proposed tile is not injected. These are kernel times, excluding host scoring.

## Method

- [pipeline_decisions.txt](pipeline_decisions.txt) records every new model
  decision and candidate score. The decision count is 84 BASE, 14
  deterministic Split-K, and 2 single-core Split-K mode 2. The earlier
  model chose 81, 13, and 6 respectively. Only cases 17, 18, 20, and 42
  change family.
- For the 96 unchanged family decisions, this analysis reuses the audited
  paired master/proposed times from the [2026-11-06 full rerun](../fp32_100_rerun_20261106/records.json).
  Reusing these measurements preserves the exact selected source path.
- The four changed decisions received fresh clean-master versus forced-family
  pairs on 910B3 devices 4–6. Each received two `msprof` rounds with
  reversed arm order; each round reports the median of five kernel runs
  after five warmups. [measure_changed.py](measure_changed.py) checks the
  clean-master/unforced-research tile match, family provenance, output
  relative error at most 2%, and idle devices. The
  [audited records](changed_records.json) contain emitted tiles, timing
  rounds, and output checks. Raw traces are at
  `/home/xianfei/solvebase_bench_20261025/cases/fp32_mode2_pipeline_100_v1`
  on 910.
- [analyze.py](analyze.py) combines those measurements and recomputes the
  tables below. It explicitly rejects an unmeasured changed decision.

The ratio is **master kernel time / selected-family kernel time**. A value
above 1 favors the model selection.

| Set | Cases | Geometric mean ratio | Model faster | Master faster |
| --- | ---: | ---: | ---: | ---: |
| All | 100 | **0.981×** | 38 | 62 |
| Family changed | 30 | 0.963× | 15 | 15 |
| Family unchanged | 70 | 0.988× | 23 | 47 |

The all-case shape-bootstrap 95% interval is **0.947–1.006×**. The interval
includes 1, so this fixed stress set does not show a reliable overall
advantage. Its unchanged-family control is also 0.988× despite identical
selected families and source tiles; timing or package variation contributes
to the apparent overall difference. The previous full-run ratio was 0.982×.

## Four new family decisions

| Shape | Master → model family | Master µs | Model-family µs | Ratio |
| --- | --- | ---: | ---: | ---: |
| 1024×2304×36864 | mode 2 → deterministic | 2965.8 | 2705.7 | 1.096× |
| 1024×2816×36864 | mode 2 → BASE | 3471.3 | 3285.3 | 1.057× |
| 2048×1536×32768 | mode 2 → BASE | 3284.4 | 3934.4 | 0.835× |
| 8192×4096×32768 | mode 2 → BASE | 33904.4 | 37681.9 | 0.900× |

The mode-2 objective improvement fixes some local decisions, but the two
BASE choices above are regressions. For 2048×1536×32768, the optimizer
scores hypothetical BASE at 2985.3 µs and mode 2 at 3004.4 µs, a narrow
predicted BASE lead; source-generated BASE instead takes 3934.4 µs and
source mode 2 takes 3284.4 µs. The source-tile mismatch remains important
for family selection.

The earlier **16×16×65408** failure is unchanged: master deterministic
Split-K takes 39.56 µs, and forced BASE takes 146.34 µs. Thus the new
mode-2 score does not address the main deterministic-to-BASE boundary error.

See [all 100 cases](comparison.csv), [changed-family cases](family_changed.csv),
[unchanged-family cases](family_unchanged.csv), and the
[aggregate analysis](analysis.json). This is an opt-in model comparison,
not a deployed online selector comparison. A final production evaluation
would need source-tile-aware scoring, a new held-out shape set, and
end-to-end timing including online scoring cost.

## Which estimator causes deterministic-to-BASE errors?

The [source-tile diagnostic](source_base_det_diagnosis.csv) separately
scores the exact BASE and deterministic tiles audited in all 21
deterministic-to-BASE decisions. Run
`diagnose_base_det.py /path/to/score_source_tiles` after compiling
[score_source_tiles.cpp](score_source_tiles.cpp) with the selector model.
The deterministic source tile has the same reduced score as the
optimized deterministic candidate to within printed precision. BASE
often differs because its optimized candidate score does not use the
source-emitted L1 K steps.

| Family, 21 cases | Mean score bias vs measured | Mean absolute percentage error |
| --- | ---: | ---: |
| Deterministic Split-K | −1.0% | 9.9% |
| Source-tile BASE | **−15.7%** | **16.7%** |

The source-tile scores rank the faster family correctly in **11/21**
cases, compared with 8/21 for the current optimized-family choice.
The remaining wrong rankings are mostly BASE underestimation:

| Shape | Det. score / measured µs | BASE source score / measured µs |
| --- | ---: | ---: |
| 1664×768×32768 | 1254 / 1338 | **1196 / 1811** |
| 2432×768×32768 | 1821 / 1923 | **1792 / 2635** |
| 2560×1024×49152 | 3663 / 3929 | **3582 / 4935** |
| 4096×64×27456 | 378 / 488 | **289 / 540** |

The 16×16×65408 failure has a separate candidate mismatch. Its
source-tile BASE score is **147.76 µs**, close to **146.34 µs** measured;
the selected hypothetical BASE score is only **33.16 µs**. The
deterministic score is **41.36 µs**, close to **39.56 µs** measured. Using
the emitted BASE tile would correctly choose deterministic there.

In the reduced BASE formula, `BaseStageUs` takes the maximum of Cube
service and shared byte service plus packet startups. The
`base_fp32_packet_issue` coefficient was calibrated on the 16×16 source
tile; the optional FP32 call and atomic-output terms are inactive in
this hardware profile. The larger-tile errors show that its current
service and overlap assumptions do not transfer to those geometries.
The existing paired timings identify the bias, but do not isolate which
device stage or dependency causes it. Deterministic Split-K also has
errors, especially some narrow-N cases, so a BASE-only correction is
not sufficient for every shape.
