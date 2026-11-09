# FP32 100-case rerun, 2026-11-06

This is a fresh paired device rerun of the [same 100 FP32 shapes](dataset.json)
used in the [earlier comparison](../fp32_100_source_family_v1/README.md).
It evaluates the current experimental selector on 910B3 against clean-master
MatMulV3. The selector chooses an implementation **family**; MatMulV3
generates the block and buffer tiling within that family. The proposed
`singleCoreM/N/K` and `baseM/N/K` in [model_scores.csv](model_scores.csv)
are recorded for diagnosis but are not injected into the timed kernel.

## Provenance and protocol

- Selector branch: `codex/matmul-v3-mip-selector` at `87b7db03`; selector
  binary SHA-256 `da3be68fbd02f6cec69fb1ff89ab12c9b9523ec59209e9624c7b984894954778`.
- Clean-master host library SHA-256
  `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`.
  The isolated family-forcing host library SHA-256 is
  `cd074481e4819bffc4e8d85d1d42e9c889d670dc3ecb213264cecb83ebbe5ad3`;
  it uses clean-master kernel binaries.
- Same-type FP32 A/B/output, NN, ND/ND/ND, no bias, HF32 disabled;
  CANN 9.0.0. Each case's master and proposed runs use the same 910B3
  device, assigned by `index % 4` to devices 4–7.
- [run.py](run.py) reuses the earlier audit and `msprof` protocol with a
  fresh output directory. It checks host-library and selector hashes,
  device idleness, tiling provenance, output error ≤2%, and that the
  unforced research package emits the clean-master tile. BASE (0),
  deterministic Split-K (3), and single-core Split-K (2) are forced
  through their source paths. A changed tiling gets two paired rounds
  with arm order reversed; an unchanged tiling gets one. Each profile
  takes the median of five kernel repetitions after five warmups.
- [analyze.py](analyze.py) reuses the earlier validation and aggregate
  analysis. These are kernel times; online scoring overhead is excluded.
  Raw traces and per-case records are under
  `/home/xianfei/solvebase_bench_20261025/cases/fp32_100_rerun_20261106`
  on 910.

The current [100 selector decisions](model_decisions.json) comprise 81
BASE, 13 deterministic Split-K, and 6 single-core Split-K mode 2. No
case in this fixed dataset selects mode 5, so this rerun cannot evaluate
the latest FP32 mode-5 estimator. The 100 mode decisions, proposed tiles,
and objective values are identical to the earlier
[mode-2 decision set](../fp32_mode2_v1/mode2_decisions.csv).

## Results

All 100 pairs finished with no audit or profiling failure. The ratio is
**clean-master kernel time / proposed-family kernel time**; values above
1 favor the selector. The 95% shape-bootstrap interval describes this
deliberately selected stress set and does not imply a random-workload
population claim.

| Set | Cases | Geometric mean ratio | Proposed faster | Master faster |
| --- | ---: | ---: | ---: | ---: |
| All | 100 | 0.982× | 38 | 62 |
| Family changed | 26 | 0.962× | 13 | 13 |
| Family unchanged, identical tiling | 74 | 0.989× | 25 | 49 |
| Master deterministic Split-K → proposed BASE | 21 | 0.921× | 8 | 13 |
| Master mode 2 → proposed BASE | 4 | 1.131× | 4 | 0 |
| Master mode 2 → proposed deterministic Split-K | 1 | 1.266× | 1 | 0 |

The all-case bootstrap 95% interval is **0.949–1.008×**, which includes
1. The model has no demonstrated overall kernel-time advantage on this
set. The unchanged-family control emits identical tiling and uses the
same kernel binaries, yet has a 0.989× ratio. That roughly 1.1% control
offset is timing or package variation and should not be credited to the
family selector. The previous mode-2 analysis reported 0.981× overall;
the fresh full rerun is consistent with it.

The largest loss remains **16×16×65408**: master deterministic Split-K
39.56 µs versus proposed BASE 146.34 µs, or 0.270×. Two other large
deterministic → BASE losses are 2432×768×32768 (1923.12 versus
2634.85 µs, 0.730×) and 1664×768×32768 (1337.96 versus
1810.66 µs, 0.739×). Among changed families, excluding the 16×16
outlier gives a 1.012× geometric mean over 25 cases, but its separate
bootstrap interval is broad (0.948–1.079×).

There are also meaningful wins: 2816×1024×49152 changes mode 2 to
deterministic Split-K and improves from 5715.53 to 4514.37 µs
(1.266×); 3072×1024×36864 changes mode 2 to BASE and improves from
4093.17 to 3296.40 µs (1.242×); and 512×512×27392 changes
deterministic Split-K to BASE and improves from 265.09 to 220.32 µs
(1.203×).

The [complete comparison](comparison.csv), [changed](family_changed.csv)
and [unchanged](family_unchanged.csv) subsets, [aggregate analysis](analysis.json),
and [portable records](records.json) contain every case's tiling,
correctness check, and paired timing. Recompute locally with
`python3 analyze.py --records records.json` from this directory. The
research package forces only the selected **family**; these timings do
not validate the model's proposed tile or include online scoring time.

The FP32 product gate should remain closed. The priority model issue is
the deterministic Split-K → BASE boundary, especially the 16×16×65408
case and the two 768-wide regressions. This fixed dataset provides no
new validation for the experimental mode-5 score. The
[FP32 BASE `singleCoreK` diagnosis](BASE_SINGLECOREK_DIAGNOSIS.md)
separates the reported full-K ownership field from the kernel's internal
K calls and compares current model scores with measured source tiles.
