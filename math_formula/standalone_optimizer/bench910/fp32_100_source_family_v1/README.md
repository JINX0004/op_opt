# FP32 100-shape source-family comparison on 910B3

This experiment evaluates the revised [FP32 source-informed tiling domain](../fp32_family_boundary_v1/forced_family_v1/FP32_SOURCE_DOMAIN_REVISION.md)
over [100 distinct shapes](dataset.json). Eighty shapes come from the earlier
[FP16 100-case dataset](../online_100_vs_master_fp16_v1/README.md) after
removing FP32 converted-A, alignment, and B-conflict cases that the current
FP32 model does not support. Twenty additional shapes bracket the FP32
K=6144, 7680, and 27392 family boundaries. The selection is deliberately
stressful and is not a random workload sample.

The measured comparison is **clean-master MatMulV3 family selection versus
the model's family selection with source-generated tiling inside that family**.
The product FP32 online gate is still closed. The standalone model's exact
proposed tile is recorded in [model_decisions.json](model_decisions.json) but
is not injected into the kernel. The research host library forces only BASE
or deterministic Split-K; MatMulV3 generates the block and buffer tiling.
This distinction matters when interpreting a family speedup.

## Protocol

- Same-type FP32 A/B/output, NN, ND/ND/ND, no bias, HF32 disabled; CANN 9.0.0.
- 910B3 devices 4–7, with each shape's master/proposed pair on the same
  device (`index % 4`). Device idleness is checked before and after every
  audit and timed run.
- The clean-master package has host-library SHA-256
  `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`.
  The isolated family-forcing package has SHA-256
  `cd074481e4819bffc4e8d85d1d42e9c889d670dc3ecb213264cecb83ebbe5ad3`;
  its kernel binaries are copied from clean master.
- The FP32 model binary SHA-256 is
  `d388c1000441711ea20d16eaba5528f461e8c78069590ef4ab1b9450b613927a`.
  The research source is on `codex/matmul-v3-mip-selector` at `d4373232`
  with the FP32 domain working-tree change.
- [run.py](run.py) audits clean master, unforced research, and forced model
  family. The unforced research tile must match clean master in every
  reported tiling field. Full-output relative error must be at most 0.02.
- `msprof` records five kernel repetitions after five warmups per arm.
  Changed tilings receive two rounds in reversed arm order; unchanged
  tilings receive one round. The result is a kernel-time comparison, without
  host selector/scoring time.
- Raw traces and per-case results are kept under
  `/home/xianfei/solvebase_bench_20261025/cases/fp32_100_source_family_v1`
  on 910. [analyze.py](analyze.py) validates records and produces the compact
  comparisons.

## Results

All 100 shape pairs completed with no failures. Each result passed the
source-tiling and output audits. The ratio below is **master kernel time /
proposed-family kernel time**; values greater than 1 favor the proposed
family. The shape bootstrap interval is descriptive because this is a
deliberately selected stress set, not a random workload sample.

| Set | Cases | Geometric mean ratio | Proposed faster | Master faster |
| --- | ---: | ---: | ---: | ---: |
| All | 100 | 0.979× | 39 | 61 |
| Family changed | 32 | 0.960× | 15 | 17 |
| Family unchanged | 68 | 0.987× | 24 | 44 |
| Master BASE or deterministic Split-K, family changed | 21 | 0.918× | 7 | 14 |
| Master mode 2, outside current model families | 11 | 1.047× | 8 | 3 |

The all-case shape-bootstrap 95% interval is 0.945–1.005×. Thirteen changed
families improved kernel time by more than 5%, while thirteen regressed by
more than 5%; eight regressed by more than 10%. The 68 unchanged-family cases
also retained identical reported tiling and kernel binaries. Their 0.987×
geometric mean and four losses over 5% indicate run-to-run or package timing
variation that should not be attributed to the selector.

The largest regression is **16×16×65408**: master deterministic Split-K took
39.89 µs, while the proposed BASE family took 145.58 µs (0.274×). For
**1664×768×32768** and **2432×768×32768**, proposed BASE was 36% and 39%
slower than master deterministic Split-K, respectively. Excluding the
16×16×65408 outlier, the changed-family geometric mean is 1.000× (31 cases),
so the tested choice still shows no broad advantage.

There are material wins at family boundaries: **512×512×27392** improved
from 275.24 to 220.00 µs (1.251×), **2048×64×27392** from 238.16 to
196.55 µs (1.212×), and **2816×1024×49152** from 5716.10 to 4476.07 µs
(1.277×). These are kernel timings for source-generated tiles within the
forced family, not timings for the standalone model's exact tile.

### Interpreting `singleCoreK`

For BASE, `singleCoreK` is the K span assigned to an output block, not the
L0 K buffer tile. MatMulV3 sets this field to the full problem K in both FP32
and BF16 BASE. The L0 tile is `baseK`, and the L1 A/B transfer spans are
`stepKa × baseK` and `stepKb × baseK`. For example:

| Shape and path | `singleCoreK` | `baseK` | L1 A/B K spans | Kernel K calls |
| --- | ---: | ---: | ---: | ---: |
| FP32 16×16×65408 proposed BASE | 65408 | 32 | 2048 / 2048 | 7 × 8192 + 1 × 8064 |
| BF16 16×16×65408 online BASE | 65408 | 64 | 4096 / 4096 | 1 × 65408 |
| FP32 2560×768×32768 proposed BASE | 32768 | 32 | 256 / 128 | 4 × 8192 |
| BF16 2048×768×27392 online BASE | 27392 | 64 | 512 / 256 | 1 × 27392 |

For this FP32 NN/ND, HF32-disabled path, the kernel subdivides a BASE
`singleCoreK` above 8192 into repeated `SetSingleShape`/`Iterate`/
`GetTensorC` calls, with subsequent output updates accumulated atomically.
BF16 does not take this FP32-specific subdivision. Thus a large reported
`singleCoreK` is source-correct but is misleading if presented as the size
of one hardware tile or one matmul call. The current BASE objective uses
total K and `baseK` but does not explicitly cost these FP32 kernel calls and
their repeated output updates. This is a model limitation to address before
relying on FP32 family decisions. The experiment also injects only the
family: the model's complete tile matches the emitted source tile in just
4 of the 32 changed-family cases, so its objective is not a prediction for
the measured tile in the other 28 cases.

The current FP32 model searches only BASE (mode 0) and deterministic Split-K
(mode 3). Master chose mode 2 in eleven cases; those comparisons are reported
separately because the model cannot select that family. The full per-shape
measurements, family transitions, reported core assignments, and model objectives are in
[comparison.csv](comparison.csv), with the [changed](family_changed.csv) and
[unchanged](family_unchanged.csv) subsets. [analysis.json](analysis.json)
contains aggregate metrics and the [portable audited records](records.json)
allow re-running [analyze.py](analyze.py) locally with `--records records.json`.
The full `msprof` traces remain in the remote case directory above.

**Decision:** keep the FP32 product gate closed. Before using this family
selector online, add source-legal FP32 single-core Split-K mode 2 to the
candidate search and cost model. The BF16 mode-6 domain cannot be reused
unchanged: the eleven measured FP32 master mode-2 cases use `baseK=64`,
usually `singleCoreK=256`, while BF16 mode 6 commonly uses `baseK=128` and
`singleCoreK=512`. Correct the FP32 BASE long-K call cost and the small-M
BASE decision, then repeat a paired validation on a fresh shape set.
Kernel-time gains also need an end-to-end check that includes online scoring
overhead.
