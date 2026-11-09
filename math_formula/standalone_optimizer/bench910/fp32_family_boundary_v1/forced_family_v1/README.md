# Same-shape FP32 BASE versus deterministic Split-K on 910B3

This experiment times **both source-emitted families at the same FP32 shape**
to test the family decisions discussed in the [27-case boundary study](../README.md).
The seven cases are listed in [dataset.json](dataset.json). The experiment ran
on 910B3 device 4 with CANN 9.0.0 on 2026-11-06. Inputs and outputs were FP32,
NN, ND/ND/ND, no bias, with HF32 disabled. The product FP32 online selector
remained disabled.

## Method and provenance

The clean-master custom-op package was copied into an isolated research
package. Only its host tiling library was replaced. Its kernel binaries and
other package files remained from clean master. The host library came from
`/home/xianfei/solvebase_bench_20261025/ops_nn_source_build` at `b2cdb471`
with the existing opt-in `MMV3_RESEARCH_SPLITK_FAMILY` probe, extended by
[enable_forced_family.py](../enable_forced_family.py) to accept `base`. The
host library SHA-256 was
`cd074481e4819bffc4e8d85d1d42e9c889d670dc3ecb213264cecb83ebbe5ad3`.
Only the source family decision was forced (`base` or `deterministic`); the
source tiler chose the tile fields within that family. The source build and
package were separate from the primary master and MIP-selector checkouts.

[run.py](run.py) audited unforced/default, forced BASE, and forced Split-K
for every shape. All **7/7** shapes passed full-output FP32 correctness and
tiling-source checks. The unforced research package emitted the same tiling
fields as the earlier clean-master audit for all seven. The forced paths
emitted mode 0 (BASE) and mode 3 (deterministic Split-K) as requested.

For each forced family, `msprof` recorded two randomized rounds of eight
kernel repetitions after five warmups. Device 4 was checked idle before and
after each run. The table uses the median of all 16 kernel times per family.
This is kernel time; it does not include online scoring, host tiling, or
end-to-end API latency. The two round medians and all compact audit/profile
records are in [results/analysis.json](results/analysis.json); raw `msprof`
traces remain at
`/home/xianfei/solvebase_bench_20261025/cases/fp32_forced_family_v1`
on 910.

## Measured family comparison

Mode 0 is BASE; mode 3 is deterministic Split-K. “Model” is the existing
free-tile FP32 CLI family decision, while “master” is the source's unforced
family decision. Neither column represents an executed optimizer-selected
tile: both measured times use source-emitted tiles for the stated family.

| M×N×K | Master | Model | BASE µs | Split-K µs | Faster | BASE / Split-K core M×N×K |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 16×16×6144 | 3 | 0 | 17.68 | **15.62** | 3 | 16×16×6144 / 384×16×192 |
| 64×64×6144 | 3 | 0 | **17.72** | 19.62 | 0 | 16×128×6144 / 384×64×192 |
| 128×128×7680 | 3 | 0 | 32.39 | **27.12** | 3 | 16×128×7680 / 384×128×192 |
| 256×256×7680 | 3 | 0 | **34.70** | 38.76 | 0 | 32×128×7680 / 384×256×192 |
| 512×512×27328 | 0 | 3 | **220.06** | 265.59 | 0 | 112×128×27328 / 384×512×192 |
| 512×512×27392 | 3 | 3 | **220.57** | 265.54 | 0 | 112×128×27392 / 384×512×192 |
| 2048×64×27392 | 3 | 0 | **201.25** | 242.70 | 0 | 112×128×27392 / 2048×384×192 |

The BASE base tiles were respectively 16×16×32, 16×128×64,
16×128×64, 32×128×64, then 112×128×64 for the final three cases.
All Split-K base tiles were 128×128×64. This matters because the source
uses padded tile dimensions and core sizes outside the current FP32 model
search domain.

The model and master each picked the faster source family in **3/7** cases.
The model's errors were 16×16×6144, 128×128×7680, and both 512×512 cases;
master's were 64×64×6144, 256×256×7680, 512×512×27392, and
2048×64×27392. Averaged over these seven shapes, per-shape regret against
the faster measured family was 10.53% for the model and 9.06% for master.
This small, deliberately boundary-focused set does not establish population
performance. Each winner had the same direction in both rounds; the 64×64
case had the most noticeable round-to-round spread in Split-K time
(19.22 versus 20.78 µs).

## What the scores reveal

[analyze.py](analyze.py) also scored each exact source-emitted tile when it
was legal in the current reduced model. The compact [comparison table](results/comparison.csv)
contains these scores. The scorer rejected source-padded geometry at
64×64×6144 and some other source tiles, so empty values mean **outside the
model domain**, not zero time. In particular:

| Shape | BASE source-tile score µs | Split-K source-tile score µs | Measured faster family |
| --- | ---: | ---: | --- |
| 128×128×7680 | 38.70 | 30.77 | Split-K |
| 256×256×7680 | 50.40 | 41.27 | BASE |
| 512×512×27328 | 232.42 | 250.99 | BASE |
| 512×512×27392 | 232.96 | 250.99 | BASE |

At 256×256×7680 the fixed-source scorer **ranks Split-K incorrectly**;
at 512×512 it ranks BASE correctly, but the free-tile CLI selects Split-K
for both K values. Thus there are two distinct problems: timing-model
accuracy for a legal tile, and free-search domain/objective behavior.
The [512×512×27328 diagnosis](DIAGNOSIS_512x512x27328.md) traces the
20-versus-16 BASE core occupancy, the source's MK 3×3 Split-K tile, and
FP32 timing assumptions in detail.
For the small 16×16×6144 BASE source tile, the current score is 45.15 µs
versus 17.68 µs measured. A diagnostic FP32 L1 K-step change gives 3.71 µs,
which overshoots in the other direction. A simple read-startup correction
cannot be deployed alone; the omitted small-tile service cost needs a
separate measurement. The diagnostic build was not deployed.

## Next model work

1. Add source-emittable padded geometries and the observed 384-sized
   Split-K cores to the FP32 domain, then verify emitted tile legality.
2. Measure BASE small-output K-loop/Cube service and transfer startups;
   include source `stepKa`/`stepKb` L1 packet counts without losing the
   currently masked latency.
3. Measure the narrow-N Split-K reduction path and the 256×256×7680 case
   separately, where the fixed-tile family ranking is wrong.
4. Repeat this same-shape test on held-out shapes after the objective and
   domain changes. Keep the product FP32 online gate closed until the
   source-emitted and free-search decisions are both validated.

To rerun the audit and profiles against the prepared isolated package on
910, copy this directory to
`/home/xianfei/solvebase_bench_20261025/fp32_family_boundary_v1/forced_family_v1`
and run `python3 run.py audit`, then `python3 run.py profile`, then
`python3 analyze.py` from that directory. `run.py` checks the host-library
hash before running; its result paths are resumable.
