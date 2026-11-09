# FP32 online selector: 30 new shapes on 910B3

## Experiment

This is a new 30-shape FP32 set, with no exact-shape overlap with the previous
100-case FP32 comparison. The [dataset](dataset.json) was fixed before measuring
either arm. It covers tiny output matrices with large K (cases 0–1), narrow N
(2–9), middle-size family boundaries (10–19), large M (20–24), and wide N
(25–29). The [offline scores](decisions.csv) were calculated after fixing the
set; they are diagnostic, while the comparison uses the family and tiling
actually emitted by the MatMulV3 host library.

The test compares clean master to the opt-in online host on the same 910B3
devices 4–7, using CANN 9.0.0 and FP32 inputs and outputs. The online package
is a copy of the clean-master package with only the MatMulV3 host library
replaced (the `liboptiling.so` symlink resolves to that same library). An
unforced branch-host run must emit exactly the master tiling for every case;
this controls for changes in default source behavior.

The [runner](run.py) checks device idleness, input/output correctness (maximum
relative error at most 2%), package hashes, source tiling provenance, and
emitted family and tile. Kernel time comes from `msprof`: five warmups followed
by five timed executions per arm. When the emitted tiling differs, the runner
takes two paired rounds with reversed arm order. Each shape is pinned to one
device. The [analysis](analyze.py) uses the median of round medians and reports
`master time / online time`, so values greater than 1 favor online. The
bootstrap interval resamples shapes and describes variability across this
chosen set; it is not a claim about all possible matmuls.

Two same-tiling cases had anomalous first-round timings: `1536×96×24448`
and `3328×64×20160`. We took two additional paired rounds with reversed order
for each and used the median of all three rounds. The repeat rounds brought
the timings within about 1–2%, consistent with the identical emitted tiling
and kernel package. Their full round data remain in the [raw records](records.json).

The measured quantity is device kernel time. The separate host-side online
scoring cost is outside this comparison.

## Provenance

| Item | Value |
| --- | --- |
| Device | Ascend 910B3, device IDs 4–7 |
| CANN | 9.0.0 |
| Online branch | `codex/matmul-v3-mip-selector` |
| Online source commit | `9ed7e5b04fe1b7d2cd423d163240d784cc1cebf8` |
| Master host SHA-256 | `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad` |
| Online host SHA-256 | `33855e2c05271c69389365bfbf344d4c6b5382607bf806fd116d9bdba7d302c0` |
| Online switch | `MMV3_ONLINE_SELECTOR=1` |
| Offline scorer SHA-256 | `e383fbaebe2756306161b00cbcb223f8715f5f084f7a936ad2a8cc9e1d84a44b` |

The offline scorer is `experimental/mip_selector`'s `mmopt_selector`, built
with `ExperimentalFp32BaseMte2910B3()`. It receives one line per dataset
shape as `M N K fp32` and writes the [decision CSV](decisions.csv). The
online host choice is independently audited from the MatMulV3 tiling log.

## Results

All 30 shapes completed with no correctness, source-provenance, package,
or device-idleness failures. The [per-case comparison](comparison.csv),
[changed-family rows](family_changed.csv), [unchanged-family rows](family_unchanged.csv),
[analysis summary](analysis.json), and [raw paired records](records.json)
retain the evidence. The offline scorer and online host agreed on all 30
family choices.

| Group | Cases | Online faster | Master faster | Geometric mean master / online |
| --- | ---: | ---: | ---: | ---: |
| All | 30 | 19 | 11 | **0.960×** |
| Family changed | 8 | 6 | 2 | 0.860× |
| Family unchanged, same emitted tiling | 22 | 13 | 9 | 0.999× |

The all-case median ratio is 1.000×. The shape-bootstrap 95% interval for the
geometric mean is 0.835–1.068×; it crosses 1. The win/loss counts include
sub-percent differences in same-tiling cases and should not be interpreted as
meaningful wins. Six cases improved by more than 5%, while two regressed by
more than 5%. Excluding the first two tiny-output cases *post hoc* gives a
1.045× geometric mean over the other 28; this is a diagnostic showing where
the loss arises, not the headline result.

### Family choices that changed

Times are device kernel microseconds, using the median of paired rounds.

| Shape M×N×K | Master → online | Master | Online | Master / online |
| --- | --- | ---: | ---: | ---: |
| 32×16×65024 | deterministic → BASE | 38.82 | 144.62 | 0.268× |
| 64×32×32704 | deterministic → BASE | 38.50 | 121.42 | 0.317× |
| 4096×96×24576 | BASE → deterministic | 1030.29 | 513.34 | 2.007× |
| 5120×128×24448 | BASE → deterministic | 638.12 | 596.67 | 1.069× |
| 2816×1024×32896 | mode 2 → BASE | 3493.19 | 2931.71 | 1.192× |
| 2048×1536×32704 | mode 2 → BASE | 3278.02 | 2914.85 | 1.125× |
| 3072×1536×33024 | mode 2 → BASE | 5264.34 | 4704.25 | 1.119× |
| 1024×2304×36736 | mode 2 → deterministic | 2948.76 | 2691.13 | 1.096× |

The severe losses at small M,N dominate the geometric mean. In both, the
online selector chose BASE with a full-K `singleCoreK`; master used
deterministic split-K with `singleCoreK=192`. This is a concrete FP32 model
and source-tiler mismatch to repair before considering default enablement. The online scorer
predicted 35.17 and 25.90 µs for those BASE paths, versus 144.62 and
121.42 µs measured (4.11× and 4.69× underestimation). The 22 unchanged
family cases emitted the same tiling in both packages and had a 0.999×
geometric mean ratio after the repeat-round check.

## Why the eight choices differ

For FP32, `MMV3_ONLINE_SELECTOR=1` selects a family but leaves tile construction
to the MatMulV3 source tiler. The [offline model decisions](decisions.csv) are
therefore scores of *model-optimal* tiles, which can differ from emitted tiles.
The [emitted-tile scorer](score_emitted_tiles.cpp) reapplies the same FP32
profile to each audited source tile; its [input](emitted_input.txt) and
[scores](emitted_scores.csv) are included so the distinction is inspectable.

| Shape | Master emitted score | Online emitted score | Master measured | Online measured |
| --- | ---: | ---: | ---: | ---: |
| 32×16×65024 | 41.81 | 149.19 | 38.82 | 144.62 |
| 64×32×32704 | 36.57 | 82.70 | 38.50 | 121.42 |
| 4096×96×24576 | 466.26 | 364.70 | 1030.29 | 513.34 |
| 5120×128×24448 | 778.95 | 516.91 | 638.12 | 596.67 |
| 2816×1024×32896 | 3512.14 | 2996.95 | 3493.19 | 2931.71 |
| 2048×1536×32704 | 2998.67 | 2979.47 | 3278.02 | 2914.85 |
| 3072×1536×33024 | 5301.50 | 4812.15 | 5264.34 | 4704.25 |
| 1024×2304×36736 | 2782.86 | 2544.13 | 2948.76 | 2691.13 |

The emitted tile is shown as `singleCoreM×singleCoreN×singleCoreK /
baseM×baseN×baseK`; workspace is in MiB.

| Shape | Master tile; workspace | Online tile; workspace |
| --- | --- | --- |
| 32×16×65024 | `384×16×192 / 128×128×64`; 20.9 | `16×16×65024 / 16×16×32`; 20.0 |
| 64×32×32704 | `384×32×192 / 128×128×64`; 21.9 | `16×32×32704 / 16×32×32`; 20.0 |
| 4096×96×24576 | `128×96×24576 / 128×96×32`; 20.0 | `4096×384×192 / 128×128×64`; 260.0 |
| 5120×128×24448 | `128×128×24448 / 128×128×64`; 20.0 | `5120×384×192 / 128×128×64`; 320.0 |
| 2816×1024×32896 | `144×1024×256 / 128×128×64`; 31.0 | `128×256×32896 / 128×256×32`; 20.0 |
| 2048×1536×32704 | `208×768×256 / 128×128×64`; 32.0 | `128×256×32704 / 128×256×32`; 20.0 |
| 3072×1536×33024 | `160×1536×256 / 128×128×64`; 38.0 | `128×256×33024 / 128×256×32`; 20.0 |
| 1024×2304×36736 | `256×512×256 / 128×128×64`; 29.0 | `384×2304×192 / 128×128×64`; 155.0 |

The deterministic path may report a `singleCoreM` or `singleCoreN` larger
than the logical shape because its source tiler uses padded block geometry;
the reduction is divided across K and the source launches 20 AICs.

In the first two cases, the optimizer scored BASE with `baseK=512`; the source
emitted `baseK=32`. It also emitted only 2 and 4 spatial output tiles,
respectively. Re-scoring those actual tiles would prefer deterministic split-K.
Across all eight changed cases, the emitted-tile score ranks the two families
in the same direction as measured runtime. This supports scoring the tile that
the FP32 host can actually emit when selecting a family.
The second case had a large between-round timing shift (online 158.66 then
84.18 µs; master 45.60 then 31.40 µs), so its measured median is less stable,
but both rounds strongly favor master and the emitted-tile scores agree on the
family ranking.

For `4096×96×24576`, the model chooses the faster deterministic family but
underestimates the runtime of both emitted paths. For `5120×128×24448`, it
overstates the benefit of deterministic split-K. The three mode-2→BASE
changes use a source BASE tile of `128×256×32` rather than the model's
`128×128×64` tile; their emitted-tile scores still rank BASE correctly.
At `2048×1536×32704`, that ranking is narrow (0.6% predicted) despite an
11% measured improvement, so nearby shapes merit a boundary check.
