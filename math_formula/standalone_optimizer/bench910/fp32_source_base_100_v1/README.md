# FP32 source BASE selector: 100-shape comparison on 910B3

## Dataset and provenance

The [dataset](dataset.json) extends the fixed selector's [30-shape experiment](../fp32_source_base_fix_v1/README.md)
with 70 distinct cases. The first 30 shapes and their paired device records are
reused unchanged. The added 70 come from the earlier [FP32 100-case stress
set](../fp32_100_rerun_20261106/dataset.json): 50 spread across its first 80
cases and all 20 FP32 family-boundary cases. The exact [source indices](source_indices.json)
were fixed before running the new comparisons. All 100 shapes are unique.
These 70 cases extend coverage; they are not independent unseen shapes with
respect to the earlier FP32 development work.

The fixed selector is 910 repo branch `codex/matmul-v3-mip-selector`, commit
`e98097f039084705f47751beaa4cb7fdfb44d2d2`. It uses opt-in
`MMV3_ONLINE_SELECTOR=1` and the measured FP32 profile, scoring the source
BASE tile before choosing a family. CANN is version 9.0.0 on Ascend 910B3.
Devices 4–7 run the new shape lanes; devices 0–3 are occupied by other work.

| Package host library | SHA-256 |
| --- | --- |
| Clean master | `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad` |
| Latest online | `b49ce309d8760e2cab3ddaf0eb286684ead170ca775fb8811b4ae1db61925c75` |

The online package is copied from clean master with only the MatMulV3 host
library replaced; `liboptiling.so` is a symlink to that library. For each new
shape, the [runner](run.py) checks device idleness, source tiling provenance,
FP32 output correctness (maximum relative error at most 2%), and that the
unforced branch host emits the same tiling as clean master. Each arm has five
warmups and five timed `msprof` kernel executions. If emitted tiling differs,
two rounds reverse arm order. The [analysis](analyze.py) combines the 30
previously verified records with 70 new records, using the median of round
medians. `master / online > 1` means the online selector is faster.

## Results

All 70 new paired runs completed on devices 4–7 with no audit, correctness,
profiling, or device-idleness failures. The [raw added records](added_records.json)
and [combined analysis](analysis.json) contain all 100 results. The
[per-case table](comparison.csv) includes emitted family, core and base tile,
workspace, and measured kernel times. Separate tables show the
[24 changed-family cases](family_changed.csv) and
[76 unchanged-family cases](family_unchanged.csv).

| Cases | Count | Online faster | Master faster | Geometric mean master / online |
| --- | ---: | ---: | ---: | ---: |
| All | 100 | 47 | 53 | **1.017×** |
| Reused fixed-selector set | 30 | 17 | 13 | 1.050× |
| Added shapes | 70 | 30 | 40 | 1.002× |
| Family changed | 24 | 19 | 5 | 1.080× |
| Family unchanged | 76 | 28 | 48 | 0.997× |

The all-case median is 1.000×. Seventeen cases improve by more than 5%,
and five regress by more than 5%; all 22 change family. A shape-bootstrap
95% interval for the all-case geometric mean is **0.989–1.044×**, so this
100-shape set does not establish a statistically clear overall improvement.
The interval describes resampling these curated shapes, not all possible
MatMulV3 workloads. Kernel times exclude host-side selector overhead.

The largest added-case gains are:

| Shape M×N×K | Master → online | Master µs | Online µs | Master / online |
| --- | --- | ---: | ---: | ---: |
| 2048×64×27392 | deterministic → BASE | 243.26 | 184.36 | 1.319× |
| 2816×1024×49152 | mode 2 → deterministic | 5676.30 | 4461.40 | 1.272× |
| 3072×1024×36864 | mode 2 → BASE | 4094.09 | 3284.28 | 1.247× |
| 512×512×27456 | deterministic → BASE | 266.47 | 220.81 | 1.207× |

The five material regressions are:

| Shape M×N×K | Master → online | Master µs | Online µs | Master / online |
| --- | --- | ---: | ---: | ---: |
| 2560×768×32768 | deterministic → mode 2 | 2033.28 | 4654.32 | 0.437× |
| 16×16×6144 | deterministic → BASE | 15.02 | 19.79 | 0.759× |
| 128×128×7744 | deterministic → BASE | 28.01 | 36.22 | 0.773× |
| 16×16×6208 | deterministic → BASE | 16.44 | 18.67 | 0.880× |
| 128×128×7680 | deterministic → BASE | 28.74 | 32.49 | 0.885× |

For **2560×768×32768**, master emits deterministic split-K with
`singleCoreM/N/K=2560×384×192`, while online emits mode 2 with
`128×256×256`. The two reversed-order rounds were consistent: master
2024.84/2041.72 µs and online 4643.36/4665.28 µs. This is a real family
selection error, and the most important case for improving the mode-2 score
against the tile the source actually emits. The other four regressions cost
2.23–8.21 µs each. The two-round online timing for 16×16×6144 and
128×128×7744 varies more than the other cases, so their exact ratios are
less stable; both rounds still favor master.

The added 70 include shapes from earlier FP32 development experiments. This
run broadens coverage of the fixed selector but is not an independent
holdout evaluation.

## 2560×768×32768: selector-tile device probe

The online selector scores mode 2 with a `128×768×256` single-core tile at
1885.87 µs, versus 1915.17 µs for deterministic split-K. In the ordinary
online host, MatMulV3 recomputes mode-2 tiling and emits `128×256×256`; the
selector applies its family choice but does not inject its candidate tile.

The [temporary build script](build_mode2_model_tile.sh) compiled a separate
host package that forces `128×768×256` only for this FP32 shape and mode 2.
It restored the source checkout after creating the package; the branch source
remained clean. The forced host library SHA-256 is
`fcb430a4353c7389b62051e15f9952f2a0db7ed8e5f7698d015c5cecb91a315d`.
The [probe runner](run_mode2_model_tile.py) used device 4, audited the exact
emitted tile and FP32 output, then profiled five warmups plus five timed
executions per arm in two reversed-order rounds. The output matched the
reference exactly for this input pattern. The [probe record](mode2_model_tile_probe.json)
contains the emitted tiling and per-execution task times.

| Mode | Single-core M×N×K | Round 0 µs | Round 1 µs | Median µs |
| --- | --- | ---: | ---: | ---: |
| Master deterministic split-K | 2560×384×192 | 2033.96 | 2030.50 | **2032.23** |
| Forced selector mode-2 tile | 128×768×256 | 2703.48 | 2700.38 | **2701.93** |
| Ordinary online mode-2 tile (previous paired run) | 128×256×256 | 4643.36 | 4665.28 | **4654.32** |

The model tile makes mode 2 **1.72× faster** than the source-emitted mode-2
tile, confirming that the tiling mismatch is substantial. It remains
**33% slower** than deterministic split-K. The 1885.87-µs model score is
**43% below** the measured 2701.93 µs even when the exact model tile is
emitted, so correcting tile emission alone does not fix this family choice.
The mode-2 timing formula needs further source-kernel calibration.

### Why the exact mode-2 tile is underestimated

The present `ScMode2PipelineSliceUs` score for the `128×768×256` tile
contains 128 sequential K packets per AIC. On a steady packet, it charges
13.98 µs for Cube, 5.69 µs for GM→L1, 5.12 µs for L1→L0, and 4.32 µs
for full atomic output. Its max-plus expression exposes only one
128×128 output-piece drain (0.72 µs) after Cube and predicts **14.70 µs
per steady packet**, producing the 1885.87-µs whole-kernel estimate.

Fresh [device-4 counter profiles](mode2_model_tile_counters.json) identify
the missing service. Busy times overlap and cannot be summed into a
runtime, but they identify which modeled stage is wrong:

| Path | MAC busy µs | MTE2 busy µs | MTE1 busy µs | Fixpipe busy µs | L2 read-hit fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Master deterministic | 1749 | 933 | 750 | 842 | 11.4% |
| Mode 2, source tile `128×256×256` | 1761 | 3662 | 751 | 1066 | 56.7% |
| Mode 2, forced model tile `128×768×256` | 1752 | 2082 | 767 | 734 | 81.4% |

The model's Cube demand is about 1789 µs across 128 packets, close to
the forced-tile MAC busy time. Its GM→L1 term totals only about **728 µs**,
while forced-tile MTE2 is busy for **2082 µs**. These are different
quantities, but the comparison shows that the modeled input path is too
cheap. The source tile has three times as many spatial output tiles (60
versus 20), 29% more reported GM→L1 requests, and still more MTE2 busy
time. Its lower L2 hit fraction may affect input service, but the counters
do not isolate that contribution. The high 81.4% hit fraction of the forced tile means a simple
L2-capacity threshold does not explain its remaining estimation gap.

This case was isolated in the earlier [FP32 mode-2 gap diagnosis](../fp32_base_formula_v2/MODE2_GAP_DIAGNOSIS.md).
With the exact tile held fixed, changing K from 32512 to **32768** and
33024 produces an isolated **574.5-µs** center slowdown, matched by an
approximately 582-µs MTE2 busy-time spike. A's FP32 row pitch at the
center is exactly **128 KiB**. That pitch effect explains about **70%**
of the 817-µs measured-minus-modeled gap. The matched A/B copy probe
reproduced most of the pitch penalty with 20 AICs, without an aggregate
L2 hit-rate collapse. Its physical cause remains unresolved: the data
do not distinguish address mapping, interconnect contention, and MTE2
request scheduling.

The remaining roughly 240 µs is a baseline mode-2 modeling gap. The
current formula recharges A for every N base piece and applies a
3.71-MB/µs **aggregate** L2-hot read rate inside each core's slice.
It also assumes much of Fixpipe overlaps the input/Cube path. These
traffic, occupancy, and overlap assumptions need source-shaped
validation; a uniform read-rate or output penalty worsened the earlier
32 source-tile profiles, as documented in the gap diagnosis.
