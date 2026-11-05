# BF16 narrow-B BASE correction at K=65408 — 910B3

**Follow-up:** The [source-emitted BASE scoring and fresh 100-shape run](../bf16_source_base_v1/README.md)
group the narrow-B read startups by L1 K step and compare the actual
source-prepared BASE tile. The observations below describe the earlier
model before that follow-up.

## Finding

The BF16 online selector inherited the old generic BASE read objective for
`16×16×65408`. It scored a hypothetical `baseK=1024` BASE plan at **18.00 µs**
and chose it over deterministic Split-K at **26.81 µs**. The source tiler
actually emitted a `baseK=64`, `stepKb=64` BASE tile. On device 4, that
source-backed BASE kernel measured **599.28 µs** across two reversed-order
rounds; clean master Split-K measured **30.78 µs**.

The earlier [FP16 narrow-B calibration](../online_base_read_calibration_v1/README.md)
found a large B ND-to-L1 cost when each B source row is 16 elements (32
bytes). We repeated its source-shaped packet probe using `bfloat16_t` and
paired FP16 controls. The BF16 and FP16 packet slopes agree within 0.2% at
both one and 20 active AICs. A BF16-only diagnostic host using the established
FP16 narrow-row coefficients chose deterministic Split-K and measured
**32.05 µs** for the failing shape. Its emitted tile was identical to
master's. The neighboring `32×16×65408` case improved from **604.00 to
30.78 µs** in the same diagnostic.

The experimental BF16 hardware profile now includes the two narrow-B row
coefficients. FP16 profile selection and parameters are unchanged. BF16
online selection still requires `MMV3_ONLINE_BF16_EXPERIMENT=1`.

## Packet measurement

[probe_kernel.cpp](probe_kernel.cpp) is the previous FP16 ND-to-L1 probe
with `half` replaced by `bfloat16_t`; [probe_runner.cpp](probe_runner.cpp)
uses the same host runner. A 16-wide B packet has 4096 K rows and 128 KiB
of input, matching the emitted BASE `stepKb×baseK=64×64=4096` step.
[measure.py](measure.py) randomized three rounds of paired dtype, operand,
width, core count, and one-versus-25-copy event measurements on idle device
4. [task_validate.py](task_validate.py) independently checked the 16-wide
B cases with eight AI Core profiler tasks per variant. The full
[event](event_sweep.jsonl) and [task](task_sweep.jsonl) summaries are saved.

The per-packet slope is `(time(25 copies) − time(1 copy))/24`.

| B packet / occupied AICs | FP16 event µs | BF16 event µs | FP16 task µs | BF16 task µs |
| --- | ---: | ---: | ---: | ---: |
| 16×4096 / 1 | 33.595 | 33.668 | 33.613 | 33.668 |
| 16×4096 / 20 | 48.544 | 48.606 | 48.524 | 48.582 |
| 32×2048 / 1 | 0.406 | 0.469 | — | — |
| 32×2048 / 20 | 0.528 | 0.582 | — | — |

The 32-wide control has the same 128-KiB packet size but avoids the extreme
16-wide row cost. A 16-wide A packet also took less than 0.6 µs per copy in
the event sweep. For this geometry the dtype difference is much smaller than
the narrow-B effect; retaining the existing FP16 fitted one-/20-core rates
of **124.2546 / 85.4893 rows/µs** is supported by these paired BF16 slopes.
These are packet-service coefficients, not full-kernel runtime fits.

## Scoring the emitted tile

[score.cpp](score.cpp) calls the reduced objective on the actual BF16 source
BASE geometry (`singleCoreM=N=16`, `baseM=N=16`, `baseK=64`, one occupied
AIC), as well as the model's chosen `baseK=1024` geometry. The saved
[model scores](model_scores.csv) are:

| Profile | Model-winning BASE, K=1024 | Emitted BASE, K=64 | Deterministic Split-K |
| --- | ---: | ---: | ---: |
| Generic BF16 | 18.00 µs | 228.03 µs | 26.81 µs |
| BF16 + narrow-B rows | 544.41 µs | 754.44 µs | 26.81 µs |

The measured narrow-B packet slope implies roughly 16 packet services for
`K=65408`, about 538 µs before other kernel work. The model's narrow-row
increment is `65408 / 124.2546 = 526.40 µs`, close in scale. However,
the 754.44-µs emitted-tile score exceeds the measured 599.28 µs. Its
generic score already charges read startup per `baseK=64` slice, while the
source groups 64 slices into one 4096-row L1 step. This source-step
accounting remains an unresolved absolute-runtime error. The row term fixes
the **family ranking** for the measured narrow cases; it is not a validated
full BASE timeline estimate.

## Source-backed boundary validation

[validate.py](validate.py) compared clean master, the previous generic BF16
selector, and an isolated BF16 narrow-row package on device 4. Each arm had
an independent full-output and emitted-tiling audit, five warmups, and five
profiled MatMulV3 tasks. The first two shapes had two reversed-order rounds;
other controls had one. All audits passed; maximum relative output error
was **0.002667**. [analysis.json](analysis.json) validates the records. The [portable records](boundary_records.json)
contain each round, tile, package path, and profiler CSV path. Full profiler
traces are under
`/home/xianfei/solvebase_bench_20261025/cases/bf16_narrow_b_calibration_v1`
on 910.

| Shape M×N×K | Master mode / µs | Generic mode / µs | Narrow mode / µs | Generic/narrow |
| --- | ---: | ---: | ---: | ---: |
| 16×16×65408 | 3 / 30.78 | 0 / 599.28 | 3 / 32.05 | **18.70×** |
| 32×16×65408 | 3 / 31.32 | 0 / 604.00 | 3 / 30.78 | **19.62×** |
| 16×16×65536 | 3 / 35.66 | 3 / 34.24 | 3 / 33.70 | control |
| 16×16×102400 | 3 / 40.14 | 3 / 41.20 | 3 / 41.94 | control |
| 256×512×16384 | 3 / 46.00 | 0 / 48.92 | 0 / 50.04 | wider-N control |

For the first two shapes, the narrow package emitted exactly master's
deterministic Split-K tile: `singleCoreM×N×K=384×16×384`,
`baseM×N×K=128×128×128`. The generic BASE tile was
`16×16×65408`, `baseM×N×K=16×16×64`.

[sweep_decisions.cpp](sweep_decisions.cpp) compared generic and narrow-row
BF16 family decisions on the previous [100-shape dataset](../bf16_100_vs_master_v1/dataset.json).
The [decision CSV](decision_sweep.csv) has **one changed family**:
`16×16×65408`, BASE → deterministic Split-K. Two shapes use the same
source fallback under both profiles; the other 97 keep their family.
This is a model-only sweep, not a fresh 100-shape device comparison.

## 100-shape results by family agreement with master

The BF16 row term changes one family decision in the previous 100-shape
dataset: `16×16×65408` becomes deterministic Split-K, matching master.
The following **composite** replaces only that shape's old generic timing
with the fresh paired timing above. It reuses the other 99 device results
from the [original comparison](../bf16_100_vs_master_v1/README.md). A value
above one means online is faster. This is a useful sensitivity calculation,
not a fresh 100-shape device run or a population estimate.

| Online family versus master | Shapes | Geometric mean master/online | Median shape ratio | Online faster / slower / tied | Emitted tile identical |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same family | 90 | 0.9992× | 1.0003× | 46 / 43 / 1 | 90 |
| Different family | 10 | 1.2187× | 1.1757× | 10 / 0 / 0 | 0 |
| **All** | **100** | **1.0193×** | **1.0007×** | **56 / 43 / 1** | **90** |

The same-family group contains 89 unchanged cases from the original run
plus the corrected `16×16×65408` case (master **30.78 µs**, online
**32.05 µs**). Both arms now emit the same deterministic Split-K tile.
Their two-round timing gap varied with order, so it does not establish a
repeatable selector cost. The other 89 cases had identical emitted tiling
and a geometric mean ratio near one.

The ten different-family cases below retain their original two-round device
measurements. Every online choice was faster in both timing orders.

| Shape M×N×K | Master → online mode | Master µs | Online µs | Master/online |
| --- | :---: | ---: | ---: | ---: |
| 256×512×16384 | 3 → 0 | 50.61 | 49.75 | 1.017× |
| 2048×768×27392 | 3 → 0 | 452.04 | 389.74 | 1.160× |
| 2304×768×32768 | 3 → 6 | 902.67 | 771.51 | 1.170× |
| 2560×768×32768 | 3 → 6 | 1070.55 | 879.76 | 1.217× |
| 2560×1024×32768 | 3 → 6 | 1314.13 | 869.08 | 1.512× |
| 2560×1024×49152 | 3 → 6 | 1419.32 | 1196.62 | 1.186× |
| 3072×1024×73728 | 2 → 3 | 4122.26 | 3710.40 | 1.111× |
| 768×2048×32768 | 3 → 0 | 632.62 | 536.22 | 1.180× |
| 1664×768×32768 | 3 → 0 | 500.80 | 320.65 | 1.562× |
| 2432×768×32768 | 3 → 6 | 989.68 | 844.76 | 1.172× |

Modes 0, 2, 3, and 6 are BASE, single-core Split-K mode 2,
deterministic multi-core Split-K, and single-core Split-K mode 6.

## Build and verification

[build_diagnostic.py](build_diagnostic.py) built the isolated row-aware host
library, copied the existing BF16 package, then restored and rebuilt the
original host. The script targets the pre-correction `aaca160a` checkout;
the current branch already contains the BF16 row term. Its diagnostic SHA-256 was
`9cead3c5b74934b0caec72c453ca5c334a435a7354fc20e4e6616630dff59eb7`;
the branch host rebuilt after applying the BF16-only profile change has the
**same hash**. The clean master and generic BF16 host hashes remain those in
the [100-shape report](../bf16_100_vs_master_v1/README.md). Both benchmark
packages use the same benchmark-only BF16 dtype-inference proto patch.

`ctest` passed `mmopt_dtype_tests` (including the new K-boundary assertion)
and `mmopt_fp16_golden`. The branch still keeps BF16 online selection opt-in.
The remaining model work is to score the BASE tile and its L1 K-step read
schedule that MatMulV3 will actually emit, then run a fresh full device suite
before considering default BF16 enablement.
