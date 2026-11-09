# FP32 source BASE scoring fix (910B3)

## Change

The FP32 online host now passes MatMulV3's already prepared plain-BASE tile to
`LookupOnlineMipMode`. The lookup replaces the reduced search's hypothetical
BASE result with a score for that source tile, using the FP32 source-aware
K-window and packet expression (`source_emitted_base=true`). The result is then
compared with deterministic and single-core Split-K family scores. This extends
the existing BF16 source-BASE comparison and does not change FP16 behavior or
the source tiler that constructs the actual tile.

The cache key already includes the source BASE geometry and `baseK`, so choices
for different source tiles remain separate. FP32 `MMV3_ONLINE_TILE` remains
disabled; this change adjusts family selection only.

The implementation is on the 910 repo's `codex/matmul-v3-mip-selector` branch
in `experimental/mip_selector/matmul_v3_mip_online.h` and
`matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp`. The two narrow
regressions have host lookup regression checks in `dtype_tests.cpp`.

## Validation protocol

The [runner](run.py) repeats the [same fixed 30-shape FP32 dataset](dataset.json)
as the [prior comparison](../fp32_online_30_new_v1/README.md). Clean master
and the fixed opt-in online host use packages that differ only by the MatMulV3
host library. An unforced run from the fixed host must still emit the same
tiling as clean master. Each case audits source tiling and output correctness
and profiles five warmups plus five timed kernel executions per arm. Cases
with changed emitted tiling use two rounds with reversed arm order. Devices
4–7 run independent shape lanes.

The fixed host library SHA-256 is
`b49ce309d8760e2cab3ddaf0eb286684ead170ca775fb8811b4ae1db61925c75`.
The source is branch commit
`e98097f039084705f47751beaa4cb7fdfb44d2d2`, pushed to
`origin/codex/matmul-v3-mip-selector`.

`mmopt_dtype_tests` and `mmopt_fp16_golden` pass on 910. The two focused
device audits emit deterministic split-K with `singleCoreK=192`, `baseK=64`,
matching master. Their paired kernel times are:

| Shape | Master µs | Fixed online µs |
| --- | ---: | ---: |
| 32×16×65024 | 38.42 | 40.08 |
| 64×32×32704 | 31.82 | 32.76 |

## Full 30-shape result

All 30 shapes completed with no audit, correctness, or device-idleness
failures. The [per-case comparison](comparison.csv), [analysis](analysis.json),
[raw records](records.json), and [previous-versus-fixed table](previous_vs_fixed.csv)
retain the measurements and emitted tiles.

| Selector | Family changes vs master | Improved changed cases | Geometric mean master / online | Shape-bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Previous FP32 online | 8 | 6 | 0.960× | 0.835–1.068× |
| Fixed FP32 online | 7 | 7 | **1.050×** | 1.008–1.110× |

The fixed selector's all-case median ratio is 1.000×; the geometric mean is
driven by its seven useful family changes. The bootstrap interval resamples
this selected 30-shape set, not the population of all matmuls. The 23
same-family cases have a 0.993× geometric mean; most differences are
sub-percent, with a few microsecond-scale fluctuations at short runtimes.

| Shape M×N×K | Master → fixed online | Master µs | Fixed online µs | Master / online |
| --- | --- | ---: | ---: | ---: |
| 1536×96×24448 | BASE → deterministic | 189.13 | 142.16 | 1.330× |
| 4096×96×24576 | BASE → deterministic | 1018.83 | 518.42 | 1.965× |
| 5120×128×24448 | BASE → deterministic | 644.45 | 580.14 | 1.111× |
| 768×512×27520 | deterministic → BASE | 371.70 | 312.87 | 1.188× |
| 2816×1024×32896 | mode 2 → deterministic | 3482.32 | 2893.54 | 1.203× |
| 2048×1536×32704 | mode 2 → BASE | 3277.22 | 2914.74 | 1.124× |
| 1024×2304×36736 | mode 2 → deterministic | 2952.07 | 2694.51 | 1.096× |

The two targeted regressions now match master’s deterministic family and
emitted tiling. The fix also changes four other families relative to the
previous online host. Three of those improve the online runtime, while
`3072×1536×33024` returns from the previous online BASE choice (4704.25 µs)
to master’s mode 2 (5266.48 µs). Its source BASE tile scores about 4812 µs,
but the reduced mode-2 search scores a hypothetical `312×768×192` core tile
at 4529 µs. The source actually emits a `160×1536×256` mode-2 tile, which
scores about 5302 µs and runs at about 5266 µs. Thus source-aware BASE
scoring exposes a remaining source-tiling mismatch in the competing mode-2
score. This is the next modeling issue; the present change deliberately keeps
mode-2 tiling and scoring logic unchanged.
