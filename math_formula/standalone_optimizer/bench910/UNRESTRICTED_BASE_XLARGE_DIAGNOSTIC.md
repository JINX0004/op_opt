# What happens if `base_core_equals_base` is disabled?

For the same [eight large shapes](packet_base_xlarge_shapes.json), the
unrestricted `SolveBase` model selected a larger `singleCoreM` and/or
`singleCoreN` than its base tile on five. However, its *own* older
byte-only objective improved by at most **0.26%** against the equality-
constrained solve. This is not an optimizer-versus-MatMulV3 speedup: the
tested plain BASE source kernel produces incomplete output for such tilings.

The two [unrestricted](packet_base_xlarge_unrestricted_diagnostic.csv) and
[restricted](packet_base_xlarge_restricted_legacy_diagnostic.csv) solver
audits use identical hardware parameters, one FP32 L0C buffer, and the
older byte-only objective. Only `base_core_equals_base` differs. The
current measured A/B packet profile rejects unrestricted BASE before
solving because its fitted event law is defined only for one base tile per
single-core output tile. It must not be used to compare these audit scores
with the packet-aware [large-shape device audit](PACKET_BASE_XLARGE_910B3.md).

| M × N × K | Unrestricted `(coreM,coreN; baseM,baseN)` | Gain vs constrained, same objective |
| --- | --- | ---: |
| 8192 × 8192 × 8192 | `(256,128; 256,128)` | 0% |
| 12288 × 8192 × 8192 | `(512,128; 256,128)` | 0.13% |
| 8192 × 12288 × 8192 | `(512,128; 256,128)` | 0.13% |
| 12288 × 12288 × 8192 | `(2048,128; 256,128)` | 0.03% |
| 16384 × 8320 × 8192 | `(4096,1664; 256,128)` | 0.26% |
| 8320 × 12288 × 16384 | `(1664,3072; 128,256)` | 0.13% |
| 12288 × 8448 × 16384 | `(256,128; 256,128)` | 0% |
| 8192 × 8192 × 16384 | `(256,128; 256,128)` | 0% |

The [prior full-output source experiment](FULL_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
forced `singleCoreM=256,baseM=128` on `(2048,4096,8192)` and found
4,194,304 output elements left zero; forcing `singleCoreN=256,baseN=128`
likewise left the second 128 columns zero. The current
[`MMV3_RESEARCH_BASE_TILE` override](enable_full_base_tiling_probe.py)
therefore requires `coreM=baseM` and `coreN=baseN`. Repeating a device
timing with one of the unrestricted tuples would measure an incorrect
kernel result.

For plain BASE key 65536, making these plans executable would require
source changes that cover every inner base tile, including tails and C
stores. The source already has a distinct
[B-in-L1 full-load implementation](FULL_LOAD_CORE_TILING_SUPPORT_910B3.md)
that correctly uses larger core tiles on eligible small-K shapes. Model its
implementation conditions and event costs as a separate family before
using those plans in paired performance trials.
