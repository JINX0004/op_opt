# BF16 mode-5 source-kernel calibration on 910B3

This experiment measures the BF16 output path of MatMulV3 single-core
Split-K mode 5 (`K=1536`, `N=384`, large `M`) and tests the revised online
family selector against clean master. CANN 9.0.0 ran on devices 5–7. The
original branch source was restored after building a benchmark-only tile
override package; the product change is confined to the BF16 estimator.

## Primitive measurements

The calibration package runs the source-sized core tile and a forced
128×128 core tile within the same mode-5 kernel. Both use 20 AICs,
`singleCoreK=384`, `baseM×N×K=128×128×128`, and `stepM×N×K=1×3×3`.
The small tile is exposed only by the benchmark package's
`MMV3_BF16_MODE5_TILE_PROBE=s` switch. Each arm has two reversed-order
rounds of eight profiled kernel calls after ten warmups. Emitted tiling,
full-output correctness, and device idleness were checked for every arm.
The raw round summaries are in [calibration](calibration/); raw `msprof`
profiles remain on 910 under
`/home/xianfei/solvebase_bench_20261025/cases/bf16_mode5_calib_v1`.

| M | Source core M×N | Source time µs | Small time µs | Source Fixpipe µs | Extra `IterateAll` cost µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 49152 | 2464×384 | 246.744 | 668.620 | 139.956 | 1.7648 |
| 65536 | 3280×384 | 318.217 | 885.003 | 185.584 | 1.7721 |
| 81920 | 4096×384 | 387.438 | 1110.956 | 230.595 | 1.7977 |

For the source-sized tile, each of four K slices writes an FP32 partial.
The effective source-in-context Fixpipe rate is
`4 × singleCoreM × 384 × 4 bytes / Fixpipe active time`: 108169,
108589, and 109134 bytes/µs for the three M values. For the small tile,
the critical AIC executes `4 × ceil((M/128) × 3 / 20)` `IterateAll` calls,
versus four for the source tile. Dividing the difference in AIC scalar
active times by the extra calls gives the last column. We froze the model
at **108000 bytes/µs** and **1.76 µs per `IterateAll`**, using the
49152 training shape. The other two shapes check those values without
refitting. The BF16 rates happen to be close to the existing FP16 rates;
they were measured separately here.

The estimator now applies the concurrent MAC/MTE2/Fixpipe mode-5 slice
formula to BF16 when this primitive pair is present. It previously used
the generic serial FP32-store/atomic proxy and selected BASE. The
standalone selector now predicts mode 5 and the source-sized tile at all
three measured shapes. Its mode-5 times are 212.21, 279.18, and
346.15 µs, respectively: **10–14% below** measured kernel time, so the
absolute timeline remains approximate.

## Source-backed family validation

The revised online host library was installed into a separate package;
the clean-master package was unchanged. `MMV3_ONLINE_TILE` was unset, so
the source tiler emitted the actual block and buffer tiles. Each arm has
two reversed-order rounds of five profiled kernel calls after five
warmups, on the same device per shape. All cases passed the full-output
check. The [validation summaries](validation/) retain each round's
tiling, correctness, and profile location; raw profiles remain under
`/home/xianfei/solvebase_bench_20261025/cases/bf16_mode5_validation_v1`.

| Shape M×384×1536 | Device | Master mode / µs | Online mode / µs | Master / online |
| ---: | ---: | ---: | ---: | ---: |
| 49152 | 6 | 5 / 250.314 | 5 / 247.844 | 1.010× |
| 57344 | 5 | 5 / 281.734 | 5 / 280.784 | 1.003× |
| 65536 | 7 | 5 / 317.386 | 5 / 314.354 | 1.010× |
| 73728 | 6 | 5 / 352.108 | 5 / 353.368 | 0.996× |
| 81920 | 7 | 5 / 405.848 | 5 / 402.136 | 1.009× |

Both packages emitted identical tiling fields at each shape. Their
kernel times are effectively equal, as expected for the same source
kernel and tile. The new model corrects the BF16 large-M family
selection seen in the [earlier comparison](../bf16_online_vs_master_20261103/README.md),
where online BASE was slower than master's mode 5. These paired runs do
not establish a speedup over master. The BF16 selector remains opt-in;
other BF16 family margins still need calibration.

## Overall targeted BF16 comparison

The earlier report measured 11 targeted shapes. The model change affects
only the three large-M mode-5 shapes, so
[the overall analysis](overall_status.json) replaces their old BASE
measurements with the fresh paired mode-5 measurements above and reuses
the other eight shapes' earlier measurements. The geometric mean
master/online ratio changes from **0.990× to 1.051×**. The revised mode-5
subset is 1.010×; its family and tile now match master. The remaining
gain in the aggregate comes mainly from two earlier mode-6 selections
(1.212× and 1.409× in the original run). See [the calculation](analyze_overall.py).

This is a composite of two experiments, not a fresh 11-shape rerun. The
shape set was chosen to exercise selector boundaries and is not a
representative workload distribution. Two reversed-order rounds per new
shape and single runs for some unchanged cases do not establish a
population-level significance claim. FP16 selection and its prior
100-shape result are unchanged.

## Reproduction

The [override](mm_bf16_mode5_override.py) temporarily patches the
experimental host tiler for the isolated calibration package. The
[measurement runner](mm_bf16_mode5_measure.py) records AIC pipeline
counters; the [validation runner](mm_bf16_mode5_validate.py) compares the
revised online package with clean master. The BF16 benchmark-only proto
patch and CANN setup are documented in the [earlier BF16 report](../bf16_online_vs_master_20261103/README.md).
The calibration package host SHA-256 is
`9326d49fb38e9996188c7e3a4e3c48604a00b323bbdfde6e18af43984719bdcb`;
the revised online package host SHA-256 is
`3e0d3a91ad4efcf50ce5af3d05ccbd452c45b8d105fc58f9edf2883d7e053471`.
