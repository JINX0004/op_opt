# MatMulV3 path that supports core tiles larger than base tiles

The `base_core_equals_base` restriction is specific to the **plain BASE
load mode** evaluated in the large-shape SolveBase studies (key `65536`,
`tilingEnableFullLoad=0`, `tilingEnableSplitCore=0`). It is not a universal
MatMulV3 rule. The source's **B-in-L1 full-load** path uses a different BASE
kernel (key `65538`, `tilingEnableFullLoad=2`, `tilingEnableSplitCore=0`)
and can compute `singleCoreM>baseM` and `singleCoreN>baseN` correctly.

Two default-tiling cases were run on source-built MatMulV3 on Ascend910B3
device 0 with CANN 9.0.0. The runner checked every FP16 output element
against an exact reference and confirmed the emitted source and tiling.
Ten warmups preceded four profiled kernel tasks. The saved records are:

| M × N × K | `(singleCoreM,singleCoreN)` | `(baseM,baseN,baseK)` | Key | Output error | Median kernel µs |
| --- | --- | --- | ---: | ---: | ---: |
| [8192 × 128 × 128](full_load_support_probe/8192x128x128/case.json) | `(256,128)` | `(128,128,128)` | 65538 | 0 | 8.260 |
| [8192 × 384 × 128](full_load_support_probe/8192x384x128/case.json) | `(256,384)` | `(128,256,64)` | 65538 | 0 | 12.482 |

The corresponding [first](full_load_support_probe/8192x128x128/tiling.json)
and [second](full_load_support_probe/8192x384x128/tiling.json) emitted
tiling records identify `tilingEnableFullLoad=2` and show the second case's
`stepN=2`. In
[`DoBL1FullLoadTilingBase`](../../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp),
the host sets `singleCoreM=2*baseM` and `singleCoreN=N`; it also sizes the
B-in-L1 depth from `ceil(N/baseN)`. The dispatched
[`MatmulBaseKernelBL1FullLoad`](../../../mat_mul_v3/op_kernel/mat_mul_l1_full_load.h)
copies B into L1 and calls `IterateAll` for the single-core region. This
explains why it can cover several inner base tiles, whereas the tested
plain BASE path's single `Iterate` left later tiles unwritten.

For FP16/BF16 ND inputs, one source gate for this full-load route requires
`M > 16*max(K,N)` and `K <= 256`. The B on-the-way condition checks
`N` against `{32,64,96,128,160,192,224,256,384}` for non-transposed B;
the B-in-L1 footprint must fit. Other higher-priority tiling branches
and dtype/transpose constraints can still select a different route, so
the inequalities alone do not guarantee key 65538. The validated examples
above show the actual outcome. The earlier square workloads with
`K=8192` or `16384` are outside this small-K full-load gate.

An optimizer can consider larger core tiles by modeling **BL1 full-load as
a separate implementation family**, with its source preconditions, B-in-L1
capacity, and `IterateAll` event costs. Simply clearing
`base_core_equals_base` while continuing to score or execute plain BASE
key 65536 mixes two distinct algorithms.
