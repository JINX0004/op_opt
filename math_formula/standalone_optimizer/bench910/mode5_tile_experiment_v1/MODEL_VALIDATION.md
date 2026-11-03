# FP16 mode-5 service model and held-out validation on 910B3

## Primitive derivation

The [paired 49152×384×1536 device run](README.md) measured the
source-sized `2464×384` tile and the online model's `128×128` tile.
Both used mode 5, 20 AICs, 384-K slices and 128×128×128 base blocks.
The critical AIC writes about 15.14 MB of FP32 partials over four K
slices. Its reported Fixpipe active time is 139.96 µs, yielding about
108 kB/µs of source-in-context total plain-plus-atomic output demand.
The small tile's corresponding rate is about 107 kB/µs. The new profile
uses `sc_mode5_fixpipe = {bytes, 0, 108000}`.

On the critical AIC, the large tile makes four `IterateAll` calls and
the small tile makes about 232. Their scalar active times differ by
`539.036 - 127.906 = 411.130 µs`; dividing by 228 extra calls gives
1.803 µs/call. The new profile uses
`sc_mode5_iterate = {count, 1.8, 1}`. These are component measurements,
not a multiplier fitted to whole-kernel time.

For one $m\times n\times k$ K slice, the optional FP16 mode-5 path uses

$$
T_{\mathrm{slice}}=
\max\!\left(\frac{2mnk}{R_C},\;
\frac{D(m,n,k)}{R_R}+2m_bn_bq_k\lambda_R,\;
\frac{4mn}{R_{F,5}}+m_bn_b\lambda_{F,5}\right)
+\lambda_{I,5}.
$$

It sums the four K slices and retains the existing AIC/AIV tile
pipeline. The Fixpipe term includes atomic updates, so the separate
serial-probe `sc_atomic_extra` is not added. The old mode-5 formulation
remains the fallback when the paired primitives are absent and for BF16.
See [the formulation](../../SOLVE_SC_SPLITK_MODE5_FORMULATION.md).

## Frozen predictions versus device time

The two coefficients above were fixed using only M=49152. The M=65536
and M=81920 predictions were generated before their device timing.
Each held-out shape passed full-output correctness and emitted the
intended source and small mode-5 tiles; see the audit records for
[65536 large](heldout_65536_large_audit.json),
[65536 small](heldout_65536_small_audit.json),
[81920 large](heldout_81920_large_audit.json), and
[81920 small](heldout_81920_small_audit.json). Two reversed-order rounds of
eight `msprof` kernel calls were run per tile after ten warmups on
device 5. [M=65536](heldout_65536_summary.json) and
[M=81920](heldout_81920_summary.json) contain round medians and AIC
pipeline counters. The raw profiles remain under
`/home/xianfei/solvebase_bench_20261025/cases/mode5_tile_heldout_v1`.

| M×384×1536 | Tile | Model µs | Device µs | Model error |
| --- | --- | ---: | ---: | ---: |
| 49152 | source 2464×384 | 212.37 | 248.31 | −14.5% |
| 49152 | small 128×128 | 618.97 | 672.58 | −8.0% |
| 65536 | source 3280×384 | 279.34 | 315.49 | −11.5% |
| 65536 | small 128×128 | 820.73 | 886.99 | −7.5% |
| 81920 | source 4096×384 | 346.31 | 397.08 | −12.8% |
| 81920 | small 128×128 | 1022.50 | 1108.82 | −7.8% |

The model selects the faster mode-5 tile at all three shapes, including
both held-out shapes. Absolute kernel time is still underpredicted,
especially for the source-sized tile. The measurement supports this
tile-ranking correction within the FP16 K=1536, N=384 host channel;
it does not establish a universal scalar setup rate.

## Online selector diagnosis and master comparison

The updated selector was built into the separate
`ops_nn_mip_mode5_package` custom-op package. The FP16 timing used host
library SHA-256
`3a5a09b78825da5f4ae06450fe563e960b1713b15d8cd1be454a386ffe3414d8`.
After timing, the measured path was restricted explicitly to FP16 so
BF16 keeps its prior formula; this does not change the timed FP16 path.
The final rebuilt package SHA-256 is
`bd4a1faa7670d7400f8f67cf4c033013004771a2bec1f471205363cea134d368`.
The previous `ops_nn_mip_grouped_package` was preserved. Both packages
passed output and emitted-tiling audits. Old online selected BASE
`128×256`; revised online selected source mode 5, with `singleCoreM`
2464/3280/4096 and `singleCoreN=384` for the three shapes. No online
mode-5 tile override was used for this comparison.

The following old-online-versus-revised comparison diagnoses the model
change; it is not the performance comparison with MatMulV3 master.
Old and revised packages were paired on the same device per shape,
with reversed order in round two: device 5 for M=49152, device 6 for
M=65536, and device 7 for M=81920. Each timing arm used eight profiled
kernel calls after ten warmups. Results are in the
[49152](online_compare_49152.json),
[65536](online_compare_65536.json), and
[81920](online_compare_81920.json) summaries; raw profiles are under
`/home/xianfei/solvebase_bench_20261025/cases/online_mode5_model_compare_v1`.

| Shape | Old BASE median µs | Revised mode 5 median µs | Old / revised |
| --- | ---: | ---: | ---: |
| 49152×384×1536 | 252.06 | 246.85 | 1.021× |
| 65536×384×1536 | 470.30 | 316.74 | 1.485× |
| 81920×384×1536 | 797.99 | 402.68 | 1.982× |

At M=49152, old-online BASE varied from 239.34 to 264.77 µs across the two
rounds, while revised mode 5 stayed near 246.85 µs. The 2.1% median
advantage is smaller than that drift and should not be claimed as a
confirmed speedup. At M=65536 and 81920, the revised mode is much
faster in both paired rounds. The revised objective predicts mode 5 at
all three shapes. At M=49152, however, it scores mode 5 at 212.37 µs
and BASE at 416.10 µs, whereas their paired device medians differ by
only 5.21 µs. The cross-family score margin remains much less
trustworthy than the within-mode-5 tile ranking.

The subsequent [fresh 22-shape comparison with clean master](../online_mode5_vs_master_v1/README.md)
used the final rebuilt revised package. Master and revised online both
selected mode 5 with identical tile fields on all three shapes. Their
master/revised times were 247.94/247.37, 315.43/316.05, and
398.71/396.84 µs, respectively. The correction aligns the selector
with master here; it does not demonstrate a speedup over master on these
three shapes.

The local CTest suite passed 6/6. The temporary host tiling override
used for the tile measurements was restored before building the online
package. The experimental branch has only the mode-5 model and profile
changes; master and previous online packages were not edited.
