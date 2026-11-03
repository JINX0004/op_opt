# Online BASE L2 fallback versus MatMulV3 master on 910B3

This controlled follow-up tests the one-condition fix for the large-M
BASE regression found in the [expanded comparison](../online_expanded_vs_master_fp16_v1/README.md).
The online selector still chooses its family and path. When it chooses
BASE and `CalcTile` cannot enable an L2 cache tile, it now applies the
same L2 block-grouping fallback as the source `ALL` tiling path.
The fix is committed and pushed on `codex/matmul-v3-mip-selector` at
`de88826180b5f31c0d0d33b50f31b5abbd394792`.

## Protocol

- Clean master package: `ops_nn_master_fp16_package`; host-library
  SHA-256 `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`.
- Patched selector package: `ops_nn_mip_l2fallback_package`, cloned
  from `ops_nn_mip_mode5_package`. Only its two host-library copies
  differ; patched SHA-256 is
  `74a2ce828e903469c7ce4e53e5951256423735bac6e3d8c0fcfdb94ee608e24b`.
- [Five regressed large-M shapes and two controls](shapes.json), FP16
  ND/ND/ND, NN, no bias. Each package is audited for full output and
  emitted tiling. Each timed arm uses ten warmups and eight profiled
  kernel calls. The five large-M cases and the BASE control use four
  reversed-order rounds; the mode-5 control uses two.
- Devices 4–7 are assigned disjoint shapes; each master/online pair
  runs on the same idle-checked device.

Raw profiles are under
`/home/xianfei/solvebase_bench_20261025/cases/online_l2fallback_vs_master_fp16_v1`
on 910. The original online package and its experiment data were kept
unchanged.

## Device results

The [merged CSV](comparison.csv) and [aggregate analysis](analysis.json)
contain all round medians; individual device summaries are
[4](device4.json), [5](device5.json), [6](device6.json), and
[7](device7.json). All seven cases passed output and visible-tiling
audits with zero observed relative error. Master and patched online
selected the same family, key, base tile, buffer steps, block count,
and workspace in each pair.

| M×N×K | Mode | Master µs | Patched online µs | Master / patched |
| --- | :---: | ---: | ---: | ---: |
| 98304×384×1536 | 0 | 621.41 | 620.40 | 1.002× |
| 65536×512×2048 | 0 | 541.57 | 541.04 | 1.001× |
| 81920×512×1536 | 0 | 523.90 | 526.47 | 0.995× |
| 86016×384×1536 | 0 | 549.33 | 548.78 | 1.001× |
| 90112×384×1536 | 0 | 572.09 | 572.87 | 0.999× |
| 65536×256×1536 | 0 control | 236.43 | 231.64 | 1.021× |
| 73728×384×1536 | 5 control | 357.44 | 355.84 | 1.004× |

The first five are the previously regressed large-M shapes. Their
four-round geometric mean master/patched speedup is **0.9995×**,
effectively equal. Before this fix, repeated online medians were
966.63, 643.79, and 680.88 µs on the first three shapes, versus
625.08, 546.05, and 526.60 µs for master. The two upper-boundary
shapes had previous one-round online times of 821.76 and 869.75 µs.
The patched package removes the large-M slowdown without changing
the mode-5 control.

## Changed-family BASE regression guards

The same condition can apply when online selects BASE and master
selects deterministic Split-K. A separate [three-shape guard
run](changed_base_device7.json) used two reversed-order rounds on
device 7. All output audits passed (maximum relative error 0.000445),
and the previously observed online wins remained:

| M×N×K | Master mode → patched mode | Master µs | Patched µs | Master / patched |
| --- | :---: | ---: | ---: | ---: |
| 2048×768×27392 | 3 → 0 | 464.75 | 388.33 | 1.197× |
| 2048×768×32768 | 3 → 0 | 731.48 | 523.11 | 1.398× |
| 768×2048×32768 | 3 → 0 | 635.47 | 505.70 | 1.257× |

The L2 grouping correction therefore preserved these changed-family
benefits in the tested cases. The second row's online round medians
were 556.49 and 489.73 µs, so its exact speedup remains sensitive to
timing variation.

## Descriptive overall estimate

The [projection script](overall_projection.py) replaces the seven
matching shapes in the previous 64-case general set plus eight-case
mode-5 boundary set with the patched paired results above. Its
[saved calculation](overall_projection.json) changes the 72-shape
geometric mean master/online speedup from **0.9955×** for the old
package to **1.0198×** for the hybrid patched estimate. A shape
bootstrap interval within this deliberately selected set is
1.0061–1.0365×. The remaining 65 shapes were measured with the
previous package, so this is **not** a fresh full-suite measurement
of the patched branch. Two residual >5% slowdowns are equal-tiling,
one-round cases in the old data; none of the changed-family cases
showed a >5% slowdown. A full 72-shape rerun is needed to verify the
updated aggregate directly.
