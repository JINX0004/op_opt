# Expanded FP16 selector comparison against MatMulV3 master

This held-out experiment compares the committed online selector on
`codex/matmul-v3-mip-selector` with a clean MatMulV3 master build. The
[runner](run.py) fixes 64 new, distinct M×N×K shapes, with no overlap
with the preceding [22-case comparison](../online_mode5_vs_master_v1/README.md).
The groups cover narrow outputs and the A-conversion boundary, family
transition shapes and M/N swaps, large-M mode-5 candidates, and larger
M/N workloads.

## Protocol

- Ascend 910B3 devices 4–7, 16 assigned shapes each. Each master/online
  pair runs on the same device, with an idle check before and after each
  audit and profiled timing.
- FP16 input/output, ND matrices, NN, no bias. Master uses the clean
  `ops_nn_master_fp16_package`; online uses `ops_nn_mip_mode5_package`
  with `MMV3_ONLINE_SELECTOR=1` and
  `MMV3_ONLINE_ROW_READ_PROFILE=1`. The host-library SHA-256 hashes are
  `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`
  (master) and
  `bd4a1faa7670d7400f8f67cf4c033013004771a2bec1f471205363cea134d368`
  (online). The online source branch was pushed at commit `e5f81dd2`.
- Both arms independently audit full output and emitted tiling. Timed
  runs use ten warmups and eight profiled calls. Changed selections and
  mode-5 controls receive two rounds in reversed arm order; equal-tiling
  controls receive one round.
- The first launch lacked the CANN library environment and failed
  before device execution. Its outputs were discarded. The recorded
  run initialized the installed CANN 9.0.0 environment.

Raw profiles are under
`/home/xianfei/solvebase_bench_20261025/cases/online_expanded_vs_master_fp16_v1`
on 910. The [analysis script](analyze.py) merges four device summaries
and checks completeness, output error, and changed tilings. The four
[device 4](device4.json), [device 5](device5.json),
[device 6](device6.json), and [device 7](device7.json) summaries,
[per-case CSV](comparison.csv), and [aggregate JSON](analysis.json)
are preserved here.

## Results for the 64 new shapes

All 64 cases completed; maximum relative output error was 0.000445.
The selector changed the family on eight shapes and emitted exactly
the same mode and tile fields as master on the other 56. Seven of the
eight changed-family cases were at least 5% faster by their paired
median; the remaining 256×512×16384 case tied. None had a measured
regression of at least 5%. The changed-family geometric mean was
1.1815×, describing only these eight selected cases.

| M×N×K | Master → online mode | Master µs | Online µs | Master / online |
| --- | :---: | ---: | ---: | ---: |
| 256×512×16384 | 3 → 0 | 48.48 | 48.47 | 1.000× |
| 2048×768×27392 | 3 → 0 | 462.28 | 423.66 | 1.091× |
| 2304×768×32768 | 3 → 6 | 903.74 | 775.09 | 1.166× |
| 2560×768×32768 | 3 → 6 | 1059.03 | 872.26 | 1.214× |
| 2560×1024×32768 | 3 → 6 | 1308.03 | 934.18 | 1.400× |
| 2560×1024×49152 | 3 → 6 | 1452.04 | 1182.10 | 1.228× |
| 3072×1024×73728 | 2 → 3 | 4116.36 | 3707.02 | 1.110× |
| 768×2048×32768 | 3 → 0 | 630.13 | 489.81 | 1.286× |

Modes are 0=BASE, 2=single-core MKN, 3=deterministic Split-K,
and 6=single-core GM-to-L1. The two rounds were run in opposite arm
orders. The 2048×768×27392 case had a much smaller second-round
advantage (467.08 vs 459.37 µs), so its 9.1% median gain is less stable
than the other material wins.

Across all 64 shapes, the raw geometric mean master/online speedup is
**1.0075×** with shape-bootstrap 95% interval **0.9818–1.0321×**.
Replacing the three same-tiling outliers below with their four-round
medians gives **1.0086×**, interval **0.9840–1.0325×**. Both intervals
include 1. This deliberately chosen suite does not establish a general
overall speedup, and equal-tiling single-round differences should not
be credited to the selector.

## Reproducible same-tiling regressions

The three largest equal-tiling deviations were repeated for four
balanced AB/BA rounds on devices 4–6. Their mode, tiling fields,
tiling key, block count, workspace, and profiled kernel name matched.
The packages have the same device-kernel artifacts; only the host
tiling libraries differ. These regressions prompted the host
selection-path investigation below.

| M×N×K | Master median µs | Online median µs | Online slowdown |
| --- | ---: | ---: | ---: |
| 98304×384×1536 | 625.08 | 966.63 | 54.6% |
| 65536×512×2048 | 546.05 | 643.79 | 17.9% |
| 81920×512×1536 | 526.60 | 680.88 | 29.3% |

The [four-round summaries](repeat_44_device5.json)
([second](repeat_46_device4.json), [third](repeat_47_device6.json))
show that the slower online timing persists in both arm orders. In a
separate two-round [control](control_44_selector_disabled_device7.json)
at 98304×384×1536, the *same online package* with
`MMV3_ONLINE_SELECTOR` disabled measured 619.00 µs against master's
624.89 µs. This isolates the problem to the enabled host selection
path; the visible tile tuple alone does not describe all performance
relevant behavior.

### Source-level cause: L2 work assignment differs

The online hook forces `tilingSelect_ = BASE` for a mode-0 decision;
master retains `tilingSelect_ = ALL`. Both reach `DoL2CacheTiling()`,
but its fallback grouping is guarded by
`tilingSelect_ == ALL && !enableCache_`. At N=384 or 512 with
`baseN=256`, the cache-tile search has two N blocks; its 20-AIC tail
test rejects the candidate, so `enableCache_` is false. The recorded
M=98304 logs enter the L2 tiler and have no `CalcTile` update. The
fallback therefore runs only in master. Its L2 grouping is serialized
into `tileL2cacheTiling`, read by the BASE kernel's block scheduler,
and was absent from the initial tiling audit. See the source
[`DoL2CacheTiling()`](../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp)
and [BASE block initialization](../../../../mat_mul_v3/op_kernel/mat_mul_base_block.h).

More generally, this fallback runs only after earlier `ALL` candidates
fail, on hardware with `supportL0c2out=true` and
`supportL12BtBf16=false`, when the full A+B+C footprint exceeds
`100 MiB × (L2 size / 192 MiB)` and `CalcTile` finds no acceptable
L2 cache tile. For 910B3's 192 MiB L2, the threshold is 100 MiB.
The fallback then groups up to four M blocks and up to
`floor(AIC count / 4)` N blocks per L2 tile. A forced `BASE` selector
does not enter this fallback even when all other conditions hold.

For 98304×384×1536, both builds use `baseM=128`, `baseN=256`, and
20 AICs. The source equations give these distinct kernel-facing L2
assignments:

| Path | `mTileCntL2` | `nTileCntL2` | `mTileBlock` | `nTileBlock` |
| --- | ---: | ---: | ---: | ---: |
| Master `ALL` fallback | 192 | 1 | 4 | 2 |
| Online forced `BASE` | 1 | 1 | 768 | 2 |

The [BASE kernel](../../../../mat_mul_v3/op_kernel/mat_mul_base_kernel.h)
uses these fields to traverse and assign output blocks. Thus the same
base tile and kernel key can have different device runtimes. The
measurements establish the impact of this work-assignment difference;
they do not separately quantify cache misses versus core scheduling.
The subsequent [L2 fallback patch and paired rerun](../online_l2fallback_vs_master_fp16_v1/README.md)
restored near-master timing on all five regressed large-M shapes while
preserving three tested changed-family BASE wins.

The 12 large-M shapes in this 64-case set all selected BASE, including
the N=384, K=1536 case at M=98304. A separate held-out boundary
sweep uses [eight additional shapes](mode5_boundary_shapes.json) to
exercise the mode-5 selection region.

## Eight additional mode-5 boundary shapes

The supplemental sweep uses the same packages and per-device paired
protocol. Each of the eight cases passed output audit with zero observed
error, and master/online emitted identical tile fields. Five cases
selected mode 5 in both builds and received two reversed-order rounds.
Their descriptive geometric mean master/online speedup was **1.0022×**:
the revised estimator matches master here, with no measured mode-5
speedup. The three BASE cases received one round.

| M×384×1536 | Both modes | Master µs | Online µs | Master / online |
| ---: | :---: | ---: | ---: | ---: |
| 45056 | 0 | 246.22 | 240.94 | 1.022× |
| 53248 | 5 | 260.36 | 260.03 | 1.001× |
| 57344 | 5 | 281.89 | 282.69 | 0.997× |
| 61440 | 5 | 295.80 | 294.43 | 1.005× |
| 69632 | 5 | 340.28 | 337.40 | 1.009× |
| 73728 | 5 | 350.54 | 350.77 | 0.999× |
| 86016 | 0 | 544.25 | 821.76 | 0.662× |
| 90112 | 0 | 569.47 | 869.75 | 0.655× |

The last two one-round BASE controls show the same large slowdown as
the four-round M=98304 diagnostic above, on two different devices.
Their [device 6](mode5_boundary_device6.json) and
[device 7](mode5_boundary_device7.json) summaries preserve the timings;
[device 4](mode5_boundary_device4.json) and
[device 5](mode5_boundary_device5.json) contain the other cases. The
sharp mode-5-to-BASE decision boundary and the enabled-selector BASE
slowdown merit separate investigation. The supplemental eight cases
are not included in the 64-case bootstrap interval.
