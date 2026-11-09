# BF16 MatMulV3 online selector versus clean master — 910B3, 2026-11-03

## Current status after mode-5 calibration

The subsequent [100-shape BF16 comparison](../bf16_100_vs_master_v1/README.md)
is complete on devices 4, 6, and 7. The calibrated selector changed family
on 11 shapes and improved ten of them in both timing orders. It also chose
BASE for `16×16×65408`, which measured 594.08 µs versus master's 29.55 µs
deterministic Split-K. Across all 100 selected shapes the geometric mean
master/online ratio is 0.9895×. The severe boundary regression must be fixed
before BF16 online selection is enabled by default.

The selector branch at `f5ccfe81` includes [BF16 source-kernel mode-5
calibration](../bf16_mode5_calibration_v1/README.md). Online now chooses
mode 5 with the same tile as master for the large-M, `N=384`, `K=1536`
cases that chose BASE in the original experiment below. Five paired
master/online validations emitted identical mode-5 tiling; their kernel
times were within about 1% by the median of two reversed-order rounds.

Across this report's 11 targeted shapes, the geometric mean master/online
ratio changes from **0.990× before calibration to 1.051×** when the three
old mode-5/BASE results are replaced by fresh paired mode-5 results.
The other eight results are reused because the model change affects only
mode 5. [The calculation and raw new pairs](../bf16_mode5_calibration_v1/overall_status.json)
are saved separately. This is a composite of two experiments, not a new
11-shape timing run or a representative workload estimate.

Three shapes in this set still have a different online family from master:

| Shape M×N×K | Master → current online | Original measured master/online |
| --- | --- | ---: |
| 384×49152×1536 | mode 2 → BASE | 1.061×; repeat was inconclusive |
| 2304×768×32768 | deterministic Split-K → mode 6 | 1.212× |
| 2560×1024×32768 | deterministic Split-K → mode 6 | 1.409× |

The model paths for these three shapes did not change in the mode-5 update;
their timings are the original measurements below. BF16 online selection
remains opt-in, and FP16 selection is unchanged.

## Question and scope

This experiment tests whether the provisional BF16 online family selector in
`codex/matmul-v3-mip-selector` improves source-backed MatMulV3 kernel time over
clean master. It covers same-type BF16 A, B, and C; NN; ND/ND/ND; no bias.
`MMV3_ONLINE_TILE` was unset, so each selected family used the source tiler's
actual block and buffer tile. These are targeted shapes around BASE,
deterministic Split-K, and single-core Split-K modes 2, 5, and 6. They are not
a random or representative workload distribution.

The online host came from `613039da`; clean master came from `b2cdb471`.
CANN 9.0.0 ran on Huawei 910B3 devices 5–7. Device 4 was occupied. Each
source-backed timing run used five warmups and five profiled MatMulV3 tasks;
reported numbers are the median task time in microseconds. Changed-family
pairs were repeated with reversed arm order or on another device. The two
packages used the same benchmark-only MatMulV3 dtype-inference proto library
(`benchmark_infer_dtype.patch`), because the existing direct `aclop` BF16
path failed before tiling without it. The patch was **not** applied to the
product branch. `MMV3_SKIP_LEGACY_OP_CACHE=1` bypassed the optional legacy
cache hook in these standalone packages.

All 36 timed runs passed the harness's patterned-input check. The maximum
observed relative error was 0.003115, below its 0.02 threshold. The emitted
family and tile were parsed from MatMulV3 tiling logs. `measurements.json`
contains each run's timing, correctness result, tile, package, and profiler
CSV path; `selection_probes.json` contains the untimed family/tiling probes
for all 11 shapes; `model_scores.json` contains provisional BF16 objectives.

## Original pre-calibration measured results

Speedup is master time divided by online time. Each cell shows the median of
independent run medians; the number of runs is in parentheses.

| Shape M×N×K | Master mode / µs | Online mode / µs | Speedup | Assessment |
| --- | ---: | ---: | ---: | --- |
| 384×49152×1536 | 2 / 264.45 (2) | 0 / 249.25 (2) | 1.061× | Inconclusive: one pair improved 11.3%, the repeat tied |
| 49152×384×1536 | 5 / 247.81 (3) | 0 / 269.42 (3) | 0.920× | Online BASE slower in all three pairs |
| 65536×384×1536 | 5 / 323.50 (2) | 0 / 417.41 (2) | 0.775× | Online BASE slower in both orders |
| 81920×384×1536 | 5 / 387.86 (2) | 0 / 518.30 (2) | 0.748× | Online BASE slower in both orders |
| 2304×768×32768 | 3 / 900.20 (2) | 6 / 742.67 (2) | 1.212× | Online mode 6 faster in both orders |
| 2560×1024×32768 | 3 / 1314.17 (2) | 6 / 932.66 (2) | 1.409× | Online mode 6 faster in both orders |
| 16×16×102400 | 3 / 41.60 (1) | 3 / 43.28 (1) | 0.961× | Same family and tile control |
| 65536×256×1536 | 0 / 226.80 (1) | 0 / 237.52 (1) | 0.955× | Same family and tile control |
| 2048×2048×27392 | 6 / 1073.30 (1) | 6 / 1071.62 (1) | 1.002× | Same family and tile control |
| 2560×1024×73728 | 3 / 2678.93 (1) | 3 / 2664.26 (1) | 1.006× | Same family and tile control, converted A |
| 384×65536×1536 | 2 / 1027.24 (1) | 2 / 1028.06 (1) | 0.999× | Source fallback; N exceeds model preflight |

The single-run unchanged-family controls vary by up to about 4.7%, so the
small mode-2 result does not establish an improvement. The mode-5 losses at
65536 and 81920, and both mode-6 gains, are much larger than those controls
and repeat with order reversed. Two independent run medians per shape do not
support a formal significance claim.

## Why the original model made these choices

The original BF16 hardware profile reused shared 910B3 rates and omitted
the measured mode-5 Fixpipe/IterateAll pair. BF16 therefore fell back to
the generic serial FP32-store surrogate in modes 2 and 5. For
65536×384×1536, it predicted 552.98 µs for BASE and
1041.71 µs for mode 5, while the emitted kernels measure about 417 and
324 µs. At 81920×384×1536, the predicted BASE/mode-5 values were 689.86 and
1298.01 µs; the measured values were about 518 and 388 µs. The later
calibration measured the missing BF16 mode-5 output primitives.

For 2304×768×32768, the provisional objective separates deterministic
Split-K and mode 6 by only 823.58 versus 818.42 µs, but the measured kernels
separate by about 900 versus 743 µs. At 2560×1024×32768, the model scores
1206.94 versus 1167.60 µs and the measured kernels are about 1314 versus
933 µs. The mode-6 choice is beneficial, but its modeled margin is too small.

## Current next steps

Keep BF16 online selection opt-in. The mode-5 Fixpipe/IterateAll calibration
and held-out source-backed checks are complete. The broader 100-shape
comparison found the narrow, just-below-65535 K BASE regression described
above. Correct that estimate/decision and rerun the affected boundary and
full suite. Then recheck mode 2's output path, the
deterministic-versus-mode-6 margin, and the inconclusive 384×49152×1536
selection.

## Reproduction notes

The benchmark used
`/home/xianfei/solvebase_bench_20261025/bf16_online_vs_master/base_aclnn_bench`
with `BENCH_API=v3cached`, `MMV3_SKIP_LEGACY_OP_CACHE=1`, and the two isolated
packages `ops_nn_master_bf16_proto_package` and
`ops_nn_mip_dtype_bf16_package`. Online runs additionally set
`MMV3_ONLINE_SELECTOR=1` and `MMV3_ONLINE_BF16_EXPERIMENT=1`. For mode 0
without conversion, msprof filtered `KERNEL_AICORE`; the other modes and
converted-A path filtered `MIX_AIC`. The package opmaster SHA-256 values were
`b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`
(master) and
`1a4b2746d1719f7f09c744bdb284e596c46f406394408a447fe9c353009c852f`
(online). Both used proto SHA-256
`b9cc437d49faa817b34b49b77b25517bf40a9450131403662da49f4070972c60`.
The branch source was restored after building that benchmark-only proto.
