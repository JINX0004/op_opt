# Experimental FP32 source-informed tiling domain

`SourceCompatibleDomain()` now enables `fp32_source_patterns` for FP32 only.
FP16/BF16 candidates and the product online gate are unchanged. An explicitly
customized tile vector uses the existing unrestricted domain, so callers can
still study alternative geometries.

The FP32 BASE domain generates 16-aligned local M/N tiles up to 256, with
`singleCoreM/N=baseM/N`. It includes 112×128, which gives twenty spatial
tiles at M=N=512, and permits the source-observed N=64 to baseN=128 padding.
L0/L1/workspace checks and the reduced timing objective select among these
capacity-admissible candidates. This is a search domain, **not** an exact
prediction of the source tiler's default BASE tile or baseK for every shape.

For deterministic FP32 Split-K, the domain follows MatMulV3's MK/NK 3×3
path: base 128×128×64, K step 3 (singleCoreK=192), a 384-wide outer output
axis, the source's 70% L2 panel split, and its small-M/N core-count rule.
The source's K>=27392 heuristic is intentionally excluded: it chooses a
family, while the optimizer is meant to compare valid families. The
model preflight now admits source-observed padded FP32 tiles on short axes.

FP32 single-core Split-K mode 2 is now a third search family for same-type
NN/ND output. Its source-informed domain uses 128×128×64 base tiles,
`singleCoreK=baseK×stepK`, an 8-element aligned M core partition, and a
64-element aligned N core partition. The source's 2×4 and 3×3 block
patterns are available; output partitions and occupied AICs are optimized
subject to the source's long-K, workspace, and spatial-load gates. The
FP32 kernel writes FP32 output directly from AIC: its `IterateAll` calls
accumulate K packets in GM and skip BF16's AIV cast/output stage. The model
uses the already measured FP32 atomic-output service as an initial proxy;
the FP32 mode-2 timing rate still needs a dedicated device calibration.
FP32 mode 5 now covers the K=1536, M>=49152, N=384 NN/ND NKM
channel. The experimental branch extends the small-K source host gate
to FP32 when MMV3_FP32_MODE5_EXPERIMENT is set, matching the newer
MatMulV3 source; clean master still selects
BASE for the measured shapes. Mode 5 uses 128x128x64 base tiles,
stepN=3, stepK=3, singleCoreK=192, and direct AIC FP32 output.
The source-compatible domain fixes singleCoreN=384 and optimizes the
8-element-aligned M partition and occupied AIC count. On device 4,
the 73728x384x1536 source tile at N=384 measured 1241.70 us; a forced
N=128 tile measured 1467.98 us, although the reduced byte/Cube score
tied them at 1308.76 us. Repeated A transfers across N panels require
a better MTE2 model before expanding that domain. The FP32 long-K mode-5
route uses transposed B and remains outside the NN/ND preflight. Mode 6
is not a source FP32 path.

## Audit

On the 27-case FP32 family-boundary dataset, all seven source BASE tiles are
legal under the revised preflight, and all twenty source deterministic
Split-K geometry/core-count/K-step tuples are reproduced. The selected BASE
tile is still chosen by the reduced objective and matches the source default
on only one of the seven BASE cases. Device timing of newly selected BASE
tiles remains necessary before treating them as deployable.

At 512×512×27328, the old free-domain scores were BASE 251.43 µs and
Split-K 232.35 µs; the revised domain scores BASE 232.42 µs at
112×128×64 (20 AICs) and Split-K 250.99 µs at 384×512×192. The measured
source kernels were 220.06 and 265.59 µs, respectively. At 512×512×27392
the revised domain also chooses BASE (232.96 versus 250.99 µs), matching
the same-shape forced-family timing, while the clean master chooses Split-K.
Across all 27 boundary shapes, three family decisions change, all at
512×512 near K=27392. Other family decisions remain limited by the FP32
timing profile, particularly small-output BASE and Split-K reduction costs.

`mmopt_dtype_tests` covers the exact 512×512, small-M/N padded, and narrow-N
source patterns. The FP16 golden test passes unchanged. The product FP32
online selector remains disabled.

## FP32 mode-2 follow-up

All eleven measured master mode-2 tuples in the 100-shape FP32 set pass the
new preflight, including the 824×2048×192 output tile with an 8-element M
alignment. The source-tile objective underestimates their measured times by
13.0% mean absolute relative error, so the reused atomic-output primitive
is provisional. Across the 100 shapes, the three-family search chooses
mode 2 on six cases, all of which master also chose as mode 2. The other
five master mode-2 cases choose BASE four times and deterministic Split-K
once; each previously measured alternative was at least as fast as master,
within measurement variation for the 2048×2048×27392 case.

The six new mode-2 choices were rerun as paired clean-master versus
family-forced source kernels on devices 4–7. All emitted tiling fields
matched, all output audits passed, and kernel times differed by less than
1%. Combining those runs with the previously measured pairs gives a
0.981× all-case geometric-mean speedup against master, compared with
0.979× before mode 2. The remaining 21 deterministic Split-K to BASE
changes are the main source of regressions. This test injects only the
implementation family, not the model's selected mode-2 tiling. The FP32
product gate remains closed.

## FP32 mode-5 follow-up

The source-compatible K=1536 NKM candidate passes the standalone
preflight and dtype tests. The 49152x384x1536 source mode-5 tile
2464x384x192 measured 825.42 us versus a model score of 877.03 us;
clean-master BASE measured 823.34 us. At 73728x384x1536, source
mode 5 at 3688x384x192 measured 1241.70 us versus a 1308.76 us
score; clean-master BASE measured 1216.34 us. All runs passed FP32
output checks. These paired profiles support the restricted domain
and BASE ranking on those shapes; they do not validate every M tile
the optimizer can propose.

## Opt-in FP32 mode-2 pipeline experiment

`ExperimentalFp32Mode2Pipeline910B3()` adds optional
`sc_mode2_l0a/b` primitives using Huawei's nominal 910B3 L1→L0A/B
link rates of 460,800/230,400 bytes/µs per AIC. Only FP32 mode 2 uses
them. The experimental slice score is the maximum of Cube plus one
output-piece drain, GM→L1 plus serial A/B L1→L0 service plus that
drain, and total Fixpipe demand. The default FP32 profile still omits
these primitives, leaving all online decisions unchanged.

The separate 32-case fixed-tile K sweep reduced source-tile mean
absolute timing error from 18.21% to 3.22% (17.76% to 2.59% when
excluding the anomalous K=49152 points). Seven other source mode-2
tiles, including wider N and a multi-panel large-M case, improved
from 9.88% to 5.68%. The 2816×1024×36864 source mode-2 score moves
from 3098.38 to 3934.88 µs, versus 3929–3943 µs measured; this
corrects its comparison against the 3358.12 µs source BASE score.
The 2048×2048×27392 source-tile comparison still favors mode 2, which
measures slightly slower than BASE. The model is therefore a diagnostic
step toward a full event graph, not a calibrated online selector.
The nominal L1→L0 rates, source A/B issue order, and mode-2 atomic
output service still need independent measurement.
