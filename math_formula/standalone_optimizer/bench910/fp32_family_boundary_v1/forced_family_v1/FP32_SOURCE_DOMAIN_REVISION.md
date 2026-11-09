# Experimental FP32 source-informed tiling domain

Implementation: `codex/matmul-v3-mip-selector` on 910, under
`experimental/mip_selector/model/` (branch working tree, not yet committed).


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
family, while the optimizer is meant to compare both valid families. The
model preflight now admits source-observed padded FP32 tiles on short axes.

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
