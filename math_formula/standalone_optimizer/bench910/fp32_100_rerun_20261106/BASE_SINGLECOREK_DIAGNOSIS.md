# FP32 BASE `singleCoreK` diagnosis

The large `singleCoreK` reported for BASE is the full K assigned to one
output block. It is not a resident L0 tile or the length of one FP32
Cube call. MatMulV3 sets `runInfo_.singleCoreK = args_.kValue` in
[`DoBasicTiling`](../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp).
The experimental [selector](../fp32_mode2_v1/remote_selector/model/mip_solver.cpp)
likewise sets `d.single_core_k = w.k` for BASE. That assignment matches
the source's BASE semantics: one AIC owns all K contributions to an
output tile. Changing the reported `singleCoreK` to 8192 without changing
the implementation would misdescribe that ownership. The standalone
[`FinalizePlan`](../fp32_mode2_v1/remote_selector/model/plan_finalizer.cpp)
also rejects a BASE plan whose `singleCoreK` differs from the workload K.

For FP32 output, ND B, HF32 disabled, and $K>8192$, the source's
[`MatmulAswBlock`](../../../../mat_mul_v3/op_kernel/arch35/mat_mul_asw_block.h)
computes an internal call schedule

$$
R=\lceil K/8192\rceil,\qquad
K_{\rm call}=\operatorname{alignUp}\!\left(\lceil K/R\rceil,64\right),\qquad
K_{\rm tail}=K-(R-1)K_{\rm call}.
$$

The [kernel loop](../../../../mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h)
then executes `SetSingleShape`, `Iterate`, and `GetTensorC` $R$ times
per output block. It sets atomic output for calls after the first. The
64-element alignment follows the source's 256-byte alignment for FP32.
For $K=65408$, the schedule is **seven 8192-element calls plus one
8064-element call**. For $K=32768$, it is four 8192-element calls.

The reduced BASE objective instead uses total $K$ in one Cube throughput
term, charges a single output write, and counts GM-read startup by
`ceil(K/baseK)`. It has no $R$-dependent call setup, repeated Fixpipe/
atomic-output service, or FP32 call-level overlap. `baseK` is a different
quantity: the L0 K packet. `stepKa × baseK` and `stepKb × baseK` are the
source's L1 K transfer spans.

## Evidence from the fresh 100-case run

The [diagnostic CSV](base_source_diagnostic.csv) scores all 21
deterministic Split-K → BASE cases twice under the **current** FP32
reduced model: once for its best BASE tuple and once for the exact
source-emitted BASE tuple. The latter uses
[`ReducedObjectiveUs`](../fp32_mode2_v1/remote_selector/model/mip_solver.cpp)
with the same FP32 BASE primitives as the selector. The measured column
comes from the [fresh paired records](records.json).

| Shape | Model's BASE tile (`M×N×baseK`) | Source BASE tile | Internal calls | Model best µs | Model source tile µs | Measured source µs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 16×16×65408 | 16×16×512 | 16×16×32 | 8 | 33.16 | 453.23 | 146.34 |
| 1664×768×32768 | 128×128×64 | 128×256×32 | 4 | 1195.73 | 1195.73 | 1810.66 |
| 2432×768×32768 | 128×128×64 | 128×256×32 | 4 | 1792.25 | 1792.25 | 2634.85 |
| 512×512×27392 | 112×128×64 | 112×128×64 | 4 | 232.96 | 232.96 | 220.32 |

The 16×16 case combines a **candidate mismatch** with a deficient
objective. The model chooses `baseK=512`; the source emits `baseK=32`.
At the source tile, the generic startup expression counts
$2\lceil65408/32\rceil=4088$ A/B requests, while the emitted
`stepKa=stepKb=64` gives only
$2\lceil65408/(64\cdot32)\rceil=64$ L1 K windows. That explains why
scoring the source tuple with the current objective overshoots its
measured time. Simply replacing 4088 with 64 would overcorrect because
the objective still omits the eight FP32 Cube calls and output updates.

For the 1664×768 and 2432×768 shapes, the model gives the source and
its own tile **the same score**: the Cube throughput branch dominates
the objective. The measured source BASE kernels are 51% and 47% slower
than those scores, respectively. This is evidence that the reduced
throughput term misses material FP32 service on those long-K paths.
The 512×512 counterexample is close to its score despite four internal
calls, so a universal fixed penalty per call is not supported by these
observations. The relative contributions of call setup, atomic Fixpipe,
operand transfer, and cache behavior require controlled measurement.

The next model change should keep `singleCoreK=K` for BASE, introduce
an explicit internal FP32 K-call count and output-update service, and
measure that service while holding source M/N and `baseK` fixed. A
source-tile score should be validated before optimizing new BASE tiles
or opening the FP32 online gate.

## Relation to the FP16/BF16 narrow-B failure

The cases share a wrong BASE-versus-deterministic family choice and a
model-versus-source `baseK` mismatch. In the
[FP16 16×16×102400 probe](../online_base_read_calibration_v1/README.md),
however, the measured dominant cost was B ND-to-L1 service for a
16-element, **32-byte** source row: about 844 µs for 25 B packets versus
21.5 µs for A packets. The measured narrow-row term corrected that
family choice. The [BF16 16×16×65408 follow-up](../bf16_narrow_b_calibration_v1/README.md)
confirmed a similar 32-byte B-row effect.

For same-type FP32 at N=16, a B row is **64 bytes**. The existing narrow-B
profile activates only when `input_bytes × baseN < 64`, so it does not
apply here. FP32 K=65408 also stays below the converted-A threshold,
while the FP16 K=102400 case crosses it. The FP32 kernel's eight
internal K calls and repeated output updates are additional behavior
absent from the FP16/BF16 BASE diagnosis. FP32 B-row service may still
matter, but it has not been measured separately; reusing the FP16
32-byte coefficient would not establish its cost.
