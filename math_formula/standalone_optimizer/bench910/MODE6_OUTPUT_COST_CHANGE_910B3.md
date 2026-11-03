# Mode-6 steady FP32 output cost: provisional model stage

This records the generic-write proxy used before the
[device-4 steady-output measurement](MODE6_OUTPUT_MEASUREMENT_910B3.md).
The measured profile now supplies a source-in-context total FP32 atomic
output demand; the table below remains a historical proxy comparison.

## Change

The earlier reduced objective used the same `sc_fp32_store` coefficient
for mode 6's first plain Fixpipe and every later atomic K packet. That
coefficient came from a [serial, distinct-address output probe](primitives/probe_kernel.cpp).
The [mode-6 source](../../../mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel_gm_to_l1.h)
instead updates the same FP32 workspace across K with an L0C/Fixpipe
pipeline. The [timing diagnosis](SPLITK_FAMILY_TIMING_DIAGNOSIS.md)
found that the serialized output term alone can exceed the measured
whole-kernel time per K packet.

[`ScMode6MacUs`](../mip_solver.cpp) now uses:

| Path | FP32 store service | Atomic increment |
| --- | --- | --- |
| Mode 2/5 | `sc_fp32_store`, else `gm_write` | Existing model |
| Mode 6, first K packet | `sc_fp32_store`, else `gm_write` | None |
| Mode 6, later K packets | `sc_fp32_store_steady` as total output demand, else `gm_write` | Included in measured steady demand; otherwise `sc_atomic_extra` when supplied |

`sc_fp32_store_steady` is a new optional **bytes** primitive. The generic
`gm_write` fallback is an optimistic steady-output proxy, not a measured
FP32 rate. The [later study](MODE6_OUTPUT_MEASUREMENT_910B3.md) measured
source-in-context Fixpipe demand and supplies `sc_fp32_store_steady` in
the 910B3 exploratory profile. This change is confined to mode 6's later output
term; its operand-copy recurrence and deterministic objective are
unchanged. The earlier [frozen model predictions](splitk_family_model_predictions.csv)
and [source scores](splitk_family_source_geometry_scores.csv) are retained
for comparison.

## Check against measured source tiles

The provisional [source-tile scores](splitk_family_source_geometry_scores_output_proxy.csv)
were produced before the measured steady primitive was added to
[`splitk_family_model.cpp`](splitk_family_model.cpp). The other
exploratory 910B3 hardware coefficients were unchanged. All values
are µs.

| M × N × K | Old mode 6 | Revised mode 6 | Measured mode 6 | Deterministic estimate / measured |
| --- | ---: | ---: | ---: | ---: |
| 2048 × 2048 × 27392 | 2502.7 | 1288.5 | 1072.6 | 1668.6 / 1251.8 |
| 2048 × 3072 × 27392 | 3763.7 | 1942.4 | 1596.9 | 2489.2 / 1819.3 |
| 2048 × 2048 × 32768 | 2972.5 | 1533.1 | 1449.7 | 2026.9 / 1717.5 |
| 2048 × 3072 × 32768 | 4470.5 | 2311.3 | 2091.1 | 3026.7 / 2460.3 |

Mode-6 source-tile overprediction falls from **105–136%** to **6–22%**.
The revised source-tile objective ranks mode 6 faster than deterministic
on all four shapes, agreeing with measured ranking. It still
overstates the **size** of the mode-6 advantage, so this is not a
calibrated family decision rule.

With the revised objective, the full MIP selects mode 6 on all four
wide shapes and deterministic on all four narrow shapes
([predictions](splitk_family_model_predictions_output_proxy.csv)). Its
family decision matches the faster executed branch on 7/8 shapes,
the same as source default; it offers **no measured speedup** on this
suite. Both selectors miss 2048 × 768 × 32768, where source mode 6
ran faster. These decisions use MIP scores, while the table above
checks source-tile scores.
The MIP selects `singleCoreM=512`, `singleCoreN=624` for the two N=3072
cases rather than the source's `208×1536`; those MIP tiles have **not**
been executed. The N=768 source mode-6 tiles remain excluded by the
existing spatial host gate, so this change does not resolve that
separate issue.

## Validation and later measurement

The native solver builds, and `mmopt_mip_tests` passes. A first build
against `/usr/local/Ascend/ascend-toolkit/latest` failed at CANN object
merge; rebuilding against the CANN 9.0.0 package used by prior probes
succeeded. The [measurement report](MODE6_OUTPUT_MEASUREMENT_910B3.md)
documents the completed output sweep, source pipeline counters, and a
larger source-kernel holdout. The original proxy comparison here should
not be read as a measured FP32 primitive rate.
