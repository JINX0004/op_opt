# Large-shape tiling comparison on 910B3

The larger-shape experiment **does not show that the current optimizer is
faster than MatMulV3's default**. On all seven declared FP16 BASE workloads,
the source-executable M-tile restriction of the reduced objective selected
the default `baseM=singleCoreM=128`. A controlled test on three of those
workloads found that forcing the alternative M tile to 64 made the kernel
substantially slower. This validates the model's local ranking of these two
tiles on those shapes, while leaving the full multi-field solver choice
untested on device.

## Workloads and selection

Device 0 is Ascend910B3 with 20 AICs, CANN 9.0.0 and 64 GiB HBM. The
initial device query showed its AIC load at 0%. The [declared seven-shape
suite](compare_optimizer_default.py) includes `(2048,2048,4096)`,
`(2048,4096,8192)`, `(4096,2048,8192)`, `(4096,4096,4096)`,
`(4096,4096,8192)`, `(8192,4096,4096)`, and `(8192,8192,4096)`.
All source MatMulV3 defaults emitted BASE key 65536, no ND→NZ conversion,
and passed the full FP16 output check. Their
[emitted tuples and initial profiler summaries](large_default_cases.jsonl)
show `baseM=singleCoreM=128`, `baseN=singleCoreN=256`, `baseK=64`,
and 20 launch blocks in all seven cases.

The restricted selector holds the source's N/K/core fields fixed and tries
16-aligned downward `baseM=singleCoreM` values through 128. It used the
then-current reduced BASE equation and the exploratory 910B3 primitive slopes,
without reading any candidate runtime. The
[saved objective values and choices](large_selection.json) show that it
selected 128 on **all seven** shapes. Since the selected tiling equals the
default, a timed optimizer-versus-default A/B test has no distinct treatment.

That absolute model was badly calibrated. For the seven emitted default
plans, its estimates are approximately **13.8–15.5×** the source kernel
times. For example, `(2048,2048,4096)` measured 140.2 µs versus 1,929.6 µs
predicted, and `(8192,8192,4096)` measured 1,986.2 µs versus 30,834 µs
predicted. Larger shapes increased timing separation between distinct tiles,
but did not remove the global-packet objective's structural error.

## Randomized controlled large-tile comparison

To test the ranking where device-time differences are large, we declared
three shapes and one source-legal alternative before timing it:
`(2048,2048,4096)`, `(4096,4096,8192)`, and `(8192,8192,4096)`, each
with the source default `baseM=128` versus forced `baseM=singleCoreM=64`.
All other emitted fields, including `baseN`, `baseK`, and 20 launch blocks,
remained under the source tiler. The [runner](repeat_large_spatial.py) made
six independent randomized rounds. Every arm passed full-output and emitted
tiling checks; each run contributed the median of eight `msprof` AIC kernel
task times. [All 36 checked runs](large_paired_m64_raw.jsonl),
[their 288 underlying profiler times](large_paired_m64_tasks.jsonl), and
the [paired analysis](large_paired_m64_analysis.json) are retained.

| Shape `(M,N,K)` | Default median | `baseM=64` median | Geometric mean `T_64/T_default` | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| `(2048,2048,4096)` | 139.98 µs | 188.18 µs | 1.343× | 1.339–1.348× |
| `(4096,4096,8192)` | 998.86 µs | 1469.41 µs | 1.470× | 1.453–1.487× |
| `(8192,8192,4096)` | 1989.01 µs | 3177.33 µs | 1.600× | 1.589–1.611× |

The intervals are two-sided Student-t intervals on six independent paired
log ratios per fixed shape, with five degrees of freedom. Default was faster
in all six rounds for each shape. Exact one-sided paired sign-test
`p=0.015625` per shape; Holm adjustment for the three declared comparisons
gives `p=0.046875` each. Thus the evidence at the 95% family-wise level is
for **the default over `baseM=64` on these three shapes**, not for optimizer
superiority over the default. The shape set is purposive rather than a
random sample of all MatMulV3 workloads, so these intervals do not imply a
population-wide speedup.

## Full solver status

The [original full `SolveReducedMip` audit](large_full_solver_audit.csv)
proposed different BASE plans, typically large `singleCoreM/N` tiles and two
occupied AICs. This exposed a modeling error: its fixed read and write
startup charged every output tile serially, despite parallel AIC assignment.
The revised [busiest-AIC audit](large_wave_solver_audit.csv) uses the same
exploratory coefficients and selects **20 AICs on all seven large shapes**.
It also selects smaller output tiles and estimates 242–3805 µs across the
suite. These are reduced-model estimates, not measured runtimes; for
`(2048,2048,4096)`, the revised 242 µs still exceeds the roughly 140 µs
measured default, while the revised plan is a different tile tuple.

These full solver proposals are not equivalent to the checked M-only
override. The current
source harness cannot apply and finalize the complete tuple, including
dependent buffer depths and L2 fields, so those plans were **not** timed.
The audit initially failed on six shapes because the optional explicit
Split-K event graph demanded whole-output FP32 partial workspace even
though the reduced/source plan reuses per-tile workspace. The
[`mip_solver.cpp`](../mip_solver.cpp) fix leaves that graph score unavailable
without aborting the valid reduced solve; a large-shape regression check was
added to [`mip_tests.cpp`](../mip_tests.cpp). All six local optimizer tests
pass. The original remote MatMulV3 source was not modified; the experiment
used the isolated research package.

The next performance claim requires a source-backed finalizer for the full
solver tuple and a validated overlap/cache timing objective. Once
those are fixed, freeze the candidate and held-out large-shape set before
profiling, then compare full optimizer and default in randomized paired
rounds with a shape-level 95% interval and a practical speedup threshold.

A later [source-compatible BASE evaluation](FULL_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
found that the tested plain BASE kernel leaves output elements zero when
`singleCoreM/N` exceeds `baseM/N`. Restricting the solver to executable
equal outer/base tiles and allowing the source's one-buffer L0C policy
produced a different, correct full tile tuple. Four randomized paired rounds
across these seven shapes measured that solver choice at 1.197× the
default's time (shape-level 95% interval 1.165–1.230×). The reduced model
scores the two orientations equally, so this is evidence of a missing
orientation effect rather than a calibrated prediction of an improvement.
