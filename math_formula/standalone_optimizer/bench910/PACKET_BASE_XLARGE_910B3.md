# Larger-shape SolveBase check on 910B3

The packet-aware optimizer and MatMulV3's source default selected the **same
tile on all eight declared larger shapes**: `(singleCoreM, singleCoreN,
baseM, baseN, baseK, blockDim) = (128, 256, 128, 256, 64, 20)`.
Consequently this suite has no distinct optimizer/default treatment and
cannot establish a speedup. The device runs do test the absolute objective:
it underpredicted the observed kernel time by 3.7–20.5% (10.2% mean
absolute percentage error across these eight shapes).

The [declared suite](packet_base_xlarge_shapes.json) was fixed before device
execution. The [local C++ solver audit](packet_base_xlarge_solver_audit.csv)
used `source_base` with the optional packet primitive profile, 20 AICs,
`singleCoreM=baseM`, `singleCoreN=baseN`, one L0C buffer, and M/N candidate
grid `{64,128,256}`. The [source default audit](packet_base_xlarge_defaults.jsonl)
was then measured on Ascend910B3 device 0 with CANN 9.0.0, FP16 ND×ND,
MatMulV3 BASE key 65536, and the isolated source-built operator package.
Each run checked the full output exactly, the emitted tile tuple and source
path, and used ten warmups followed by four profiled kernel tasks. The
[frozen selection](packet_base_xlarge_selection.json) records the matching
choices and hashes the two audits. Times below are medians of the four
device kernel tasks, in microseconds; these are not host API timings.

| M × N × K | Objective (µs) | Device (µs) | Objective error |
| --- | ---: | ---: | ---: |
| 8192 × 8192 × 8192 | 3873.72 | 4020.58 | −3.7% |
| 12288 × 8192 × 8192 | 5790.57 | 6038.33 | −4.1% |
| 8192 × 12288 × 8192 | 5790.57 | 6171.49 | −6.2% |
| 12288 × 12288 × 8192 | 8684.52 | 9159.66 | −5.2% |
| 16384 × 8320 × 8192 | 7932.48 | 9061.28 | −12.5% |
| 8320 × 12288 × 16384 | 11664.50 | 14639.56 | −20.3% |
| 12288 × 8448 × 16384 | 11888.70 | 13083.78 | −9.1% |
| 8192 × 8192 × 16384 | 7702.34 | 9693.72 | −20.5% |

For example, the model gives identical 5790.57 µs objectives to the
transposed 12288×8192×8192 and 8192×12288×8192 workloads, but device
medians differ by 2.2%. This points to orientation-dependent scheduling or
data movement that the reduced objective does not capture completely. The
larger errors at K=16384 are another sign that its primitive extrapolation
needs checking; they do not by themselves identify which primitive is at
fault. These are single four-task runs per shape, so the table is an
accuracy diagnostic, not a confidence interval for the error or an A/B
performance result.

For the next meaningful optimizer/default comparison, first improve the
modeled source L2 tile and AIC assignment described in the
[prior packet-objective report](PACKET_BASE_OPTIMIZER_VS_DEFAULT_910B3.md),
then freeze a new suite with source-valid shapes where optimizer and default
actually differ. Additional same-tile A/B timings would measure the same
implementation twice.
