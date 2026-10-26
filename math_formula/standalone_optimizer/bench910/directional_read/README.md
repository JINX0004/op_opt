# Operand-shaped GM→L1 transfer probe

This Ascend C probe isolates the two ND→NZ `DataCopy` shapes used by FP16
BASE tiles. A copies a `[baseM, baseK]` packet into A1 from an ND matrix with
row pitch `K`; B copies a `[baseK, baseN]` packet into B1 with row pitch `N`.
The tested `baseK` is 64, the tile axis is 128 or 256, and each of 20 AICs
issues 8–64 serial packets. Both operands use the same 20-core launch and
profile procedure. The output token checks that each AIC completed.

On the 910B3 with CANN 9.0.0:

```bash
source /path/to/cann-9.0.0/set_env.sh
export PATH=/path/to/cann-9.0.0/tools/bisheng_compiler/bin:$PATH
cmake -S . -B build -DASCEND_CANN_PACKAGE_PATH=/path/to/cann-9.0.0 \
  -DSOC_VERSION=Ascend910B3 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python3 sweep.py --rounds 3
python3 analyze.py > directional_read_analysis.json
```

The sweep uses a 2048×2048×4096 FP16 matrix shape, randomizes the 16
conditions within each round, warms up ten kernels per condition, and records
the last eight `msprof` AI Core task durations. `directional_read_raw.jsonl`
is append-only and lets an interrupted sweep resume. The fit is

`task_us = intercept_us + packet_us × copies + us_per_core_MiB × requested_MiB_per_core`.

The exploratory aggregate rate is `20 MiB / us_per_core_MiB`. The separate
packet slope represents serialization on one AIC; it is not a device-wide
startup per packet. The fit should only be used if its slopes are physical,
residuals are small, repeated rounds agree, and independent whole-kernel
orientation measurements validate its ranking. The input working set is
16 MiB per operand and becomes cache-hot during profiling. This probe does
not reproduce MatMul's L1→L0 loads, Cube, Fixpipe, synchronization, L2
residency, or its overlap and A/B reuse schedule. It is diagnostic evidence,
not a complete kernel latency model.
Within one launch, `core % (M/baseM)` or `core % (N/baseN)` also repeats
some source slices across the 20 AICs. The fitted rate divides *requested*
bytes by time; it is not a measured physical HBM byte rate.

## 910B3 result, 26 October 2026

The [48 profiler points](directional_read_raw.jsonl) and [fit](directional_read_analysis.json)
come from three randomized rounds on the same 910B3 and CANN 9.0.0 setup
as the [source-backed MatMul comparison](../FULL_BASE_OPTIMIZER_VS_DEFAULT_910B3.md).
The pooled effective aggregate rates are **1.117 MB/µs for A** and
**3.323 MB/µs for B**. By-round rates are 1.086–1.181 MB/µs for A and
3.071–3.480 MB/µs for B. Median absolute point residuals are 0.300 µs
for A and 0.162 µs for B. The A packet slope is close to zero and changes
sign in two rounds, so it is not an identified positive per-packet cost.

For the 2048×2048×4096 tile pair, the existing reduced model requests
128 MiB of A plus 256 MiB of B for the source's 128×256 tile, versus
256 MiB of A plus 128 MiB of B for 256×128. The measured effective rates
therefore predict that the source orientation is faster. The isolated-read
gap is **79.7 µs** (74.8–85.0 µs across the three round fits), whereas
the actual full-MatMul gap is **21.4 µs**. This is a directional validation,
not quantitative validation of the reduced objective.

[`model_check.cpp`](model_check.cpp) calls the actual C++ `ReducedObjectiveUs`
for both source-executable tuples on the seven prior workloads. To reproduce
its [CSV output](model_check.csv) after building the standalone optimizer:

```bash
python3 analyze.py > directional_read_analysis.json
cmake -S ../.. -B ../../build/debug
cmake --build ../../build/debug --target mmopt_directional_read_check -j4
../../build/debug/mmopt_directional_read_check \
  0.0008241901108267946 1117456.9210503427 \
  0.0799463128729746 3323227.4422072917 > model_check.csv
```

The recomputed objective favors 128×256 on all seven shapes, but predicts
a **79.7 µs** gap on the 2048 square and **318.9–1275.6 µs** gaps on the
other six. All are much larger than the checked whole-kernel gaps in the
source-backed comparison. The reduced model counts A and B request bytes
without resolving cache hits, reuse between cores, transfer overlap, or
their interaction with Cube. It cannot treat these effective single-operand
rates as validated `gm_read_a`/`gm_read_b` hardware parameters. They are
deliberately **not** installed in the default solver profile. The next
measurement should replay interleaved A/B packets under the source's tile
assignment at both cache-fitting and cache-exceeding working-set sizes,
then validate predicted gaps on new held-out MatMul shapes.
