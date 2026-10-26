# Measuring `SolveBase` parameters on a 910B device

This is the execution plan for replacing the illustrative primitive profile
used by [`SolveBase`](SOLVE_BASE_FORMULATION.md). It starts with contiguous
FP16 ND inputs, no transpose or bias, `K <= 65535`, and the BASE kernel path
with emitted `nd2nzA=nd2nzB=0`. That scope avoids the model's separate A
ND→NZ conversion term. Measure BF16,
conversion, other layouts, and other implementation families in later rounds.
The target is a reusable *device and CANN build* profile, not a different
fit for every tiling.

The installed-CANN measurement harness and first pilot are now in
[`bench910`](bench910/README.md), with a
[provenance and results report](bench910/PILOT_910B3_CANN9.md). The hardware
query, direct MatMulV3 smoke test, emitted tiling capture, correctness
check, and short profiler runs in the first-session checklist are complete.
The first pilot invokes the CANN-bundled operator. A separate
[source-matched run](bench910/SOURCE_MATCHED_910B3_CANN9.md) now invokes the
user's remote `ops-nn-sribd/matmul/mat_mul_v3` implementation through an
isolated package, including a controlled eight-core run. Primitive
isolation and controlled *spatial* tilings have also been exercised. Their
[exploratory results](bench910/PRIMITIVE_910B3_CANN9.md) showed a structural
mismatch between per-AIC packet latency and the original serial
global-packet `SolveBase` objective. The revised code now distributes
output tiles across AICs, but no measured profile has been accepted yet.
The remaining acceptance work is overlap and cache modeling, L2 versus
HBM strata, and broader held-out tiling-rank validation.
The later [source-compatible BASE evaluation](bench910/FULL_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
shows a concrete selection failure: the revised model ties the 128×256 and
256×128 source-executable orientations, and its chosen orientation runs
1.197× slower than the default across seven large shapes. This is direct
evidence that A/B direction and cache effects must enter the objective
before the exploratory primitive coefficients are promoted to a profile.
A later [48-point A1/B1 transfer probe](bench910/directional_read/README.md)
isolates the directional difference and correctly ranks that tile pair,
but its cache-hot effective rates overpredict the full-kernel gap by about
3.7× on the 2048 square. Thus the read experiment has advanced, while the
L2 residency, reuse, and overlap acceptance work remains open.

## 1. Establish the exact device and execution path

On the target machine, save the output of `npu-smi info`,
`npu-smi info -t board -i <device-id>`, the installed CANN and driver versions,
and the device's visible-core restrictions. Query AIC count and L1/L0
capacities from `PlatformAscendC` (`GetCoreNumAic`, `GetCoreNumAiv`,
`GetCoreMemSize`) using the installed SDK. The present standalone profile
assumes **910B3 with 20 AICs**. A machine called “910B” may expose a
different count or be partitioned; use the queried values rather than the
reference profile. Record the device ID, chip name, CANN/driver versions,
clock or power mode if available, and whether other processes use the NPU.

Build the local `mat_mul_v3` source against that CANN installation. For each
test case, record the **emitted** kernel family, tiling key, block dimension,
`usedCoreNum`, `singleCoreM/N/K`, `baseM/N/K`, `stepM/N/Ka/Kb`,
`depthA1/B1`, iteration order, L2 settings, `nd2nzA/B`, fix/output mode,
and workspace. The host already assigns those fields in
[`matmul_v3_base_tiling.cpp`](../../mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp)
near `SetTilingData`. Add a temporary structured dump there if ordinary
logs omit fields. Reject a result from the BASE dataset if the emitted
path is a split-K or special path, or if either conversion flag is set in
this first round. Group distinct L2 and Fixpipe modes separately rather
than mixing them into one primitive fit. Keep the exact source commit with
the results.

The existing [`run_mm_bench.py`](../../mat_mul_v3/experiments/run_mm_bench.py)
is useful for a shape smoke test, but its `torch.mm` timing includes a
high-level call path, uses an approximate family classifier, and does not
control the emitted tiling. It cannot by itself identify primitive rates.

## 2. Define the quantity to fit

For a realized geometry $g=(C_M,C_N,B_M,B_N)$, `baseK=16b`, and occupied
AIC count $c$, the current reduced objective uses the per-AIC loads
$E_q,S_q,O_q$ defined in [`SOLVE_BASE_FORMULATION.md`](SOLVE_BASE_FORMULATION.md):

$$
T_B=\lambda_{\rm launch}+\max\left(
\max_q\frac{2K E_q}{R_C},
\frac{D(g)}{R_R}+\frac{2MN}{R_W}
+\max_q\left(2S_q\left\lceil\frac{K}{16b}\right\rceil\lambda_R
+O_q\lambda_W\right)\right),
$$

where $D(g)=2K(MN_N+NN_M)$, $S=N_MN_N=\sum_qS_q$, and
$N_o=\lceil M/C_M\rceil\lceil N/C_N\rceil=\sum_qO_q$ includes outer-tile tails as
defined in [`SOLVE_BASE_FORMULATION.md`](SOLVE_BASE_FORMULATION.md).
The parameters to estimate first are

| Parameter | Meaning in this *model* | Primary measurement |
| --- | --- | --- |
| $\lambda_{\rm launch}$ | Fixed on-device dispatch/start cost | Minimal device kernel and zero-work intercept; verify with tiny BASE runs |
| $R_C$ | Sustained FP16 Cube FLOP/µs **per occupied AIC** | L1-resident repeated Cube operations, swept by repeat count and AIC count |
| $R_R,\lambda_R$ | Effective GM read bytes/µs and fixed cost per modeled operand read | MTE2 GM→L1 read probes, varying bytes and packet count independently |
| $R_W,\lambda_W$ | Effective output-write bytes/µs and fixed cost per modeled output write | Actual BASE output/Fixpipe path or a matching output microkernel |

These are *effective path parameters*, not necessarily raw HBM peak rates
or literal single-instruction latency. A fit from whole-matmul times alone
is not identifiable: the `max` hides the faster branch, and all work terms
change together when shape or tiling changes. Isolated probes are needed.
Also keep the rate units used by `PrimitiveTiming`: FLOP/µs for Cube and
bytes/µs for read/write; latencies are µs.

## 3. Build one measurement harness

Use a small C++ AscendCL/Ascend C runner, starting from
[`examples/test_aclnn_matmul.cpp`](../../mat_mul_v3/examples/test_aclnn_matmul.cpp).
Allocate and initialize inputs and output **outside** the timed region,
reuse one stream and preallocated workspace, and record both

1. device-event elapsed time for batches of repeated launches, divided by
   the number of launches; and
2. per-kernel device time from a short `msprof` run with task-time and AIC
   metrics enabled (for a supporting installed version, for example,
   `msprof --output=<dir> --task-time=on --ai-core=on <runner> ...`).

Warm up until timing stabilizes. Then collect at least five independent
batches per point, each long enough that event resolution is negligible;
save every raw batch rather than only the mean. Randomize the order of
test points, repeat the baseline at the beginning and end, and flag runs
with concurrent NPU activity or thermal/clock drift. Use the profiler on
short diagnostic runs because profiling can perturb the timed run. Check
output correctness on representative cases outside timing. Do not time
host allocation, tensor construction, first-time compilation, H2D/D2H
copies, or stream synchronization as part of the kernel interval.

The BASE validation runner should accept a row of `(M,N,K,dtype,tiling)`
and write a machine-readable record. To compare two tilings at the same
shape, add a *temporary research-only* host-tiling override that accepts
specific source-legal fields **before** the normal finalizer. The normal
path must still reject illegal tuples. If this cannot be built immediately,
measure only auto-selected plans and label the dataset “observational”; do
not claim a controlled tile comparison.

## 4. Probe the primitives

Run these as separate experiments, preserving the actual kernel and memory
path. A short pilot can use 1, 2, 4, 8, 16, and (if available) 20 AICs.

1. **Launch:** Run a minimal no-op or minimal-work kernel with the same
   launch mode as BASE. Sweep launch count. The intercept is only a
   candidate for $\lambda_{\rm launch}$, because even a no-op kernel has
   some device work. Cross-check it against the small-work BASE intercept.
2. **Cube:** Use resident A/B tiles and repeat Cube accumulation many times
   per launch while keeping GM traffic approximately fixed. Sweep K-repeat
   count and valid $B_M,B_N$, then AIC count. Fit a slope in FLOPs and test
   whether aggregate rate is approximately $cR_C$. Retain per-shape rates
   if tail or tile size materially changes utilization. Look for a
   nonzero per-Cube-block startup: the current `SolveBase` equation requires
   zero fixed Cube latency, so a significant startup calls for a model
   change rather than forcing it into $R_C$.
3. **Read:** Use a GM→L1 path matching BASE's operand loads. Vary total
   transferred bytes **and** the number of read packets at fixed bytes so
   that $1/R_R$ and $\lambda_R$ can be separated. Sweep working sets below
   and above the measured L2 size and vary addresses to distinguish cache
   hits from main-memory reads. The current $D(g)$ assumes no cache reuse;
   if measured BASE runs hit L2, document that as a model discrepancy.
4. **Write:** Measure the output path used by BASE, including L0C/Fixpipe
   conversion when present. Vary output bytes and the number of output
   tiles independently. A generic vector store is not a substitute for
   that path. Check whether one $R_W,\lambda_W$ pair fits multiple output
   layouts and tile sizes.

For a primitive probe with independent work amount $x$ and event count $n$,
fit $t=\alpha+x/R+n\lambda$ using nonnegative $\alpha,\lambda$ and
positive $R$. Keep confidence intervals from run-to-run variation. If
changing $n$ also changes utilization or cache state, stratify the fit or
redesign the probe instead of interpreting that effect as latency.

## 5. Validate the BASE objective without tuning it on validation data

Hold out a matrix of FP16 ND shapes spanning small/large $M,N$ and K, with
`K <= 65535`, emitted `nd2nzA=nd2nzB=0`, and one consistent BASE
subpath. A first pilot grid is $M,N\in\{256,512,1024,2048\}$ and
$K\in\{256,1024,4096,8192\}$, filtered by memory budget and actual BASE
path. Add non-divisible M/N tails and asymmetric shapes afterward. For
several shapes, test source-legal variants of `singleCoreM/N`,
`baseM/N/K`, and occupied core count so that Cube-bound, GM-bound, and
transition cases are all represented. Use measured parameters unchanged.

For each measured plan, compute $F,D,S,N_o,P_B$ from the **emitted** tiling.
Compare predicted vs measured time and, more importantly, compare the
ranking and regret of the selected tiling on the same shape. Plot residual
against $M,N,K,c,B_M,B_N,B_K$, packet count, and working-set size. A
consistent error pattern is evidence that the traffic count, overlap rule,
cache policy, or source-path mapping needs revision; it should not be
absorbed into a per-candidate correction. A useful initial acceptance gate
is stable timing batches plus low held-out ranking regret (for example,
median below 10% and 90th percentile below 20%); revise that gate based on
the intended optimization benefit and measured noise.

Only after this round should we add the `K>65535` A ND→NZ conversion
probes ($R_{\rm ND\to NZ}$ and fixed/sync costs), then BF16 and additional
source paths. The solver currently requires positive profile entries for
`vector_add`, `sync`, and `nd2nz` even when this first BASE dataset never
uses them. Leave those fields marked provisional rather than reporting
them as measured BASE parameters.

## Deliverables and data format

Store one raw CSV/JSONL row per timing batch with at least:

```text
run_id,device_id,soc,driver,cann,source_commit,kernel_family,tiling_key,
M,N,K,dtype,layout,transA,transB,bias,usedCoreNum,singleCoreM,singleCoreN,
singleCoreK,baseM,baseN,baseK,stepM,stepN,stepKa,stepKb,depthA1,depthB1,
iterateOrder,l2_mode,nd2nzA,nd2nzB,workspace_bytes,warmup,batch_index,
launches_in_batch,device_elapsed_us,profiler_kernel_us,temperature,power
```

Keep a separate table for primitive probes with `probe_kind`, transferred
bytes or FLOPs, packet count, repeats, AIC count, working-set size, and raw
timings. Deliver the fitted profile with units, uncertainty intervals, fit
script/version, and the held-out prediction/ranking report. Then replace the
illustrative values in a **new measured profile file**, leaving examples
clearly illustrative and preserving raw data for re-fitting.

## First session after SSH access

1. Capture `npu-smi info`, board information, CANN/driver versions, and
   available AIC/AIV counts and memory capacities. Verify the device is
   idle and the installed CANN can build this repository's `mat_mul_v3`.
2. Build and run one FP16 ND BASE smoke case; save the emitted family and
   full tiling tuple. Confirm the timing harness can report device-event
   time and that a short `msprof` run identifies the same kernel.
3. Run a tiny pilot of 3–5 cases twice to estimate timing variability.
   Only then launch the larger primitive sweeps.

Huawei's [event timing API](https://www.hiascend.com/doc_center/source/en/canncommercial/601/inferapplicationdev/aclcppdevg/aclcppdevg_03_0054.html),
[task profiling command](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0011.html),
[platform resource APIs](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1026.html),
and [profile data fields](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/msProf/docs/en/user_guide/profile_data_file_references.md)
should be checked against the versions actually installed on the machine.
