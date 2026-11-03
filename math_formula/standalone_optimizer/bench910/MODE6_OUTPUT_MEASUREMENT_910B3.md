# Mode-6 FP32 output measurement on 910B3, device 4

## Measurement question

The previous mode-6 objective multiplied a serial FP32 Fixpipe slope by
every K packet. The [device-4 family experiment](SPLITK_FAMILY_SELECTION_910B3.md)
showed that this overpredicted mode 6. This study measures both an
isolated FP32 output stream and the Fixpipe pipeline **inside the real
source kernel**. The two must be distinguished: an isolated stream's
wall time is not automatically its exposed service in a MatMul pipeline.

Hardware and software: Ascend 910B3 device 4, 20 AICs, CANN 9.0.0 from
`/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0`.
Device 4 was idle before and after each sweep round. The source-kernel
profiles came from the earlier correctness- and occupancy-checked
FP16 NN experiment; two new 2048×4096×32768 source runs passed their
output checks with zero relative error. The initial build against
`/usr/local/Ascend/ascend-toolkit/latest` failed at CANN object merge.
Rebuilding with the same CANN 9.0.0 package and Release configuration
as the earlier primitive probes succeeded.

## Isolated output stream

The [probe kernel](mode6_output/probe_kernel.cpp) assigns 20 AICs the
source's 208-row spatial partition, including the 176-row tail, at
global N=2048 or 3072. It precomputes two L0C 128×128 tiles, then
Fixpipes them to the source-shaped FP32 output layout with repeated or
distinct K destinations. The [runner](mode6_output/probe_runner.cpp)
checks the completion token and expected FP32 atomic sum after every
run. The [sweep](mode6_output/sweep.py) uses two randomized rounds,
three ACL event batches of eight launches per condition, and 10 warmups;
one `msprof` check agreed with ACL event timing. These are
**output-only** streams: Cube and MTE are not active during the K loop.

From the K=16→64 slope, in µs per 512-K output packet:

| Output tile per full AIC | Plain, no barrier | Atomic, no barrier | Atomic increment | Atomic, barrier after each Fixpipe |
| --- | ---: | ---: | ---: | ---: |
| 208×1024 | 24.26 | 34.38 | 10.12 | 34.63 |
| 208×1536 | 35.94 | 52.01 | 16.07 | 51.94 |

At K=4, atomic repeated versus distinct destinations differed by only
0.53 µs for 208×1024 and 0.25 µs for 208×1536 **per whole launch**.
Removing the per-Fixpipe barrier also changed the long-K atomic slope
by under 0.3 µs per packet. Thus neither address reuse nor that barrier
alone explains why the earlier serial coefficient failed to predict
the source kernel. All [68 sweep records](mode6_output/output_sweep_device4.jsonl)
passed the probe's output checks.

## Source-in-context Fixpipe demand

[`extract_source_pipeline.py`](mode6_output/extract_source_pipeline.py)
reads the final eight `msprof` tasks from each source run's
`op_summary` and saves the reported AIC pipeline active times in
[16 source records](mode6_output/source_pipeline_device4.jsonl).
For the four wide shapes, one spatial tile is assigned to each AIC.
M-tail AICs are slightly smaller; fitting against the full critical
tile gives the following approximate demand:

$$
T_{\mathrm{FIX,active}} \simeq
\frac{4\,m_{\rm core}n_{\rm core}\,P_K}{R_{\rm FIX,mode6}},
\qquad R_{\rm FIX,mode6}=90{,}959\ \text{bytes/µs per AIC}.
$$

The fit sets the packet fixed term to zero. A four-shape cluster
bootstrap gives 89,666–92,280 bytes/µs (a descriptive interval for
these shapes, not a general hardware guarantee). Using K=27392 alone
predicts the K=32768 Fixpipe active times within 3.8%.

| Source shape, M×N×K | K packets | Reported Fixpipe active (µs) | Per K packet (µs) | MAC active (µs) | MTE2 active (µs) | Kernel (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048×2048×27392 | 54 | 501.6 | 9.29 | 821.3 | 899.1 | 1072.6 |
| 2048×2048×32768 | 64 | 612.2 | 9.57 | 983.3 | 1225.5 | 1449.7 |
| 2048×3072×27392 | 54 | 745.8 | 13.81 | 1231.8 | 1332.7 | 1596.9 |
| 2048×3072×32768 | 64 | 903.9 | 14.12 | 1474.4 | 1774.1 | 2091.1 |

The source's Fixpipe active demand is **below its MAC active demand**
on all four shapes and far below its MTE2 active demand. These pipeline
times overlap and must not be summed. The isolated stream costs about
34/52 µs per K packet, while the source reports about 9/14 µs of
Fixpipe active time per packet. This large difference survives the
barrier and address-layout controls above. The precise source scheduling
feature responsible is not isolated by this experiment; an
output-only rate is therefore unsuitable for the source objective.

### Independent shape check

The new [2048×4096×32768 source run](mode6_output/holdout_2048x4096x32768_device4.json)
emitted mode 6 with `singleCoreM=208`, `singleCoreN=2048`, 64 K packets,
and 20 output tiles. Across two rounds of eight kernel tasks, its median
Fixpipe active time was **1226.4 µs**. The 90,959-byte/µs rate predicts
**1198.9 µs**, 2.2% low. The verified full kernel took 2706.2 µs.
This validates the output-demand rate for
one larger tile; it does not validate the entire kernel objective. At
that exact source tile, the revised full objective is **3089.5 µs**,
14.1% above the measured kernel time.

## Model use and limit

The device-4 exploratory profile now supplies

```cpp
h.primitives["sc_fp32_store_steady"] = {"bytes", 0.0, 91000.0};
```

For mode 6, this is the **total** later-K FP32 Fixpipe/atomic pipeline
demand. [`ScMode6AtomicOutputUs`](../mip_solver.cpp) uses it **instead
of** adding `sc_atomic_extra`, which was measured in a serial probe.
The first plain K packet retains `sc_fp32_store`. The measured demand
enters the output throughput and one-piece bounds of `ScMode6MacUs`;
it is not a fitted whole-kernel correction.

At the four wide source tiles, the resulting mode-6 objectives are
1286, 1939, 1533, and 2311 µs, respectively, versus measured 1073,
1597, 1450, and 2091 µs. They retain the correct family ranking on
these four cases. The full MIP still chooses deterministic for the four
narrow cases and mode 6 for the four wide cases, matching the faster
executed branch on 7/8 shapes, the same as source default. See the
[measured-output source scores](splitk_family_source_geometry_scores_measured_output.csv)
and [MIP predictions](splitk_family_model_predictions_measured_output.csv).
These are **in-sample** family checks; the new holdout checks only the
Fixpipe demand. Remaining mode-6 runtime error lies in other service
or overlap terms, especially MTE2/panel scheduling. A broader
held-out family-decision suite is needed before replacing the source
tiler.

The [analysis script](mode6_output/analyze.py) reproduces the fits,
slopes, and holdout calculation from the saved raw data; its
[output](mode6_output/analysis_device4.json) contains the unrounded
values.
