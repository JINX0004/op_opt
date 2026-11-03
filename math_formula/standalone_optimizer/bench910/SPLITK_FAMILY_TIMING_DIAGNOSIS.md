# Split-K family timing diagnosis on 910B3

This document decomposes the **frozen pre-change objective**. The
[mode-6 output-cost change](MODE6_OUTPUT_COST_CHANGE_910B3.md) records
the updated provisional objective and its comparison with these runs.
The later [device-4 output measurement](MODE6_OUTPUT_MEASUREMENT_910B3.md)
shows that barrier removal and address reuse have little effect in an
isolated stream; the source kernel reports far lower Fixpipe active
time while MAC and MTE2 are active. The possible causes below were
pre-measurement hypotheses, not established hardware mechanisms.

## Measured versus estimated runtime

The [device-4 experiment](SPLITK_FAMILY_SELECTION_910B3.md) executed
FP16 NN MatMulV3 kernels on an idle 910B3. All 32 timed tasks passed
output, emitted-tiling, and device-occupancy checks. This table scores
the **exact source-emitted tiles** with `ReducedObjectiveUs`, so the
discrepancy is present before the MIP searches for tiles. `S` is
single-core mode 6 and `D` is deterministic mode 3; times are µs.

| M × N × K | S predicted | S measured | D predicted | D measured | Predicted S−D | Measured S−D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 × 2048 × 27392 | 2502.7 | 1072.6 | 1668.6 | 1251.8 | +834.1 | −179.2 |
| 2048 × 3072 × 27392 | 3763.7 | 1596.9 | 2489.2 | 1819.3 | +1274.5 | −222.3 |
| 2048 × 2048 × 32768 | 2972.5 | 1449.7 | 2026.9 | 1717.5 | +945.7 | −267.8 |
| 2048 × 3072 × 32768 | 4470.5 | 2091.1 | 3026.7 | 2460.3 | +1443.8 | −369.2 |

The model overpredicts S by 2.05–2.36× and D by 1.18–1.37×. The
family-ranking error is mainly the unequal overprediction of mode 6.
The four N=768 cases are omitted from this table because their forced
source mode-6 tiles fail the model's spatial host gate. One of those,
2048 × 768 × 32768, measures S=654.2 and D=747.2 µs, yet both source
default and MIP choose D: a separate eligibility/selection issue.

Evidence: [device-4 summary](splitk_family_selection_device4_v1/summary.json),
[source-tile audit](splitk_family_selection_device4_v1/audit), and
[native geometry scores](splitk_family_source_geometry_scores.csv).
[`diagnose_splitk_family_timing.py`](diagnose_splitk_family_timing.py)
reproduces the [objective decomposition](splitk_family_mode6_diagnostics.json)
and asserts its totals match the native objective within 0.001 µs.

## Mode-6 objective and leading discrepancies

The wide source tiles use all 20 AICs, with one output tile per AIC:
`singleCoreM=208`, `singleCoreN=1024` or `1536`, and
`singleCoreK=512`. Each AIC executes 54 or 64 K packets.
[`ScKChainUs`](../mip_solver.cpp) charges a steady packet approximately
as

```text
T_packet = max(T_cube, T_FP32_store + T_atomic_extra, T_one_piece)
           + max(0, T_A_copy + T_B_copy − T_last_N_cube_overlap).
```

For 2048 × 2048 × 32768, the per-512-K-packet terms are:

| Term | Model time (µs) |
| --- | ---: |
| Cube | 14.82 |
| Plain FP32 Fixpipe store | 28.81 |
| Incremental FP32 atomic add | 8.86 |
| MAC/write critical term | 37.67 |
| A+B GM→L1 copy | 10.68 |
| Copy overlap credited | 1.85 |
| Residual copy added to K chain | 8.83 |

The objective therefore charges ~46.5 µs per steady packet. The entire
measured kernel takes 1449.7/64 = 22.65 µs per K packet, **including**
startup, tail, and AIV completion. The modeled MAC/write term alone
exceeds this observed average. At the 1536-wide tile, the model charges
56.50 µs MAC/write plus 13.46 µs residual copy, whereas the measured
whole-kernel average is 32.67 µs per packet. This establishes that a
serial per-packet service assumption does not transfer to the source
kernel; it does not by itself identify which hardware effect is dominant.

### 1. The isolated FP32 output service is not the in-kernel demand

[`ScMode6MacUs`](../mip_solver.cpp) applies the FP32 store coefficient
and atomic increment to every output tile on every K packet. The
[primitive probe](primitives/probe_kernel.cpp) writes successive packets
to **distinct output addresses**, with `PipeBarrier<PIPE_FIX>()` after
every Fixpipe. The [source mode-6 kernel](../../../mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel_gm_to_l1.h)
sets `SetAtomicAdd<float>()` after the first K packet, uses L0C
ping-pong events, and Fixpipes later packets to the **same FP32 output
workspace**. The probe's 34,190 bytes/µs plain-store slope is valid for
its serialized schedule, but is not an isolated steady-state service
rate for this source schedule. The [follow-up measurement](MODE6_OUTPUT_MEASUREMENT_910B3.md)
found that repeated versus distinct addresses and removal of the
per-Fixpipe barrier change isolated stream time very little. In the
real kernel, `msprof` reports only about 9/14 µs of Fixpipe active time
per K packet at the two wide tile widths, below MAC and MTE2 active
time. The isolated stream's ~34/52 µs per packet cannot be charged as
the source's exposed output service. The exact scheduling mechanism
behind the difference has not been isolated.

A sensitivity calculation substitutes the generic GM-write coefficient
for `sc_fp32_store` while retaining the measured atomic increment. It
reduces the 2048² × 32768 estimate from 2972.5 to 1519.1 µs, near the
1449.7 µs measurement; the four wide estimates fall by about 49%. This
**does not validate** the generic coefficient for FP32 Fixpipe. It shows
that the output term is large enough to explain most of the ranking
failure. Removing atomic increment alone still predicts 2414.6 µs on
that shape, so atomic arithmetic alone does not explain the gap.

### 2. The model likely undercredits copy/MAC overlap

[`ScMode6ReadUs` and `ScKChainUs`](../mip_solver.cpp) add the A and B
copy services and credit only one last-N Cube stripe (1.85 µs here) as
overlap. The [copy probe](singlecore_copy/probe_kernel.cpp) serializes
each MTE2 copy with `PipeBarrier<PIPE_MTE2>()`; the source prefetches B
for later N/K work and A for later K work while MAC advances. The
isolated serial rates therefore do not measure the exposed copy time in
the source dependency graph. Even granting complete overlap of the
modeled residual copy alone leaves 2416.5 µs for 2048² × 32768. Copy
overlap is material but cannot close the gap under the present output
service coefficient.

The wide A+B operands exceed the reported 192 MiB L2 (about 214–320
MiB depending on K and N). The copy slope used by this objective came
from L2-resident probes; the [above-L2 probe](SINGLE_CORE_SPLITK_PRIMITIVES_910B3.md)
found a slower, variable B path. This limits extrapolation, but a slower
copy slope by itself would **increase** an already high mode-6 estimate.
The modeled AIV cast is only 1.3–1.9 µs per source tile, far too small
to explain the 1–2 ms gap. The ablations cannot simply be added: using
both generic output service and full copy overlap drops 2048² × 32768
to 963.0 µs, below measurement.

## Deterministic mode-3 objective

[`SplitStageUs`](../mip_solver.cpp) builds six M panels per wide source
geometry and charges each full panel

```text
T_panel = max(T_cube,
              T_operand_bytes / R_read +
              2 × spatial_packets × critical_K_packets × L_read).
```

It overlaps each panel's AIV reduction with the next panel's Cube/read
stage. For 2048² × 32768, full-panel Cube is 205.2 µs, the read-byte
term is 217.1 µs, and repeated generic read fixed latency is 157.9
µs. That latency is a substantial part of the 2026.9 µs objective versus
1717.5 µs measured. Removing it entirely gives 1185.0 µs, an
underestimate; an effective pipelined packet issue cost is more likely
than either full serial charging or zero cost. The source also ping-pongs
workspace and UB buffers with AIC/AIV events, so this attribution
remains a hypothesis until measured in a source-shaped panel experiment.

## Recommended next measurements and correction

1. Probe the source's 208×1024 and 208×1536 FP32 Fixpipe streams on 20
   AICs, comparing distinct versus repeated output addresses, plain
   versus atomic, and serial barriers versus source-like L0C events.
   Match the wide-case workspace footprint.
2. Probe source-shaped A/B prefetch with MAC and Fixpipe active. Disable
   each copy stream in turn to identify **exposed** copy time. Include
   resident and above-L2 operands.
3. Fit a mode-6 steady-state K-packet initiation interval and separate
   first/tail costs from those measurements, then update the recurrence.
   Measure the deterministic panel issue cost separately. Keep the model
   analytical rather than adding a whole-kernel fitted correction.
4. First validate the S−D sign and rough gap on held-out source tiles;
   only then use MIP-selected tiles. Where measured uncertainty overlaps
   the family gap, retain the source family decision.

These results cover FP16 NN mode 6 versus mode 3 below K=65535; they do
not validate modes 2/5, BF16, other layouts, or long-K conversion.
