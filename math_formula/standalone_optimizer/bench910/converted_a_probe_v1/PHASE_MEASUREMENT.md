# In-context AIC phase timing for converted-A MatMulV3

This experiment times AIC block zero in the source BASE and deterministic
Split-K kernels at `M=2560`, `N=1024`, FP16, K=65536 or 73728 on 910B3.
An isolated copy of the selector source was patched with
[`patch_phase_probe.py`](patch_phase_probe.py), built as a source-only JIT
operator package under
`/home/xianfei/solvebase_bench_20261025/ops_nn_phase_probe_install`.
The selector production checkout and installed CANN package were not
modified. The [four-device runner](phase_sweep.py) and
[compact raw summary](phase_sweep_summary.json) make the timings auditable.

The patch samples `GetSystemCycle()` on AIC block zero before waiting for
the vector conversion handoff, immediately after that wait, and after
its AIC matmul function. It prints the two differences only after both
intervals have ended. `wait` therefore includes pre-handoff AIC work and
the AIV conversion/barrier; it is **not** a pure AIV conversion duration.
`post-wait AIC` includes operand reads, Cube work, partial/output writes,
and synchronization inside the AIC matmul function; it is **not** pure Cube
instruction time. In Split-K it ends before vector reduction finishes.

The installed CANN code uses 50 system cycles per microsecond to convert
`GetSystemCycle()` to time. This scale is corroborated by the BASE
K=65536 phase sum (3.273 ms) versus its previously measured full kernel
(3.280 ms). Each of the four cases ran once on each of devices 4–7. Every
run used one correctness call, five warmups, and eight measured calls;
the last eight phase markers were analyzed. All 16 output checks had zero
relative error, the parsed source tile matched, and the devices were idle
before and after their runs. Every run emitted exactly 14 phase markers,
matching its correctness call, warmups, and measured calls. Table entries
are medians of four per-device
eight-call medians; ranges span those four medians.

| K | Family | AIC wait µs | Post-wait AIC µs | AIC phase sum µs | Prior full kernel µs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 65536 | BASE | 1117.60 (1112.26–1121.80) | 2157.56 (2146.25–2179.34) | 3273.44 | 3280.26 |
| 65536 | Split-K | 1177.74 (1174.48–1184.68) | 1575.56 (1568.27–1589.07) | 2756.61 | 2795.05 |
| 73728 | BASE | 850.89 (843.62–860.28) | 2389.62 (2382.06–2395.46) | 3237.50 | 3276.23 |
| 73728 | Split-K | 905.71 (897.47–912.34) | 1715.65 (1703.79–1724.78) | 2621.33 | 2653.41 |

BASE used `singleCoreM×N=128×256`, `baseM×N×K=128×256×64`, key 0.
Split-K used `singleCoreM×N=2560×384`, `baseM×N×K=128×128×128`, stepK=3,
key 48. Both had 20 AICs, A conversion, and no B conversion. At a fixed
K, Split-K's AIC wait is only 55–60 µs longer than BASE's, whereas its
post-wait AIC interval is **582 µs shorter at K=65536** and **674 µs
shorter at K=73728**. The measured full-kernel advantages of Split-K are
485 and 623 µs respectively. The family-ranking error is therefore
concentrated in the AIC matmul stage, not the shared conversion primitive.

## BASE objective against the measured phases

The current common `ConversionUs` term is 322.75 µs at K=65536 and
363.04 µs at K=73728. Subtracting it from the source-tile BASE objective
gives the modeled `BaseStageUs` shown below. These are only approximate
phase comparisons: the model's common conversion term is not identical to
the measured AIC wait, and its stage term can include work outside the
block-zero AIC interval.

| K | Modeled conversion µs | Measured AIC wait µs | Modeled BASE stage µs | Measured post-wait AIC µs |
| ---: | ---: | ---: | ---: | ---: |
| 65536 | 322.75 | 1117.60 | 1987.59 | 2157.56 |
| 73728 | 363.04 | 850.89 | 2235.51 | 2389.62 |

At K=73728, the wait exceeds the modeled common conversion by 487.85 µs
and the post-wait AIC interval exceeds the modeled BASE stage by
154.11 µs. Together they explain most of the 677.68 µs full-kernel
underestimate; the remaining difference also includes block-zero versus
whole-kernel timing and marker boundaries. At K=65536 the corresponding
gaps are 794.85 and 169.96 µs. Improving the BASE estimate therefore
requires a source conversion-tile and handoff model as well as a check of
the BASE read/Cube overlap; changing only the shared conversion term
cannot repair the family ranking.

## Reduced-objective diagnosis

The previous `SplitStageUs` charged every source base-packet read with a
fixed generic `gm_read` latency. On the source `2560×384` Split-K tile,
the memory branch is larger than the Cube branch for each of the three
N panels. Recomputing its documented terms gives:

| K | Modeled GM payload service µs | Modeled per-packet read startup µs | Modeled panel-memory sum µs | Measured post-wait AIC µs |
| ---: | ---: | ---: | ---: | ---: |
| 65536 | 1447.01 | 947.12 | 2394.13 | 1575.56 |
| 73728 | 1627.88 | 1052.35 | 2680.24 | 1715.65 |

At K=73728 the model gives a busiest AIC 30 base-K packets per panel.
It counts 60 spatial base packets in each full 384-column panel and 40
in the 256-column tail. Thus the fixed startup contribution is
`(2×60×30 + 2×60×30 + 2×40×30) × 0.109620 µs = 1052.35 µs`.

The fixed startup term alone is close to the model's excess. Omitting it
*as a diagnostic*, the modeled payload service is within 129 µs and 88 µs
of the measured AIC interval. This does not prove the true startup cost is
zero: source L1 step transfers, DMA pipelining, L2 hits, and overlap can
hide or combine base-packet startups. It does show that adding one generic
latency for every base packet on the critical path is a poor representation
of this source Split-K path.

## L1-step startup revision

On the measured converted-A path (`K>65535`), the reduced objective now
counts each operand's fixed read startup once per L1 K step:
`2 × spatial packets × ceil(qcrit / stepK) × λread`. The unconverted
path retains its prior effective per-baseK startup. The bytes term remains
unchanged. The source uses `stepKa=stepKb=3` on the converted-A tiles,
so the K=73728 startup subtotal becomes **350.78 µs** rather than
1052.35 µs. The reproducible source-tile diagnostic is
`mmopt_full_solver_audit converted_a_source`.

| K | Prior BASE model µs | Revised Split-K model µs | Measured BASE µs | Measured Split-K µs |
| ---: | ---: | ---: | ---: | ---: |
| 65408 | 1983.72 | 2418.87 | 2315.70 | 1623.99 |
| 65536 | 2310.34 | 2113.03 | 3280.26 | 2795.05 |
| 73728 | 2598.55 | 2369.29 | 3276.23 | 2653.41 |

The BASE objective is unchanged; its measured source tiles are listed to
show the resulting family choice. The revised objective selects the
measured faster Split-K family for the **two converted-A cases**. It still
chooses the wrong family at K=65408, where A is unconverted. Absolute
times remain inaccurate, especially at K=65536. The model still treats
the source conversion tile as a common byte-rate term, whereas the source
changes its `baseAN×baseAD` tile
between the two converted-A cases.

Applying L1-step grouping to *all* Split-K paths initially lowered the
mean absolute timing error from **490.64 µs to 140.77 µs** on four
recorded unconverted-A source tiles. But it worsened the more important
family decision on a separate 16-shape held-out suite: correct-family
count fell from **12/16 to 8/16**, and geometric mean speedup over
default fell from **1.0336× to 0.9747×**. Two new regressions appeared
where master had chosen single-core mode 6. The final change therefore
groups read startup only on the measured converted-A path. Replaying
all 16 held-out inputs gives byte-for-byte identical model predictions
to the prior version. These checks reuse earlier device measurements;
no new device run was made for this revision. Source-tile scoring required
a diagnostic 2-GiB workspace cap because the K=73728 source Split-K
tuple needs 530 MiB, above the
normal 512-MiB model cap. The online selector also searches different
geometries, so a correct ranking of these two source tuples does not yet
establish a correct online decision.

## Why the optimized `singleCoreM` can be too small

The reproducible `mmopt_full_solver_audit converted_a_core_m` diagnostic
holds the FP16 workload at `2560×1024×73728`, with 20 AICs and
`baseM=baseN=baseK=128`. It varies only the outer M/N tile and K step:

| stepK | singleCoreN | singleCoreM=128 model µs | singleCoreM=2560 model µs | Workspace at M=2560 | Fits 512 MiB? |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 128 | 3035.62 | 3035.62 | 450,887,680 B | yes |
| 3 | 128 | 2369.13 | 2369.13 | 450,887,680 B | yes |
| 3 | 384 | 2369.29 | 2369.29 | 555,745,280 B | no |

For the NK-order branch (`M>N`), `SplitStageUs` uses the full M extent
in each N panel. If `singleCoreM` and `baseM` divide M, then
`AxisPackets(M,singleCoreM,baseM)=M/baseM` regardless of
`singleCoreM`. The Cube work, modeled operand bytes, fixed read count,
and vector shard also remain identical. The workspace formula
`W=W0+8×cores×singleCoreM×singleCoreN` is only a feasibility gate.
Consequently the reduced objective **cannot identify an optimal outer M
tile** among these feasible values; the one-hot solver can return any
cost-tied geometry. A small returned M is not evidence that small M is
faster.

MatMulV3's source NK tiler starts from `singleCoreM=M` and
`singleCoreN=3×baseN`, then splits M only when its L2 rule requires it.
For this source tile, the rule's M limit is about 2930, above M=2560,
so the source emits 2560×384. The standalone default domain omits
`singleCoreN=384`, and that full source tuple exceeds its configured
512-MiB workspace cap. Neither restriction means the emitted source
tile is invalid: the benchmark allocates its actual 530-MiB workspace.
To optimize an executable tile, the candidate domain must represent the
source branch and configured workspace, and the objective needs a
measured outer-tile term before it can rank different `singleCoreM`
values by runtime.

## Current standalone family decision versus the 910 branch

The local `mmopt_full_solver_audit converted_a_select` diagnostic uses
the measured-output primitive profile and a 512-MiB workspace budget.
With the online-style two-buffer BASE L0C preflight it estimates
**3006.17 µs BASE**, **2369.13 µs deterministic Split-K**, and
**4711.74 µs single-core mode 2**; modes 5 and 6 are infeasible.
The resulting local winner is deterministic Split-K, matching the
source default's family. Allowing the source's one-buffer BASE L0C lowers
its estimate to 2598.55 µs, still above Split-K's estimate.

At the time of the phase probe, this local result was not installed in
the 910 online selector. The subsequent
[20-shape source-backed comparison](../online_grouped_vs_master_v1/README.md)
copied only this converted-A read change to the uncommitted experimental
branch working tree, built a separate package, and measured it against
master. On this target shape the revised selector chose deterministic
Split-K and emitted the same source tile as master.

The instrumented task includes a diagnostic `printf` after the AIC timing
markers, so its host event duration is deliberately excluded. The
prior full-kernel times above come from uninstrumented packages and are
included only for scale and family comparison. AIC block zero may not be
the slowest block, and Split-K vector reduction runs beyond its AIC marker.
