# FP32 BASE narrow-N MTE2 service (910B3, CANN 9.0.0)

## Question and result

The FP32 BASE tile `singleCoreM×singleCoreN×baseK = 256×64×32` for
`4096×64×27456` was scored at 342.7 µs by the selector but took 522.5 µs
when executed. The profiler reported 99.2% MTE2 utilization, and its
GM→L1 byte count equaled the estimator's issued-byte formula. The missing
term is a slower **service time for those requests**, not missing volume.

An opt-in [source-model profile](../fp32_mode2_v1/remote_selector/model/hardware_profile.h)
now adds a shared MTE2 bound and a busiest-core bound for an experimentally
supported access class. It scores the executed tile at **520.9 µs** and the
source-generated `128×128×64` BASE tile at **525.7 µs**, versus **522.5 µs**
and **531.8 µs** measured. The default MatMulV3 FP32 source path remains
unchanged; the CLI and opt-in host path now use this profile.

## Formulation

For one BASE geometry, the issued GM→L1 bytes are

$$
D_A=b_A K M\left\lceil\frac{N}{B_N}\right\rceil,\quad
D_B=b_B K N\left\lceil\frac{M}{B_M}\right\rceil,\quad D=D_A+D_B.
$$

For the abstract balanced source assignment, let $a_i$ and $b_i$ be
the sums of actual A-row and B-row extents of the spatial packets
assigned to core $i$. If the source L1 A step spans
$S_A=\mathrm{stepKa}\,B_K$ columns, that core requests

$$
D_i=K(b_Aa_i+b_Bb_i),\qquad
W_i^A=\left\lceil\frac{K}{S_A}\right\rceil a_i
$$

bytes and A row windows. The new lower bound is

$$
T_{\mathrm{MTE2}}=\max\left\{
\frac{D}{R_{\mathrm{shared}}},
\max_i\left(\frac{D_i}{R_{\mathrm{core}}}+\tau_A W_i^A\right)
\right\}.
$$

`BaseStageUs` takes the maximum of this and the existing Cube and
read/packet bounds before adding any source FP32 call/atomic terms. The
rates are **effective in-kernel services** with Cube work in flight,
inferred from measurements. They are not isolated hardware bandwidths
or a constant runtime correction:

| Parameter | Experimental value |
| --- | ---: |
| $R_{\mathrm{shared}}$ | 1,290,000 bytes/µs |
| $R_{\mathrm{core}}$ | 119,000 bytes/µs |
| $\tau_A$ | 0.00405 µs per A row window |

The term activates only for FP32 direct-copy BASE, `coreM=baseM`,
`coreN=baseN`, `N∈{64,128}`, A footprint greater than 256 MiB, a row pitch
**not** divisible by 16 KiB, and no feasible four-by-four source L2
region under its 100 MiB target. These conditions delimit the measured
regime; 256 MiB is not presented as the physical L2 size. The existing
32/64 KiB aligned-pitch diagnostic bound still applies independently.

## Measurements and validation

The [grid harness](measure_grid.py) forces the source BASE family,
checks correctness, audits emitted tiling, and profiles the kernel on
devices 4–7. [Raw records](records.json),
[broader K holdouts](broad_holdout_records.json),
[new shape probes](final_holdout_records.json), and
[alignment holdouts](alignment_holdout_records.json) retain each run's
tiling, task time, and pipeline counters. The
[memory-counter comparison](memory_records.json) covers N=64/96/128 at
`M=4096,K=32640`. The [evaluation script](evaluate.py) rebuilds the C++
scorer and writes the [per-case table](comparison.csv) and
[summary](comparison_summary.json).

| Evaluated set | Cases affected | Old MAPE on affected | New MAPE on affected |
| --- | ---: | ---: | ---: |
| Initial calibration | 2 | 45.2% | 0.7% |
| First holdouts | 4 | 34.2% | 2.3% |
| Broader K holdouts | 2 | 41.2% | 6.8% |
| New-shape development set | 5 | 37.9% | 8.1% |
| Independent alignment holdout, unaligned K=20160 cases | 2 | 34.4% | 3.3% |

The two newly held-out `K=20160` source tiles scored 386.8 and 514.8 µs,
versus 388.1 and 484.3 µs measured for N=64 and N=128. The earlier
`K=27392` and `K=32640` holdouts are also within about 0–9% where this
term activates. The first calibration row at `2048×64×27328` is below
the activation footprint and remains underestimated by about 30%.

## Limits discovered by the sweeps

The simple service is **not valid for all pitches**. With a 16 KiB aligned
A pitch, `5120×64×20480` took 381.6 µs while the general-pitch bound
would have scored 490.4 µs. At `3072×64×36864`, the same proposed
16 KiB class rate would have underpredicted a 649.4 µs run. A single
rate for 16 KiB alignment therefore failed its independent holdout;
the model does not use one.

`N=96` is a separate direct-copy regime: `4096×96×32640` took
1484.5 µs, compared with 597.2 µs for N=64 at the same M,K. The
GM→L1 requested-byte counters match the volume formula for both, and
N=96 had a higher L2 read-hit fraction. This excludes a missing-byte or simple
L2-capacity explanation. The source uses `baseK=32` for N=96 versus
`baseK=64` for N=64/128; its per-request service needs a separate probe.
N=80/112 take an ND→NZ B path and are not represented by the direct-copy
scorer.

These limits mean the profile supports experimental online FP32 family
selection under `MMV3_ONLINE_SELECTOR=1`, with the default source path
retained. They do not support enabling it by default.

## Rerun of the 100-case FP32 selector comparison

The [100-case scorer](score_100.cpp) reruns the same
[FP32 stress dataset](../fp32_100_rerun_20261106/dataset.json) with
`ExperimentalFp32BaseMte2910B3()` and `SourceCompatibleDomain()`.
The [decision table](selector_100.csv) contains all family scores and
proposed tiles. Relative to the [previous opt-in profile](../fp32_base_formula_v2/new_selector_decisions.csv),
only three decisions change, all from BASE to deterministic Split-K at
`4096×64×K`, K=27328/27392/27456. The other 97 family choices are
identical.

The [paired-run harness](measure_100_changes.py) freshly measured all
three changed choices against clean master on devices 4–6. It checked
package hashes, output correctness, source tiling, and device idleness,
then ran two `msprof` rounds with reversed arm order and five timed
repetitions after five warmups per round. The
[raw paired records](selector_change_pairs.json) retain the audit and
timing details. The [100-case analysis](analyze_100.py) reuses prior
audited pairs for the other 97 unchanged decisions; it does **not**
claim a fresh device run for all 100 shapes.

| Shape | Master family | New selector family | Master µs | Selected family µs | Master / selector |
| --- | --- | --- | ---: | ---: | ---: |
| 4096×64×27328 | BASE | deterministic | 538.32 | 482.44 | 1.116× |
| 4096×64×27392 | deterministic | deterministic | 488.09 | 490.60 | 0.995× |
| 4096×64×27456 | deterministic | deterministic | 489.52 | 491.00 | 0.997× |

Across the [100-case comparison](selector_100_comparison.csv), the
geometric mean master/selector ratio is **0.986×**, compared with
**0.984×** for the prior profile. The shape-bootstrap 95% interval is
**0.950–1.014×** and includes 1. There are 41 selector wins and 59
master wins; 24 cases differ in family from master. Thus the new
service fixes a local narrow-N choice but does not establish an
overall improvement on this stress set. These are kernel times with
source-generated tiling inside the selected family; they exclude
host-side selector time and do not time the model's proposed tile.
The [machine-readable summary](selector_100_analysis.json) records
the complete counts and interval.

## Opt-in MatMulV3 host smoke test

The 910 selector branch now uses `ExperimentalFp32BaseMte2910B3()` for
FP32 CLI rows and FP32 host decisions when `MMV3_ONLINE_SELECTOR=1`.
The MatMulV3 host gate accepts same-type FP32 NN/ND under that switch;
the switch-unset path still uses source selection. Mode 5 additionally
requires its source experiment gate, and mode 6 is excluded for FP32.

The rebuilt CANN 9.0.0 host library was placed in an isolated test
package. At `4096×64×27328`, the [device smoke record](online_smoke_record.json)
shows the same package emitted source BASE (`128×128×64`, 538.30 µs)
with the switch unset and deterministic Split-K (`4096×384×192`,
483.16 µs) with it set. Both arms reported zero output error.
The standalone CLI's 100 decisions and objective values match
[the previous profile sweep](selector_100.csv) exactly. This validates
the wiring and this one host case; the 100-case performance comparison
above remains a family-forcing experiment, not a full online-host
device rerun.
