# BF16 source-emitted BASE scoring and 100-shape device comparison

This experiment compares the revised online selector with clean MatMulV3
master on the same 100 BF16 shapes as the previous
[comparison](../bf16_100_vs_master_v1/README.md). A and B and output are
BF16, accumulation is FP32, inputs and output are NN ND/ND/ND, and there is
no bias. BF16 online selection remains opt-in.

## Model change

The source prepares a plain BASE block and buffer tile before online family
selection. The online selector now passes those dimensions to the reduced
objective and compares the cost of the tile the source can emit. For this
source BASE path, L0C uses one buffer; the model's general two-buffer
preflight would reject a valid 128×256 source tile. The source BASE score
therefore uses the one-buffer preflight. Other implementation candidates
retain their existing optimization and preflight.

For the measured BF16 narrow-B path (`baseN × sizeof(B) < 64` bytes), the
fixed A/B read startup is charged once per source L1 K-step:

\[
P_A=\left\lceil\frac{K}{\mathrm{stepKa}\,\mathrm{baseK}}\right\rceil,
\qquad
P_B=\left\lceil\frac{K}{\mathrm{stepKb}\,\mathrm{baseK}}\right\rceil.
\]

The variable-byte service and measured narrow-B row service remain as in
the prior model. Wider BASE tiles and FP16 retain their previous read
startup objective. The source-emitted `16×16×65408` BASE tile has
`baseK=64`, `stepKa=stepKb=64`; its revised reduced score is **533.88 µs**,
versus **754.44 µs** before the K-step correction and **599.28 µs**
measured in the previous source-backed diagnostic. This is still an
approximate timeline, but it no longer charges 1022 separate baseK read
startups where the source makes 16 L1 requests.

## Device protocol

The [runner](run.py) assigns 25 shapes each to Ascend 910B3 devices 4–7.
Every shape has an independent master and online output/tiling audit,
five warmups, and five profiled MatMulV3 kernel tasks. A tile or family
change, or a mode-5 arm, gets two rounds with reversed run order. Each
worker verifies device idleness and the exact host-library hashes before
recording a shape. The online package includes the same benchmark-only
BF16 dtype-inference proto patch as the clean master package. The patch
is not part of the selector branch.

- Clean master host SHA-256: `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`.
- Revised online host SHA-256: `1fd89d86dc5a7b37af3da56cbf0bad74da4be440d940de5b96faa8308d9ee4d1`.
- Source-backed raw traces: `/home/xianfei/solvebase_bench_20261025/cases/bf16_source_base_v1` on 910.

The [analyzer](analyze.py) checks distinct-shape coverage, paired rounds,
output error, and family/tiling consistency before reporting ratios. Its
speedup is clean-master kernel time divided by online kernel time; a value
above one favors online. Host selector overhead is outside the profiled
kernel time. The 100 shapes deliberately exercise implementation
boundaries, so their aggregate is descriptive for this dataset.

## Results

The [analysis](analysis.json) validates all 100 distinct shapes, with no
failed records. The maximum relative output error is **0.003115**, below
the benchmark's 0.02 bound. The [per-shape timings](comparison.csv) are
kernel times; they exclude host selector overhead. Speedup is master time
divided by online time.

| Family selection versus master | Shapes | Geometric mean speedup | Online faster / slower | Emitted tile identical |
| --- | ---: | ---: | ---: | ---: |
| Changed | 18 | **1.3494×** | 18 / 0 | 0 |
| Unchanged | 82 | 0.9975× | 36 / 46 | 82 |
| **All** | **100** | **1.0532×** | **54 / 46** | **82** |

Every changed-family choice was faster online in **both** reversed-order
rounds. The changed choices are 14 deterministic Split-K → BASE, three
source mode 6 → BASE, and one source mode 2 → deterministic Split-K.
The median shape speedup across all 100 is 1.0011×. Resampling these
**selected shapes** gives a 95% bootstrap interval of 1.0255–1.0858×
for their geometric mean. The shape set targets implementation boundaries;
that interval is not a population confidence interval for arbitrary
MatMul workloads.

### All 18 changed-family cases

The times below are the median of the two paired timing rounds for each
arm. The single-core tile is `singleCoreM×singleCoreN×singleCoreK`.
Speedup is master time divided by online time.

| Shape M×N×K | Device | Mode master→online | Single-core tile master→online | Master µs | Online µs | Speedup |
| --- | ---: | :---: | :--- | ---: | ---: | ---: |
| 1792×768×32768 | 5 | 3→0 | 1792×384×384 → 128×256×32768 | 568.15 | 483.00 | 1.176× |
| 2048×768×27392 | 6 | 3→0 | 2048×384×384 → 128×256×27392 | 459.99 | 388.40 | 1.184× |
| 2304×768×32768 | 7 | 3→0 | 2304×384×384 → 128×256×32768 | 897.86 | 516.45 | 1.739× |
| 2560×768×32768 | 4 | 3→0 | 2560×384×384 → 128×256×32768 | 1072.31 | 547.08 | **1.960×** |
| 2048×1024×32768 | 5 | 3→0 | 2048×384×384 → 128×256×32768 | 892.33 | 648.32 | 1.376× |
| 2304×1024×36864 | 6 | 3→0 | 2304×384×384 → 128×256×36864 | 924.32 | 817.62 | 1.131× |
| 2816×1024×36864 | 7 | 6→0 | 144×1024×512 → 128×256×36864 | 1204.27 | 1059.31 | 1.137× |
| 3072×1024×36864 | 5 | 6→0 | 160×1024×512 → 128×256×36864 | 1204.41 | 1086.06 | 1.109× |
| 2560×1024×32768 | 5 | 3→0 | 2560×384×384 → 128×256×32768 | 1305.80 | 772.62 | 1.690× |
| 2560×1024×49152 | 6 | 3→0 | 2560×384×384 → 128×256×49152 | 1414.75 | 1362.85 | 1.038× |
| 3072×1024×73728 | 6 | 2→3 | 160×1024×512 → 1536×384×384 | 4130.19 | 3706.59 | 1.114× |
| 768×2048×32768 | 7 | 3→0 | 384×2048×384 → 128×256×32768 | 624.57 | 487.73 | 1.281× |
| 1664×768×32768 | 7 | 3→0 | 1664×384×384 → 128×256×32768 | 486.60 | 313.88 | 1.550× |
| 1920×768×32768 | 7 | 3→0 | 1920×384×384 → 128×256×32768 | 649.09 | 500.30 | 1.297× |
| 2176×768×32768 | 5 | 3→0 | 2176×384×384 → 128×256×32768 | 819.18 | 496.76 | 1.649× |
| 2432×768×32768 | 6 | 3→0 | 2432×384×384 → 128×256×32768 | 983.67 | 540.00 | 1.822× |
| 2304×1024×32768 | 7 | 3→0 | 2304×384×384 → 128×256×32768 | 1091.52 | 710.55 | 1.536× |
| 2816×1024×49152 | 5 | 6→0 | 144×1024×512 → 128×256×49152 | 1607.60 | 1580.94 | 1.017× |

Modes 0, 2, 3, and 6 are BASE, ordinary single-core Split-K,
deterministic multi-core Split-K, and GM-to-L1 single-core Split-K.
All 17 online BASE cases emitted `baseM×baseN×baseK=128×256×64`,
`stepKa/stepKb=8/4`, and `singleCoreK=K`. Every Split-K case used a
`128×128×128` base tile. All 18 ran faster online in both orders, though
the 1.017× and 1.038× differences are small. The [portable records](records.json)
contain the full emitted tiling, including steps and block dimension,
plus each timing round.

Relative to the earlier corrected set of 10 family differences from
master, nine cases remain here and nine are new. The earlier
`256×512×16384` case now matches master's deterministic Split-K choice.
Comparing the two online selector versions directly, 16 family decisions
changed.

The [decision audit](decision_audit.csv) found **98 explicit online
decisions**. The online selection gate did not activate for `65536×448×1536`
and `8192×16384×4096`; both packages emitted the same BASE tile for
those two shapes. Among all **58** online BASE records, including those
two source bypasses, the model's source-L1 arithmetic exactly matches
the emitted `stepKa` and `stepKb`. The complete [portable records](records.json)
retain each audit, timing round, tile, device, and profiler path. Full
profiler traces remain on 910 at the raw path above.

The earlier [generic BF16 100-shape run](../bf16_100_vs_master_v1/README.md)
had a 0.9895× geometric mean, largely because its `16×16×65408` BASE
choice was about 20× slower than master's Split-K. The intervening
[narrow-B correction](../bf16_narrow_b_calibration_v1/README.md) reported
a 1.0193× **composite** using one newly measured shape and 99 historical
results. This experiment is a fresh 100-shape run of the source-tile
comparison. Relative to the earlier generic selector, 16 online family
choices changed. Cross-run speedup differences include measurement
variation and should not be interpreted as a paired timing experiment
between selector versions.

The largest raw same-tile gap was `512×128×73728`: the main run measured
125.34 µs master and 145.14 µs online, despite identical emitted tiling.
An independent [two-order repeat](repeat_011.json) on device 4 measured
master at 124.32/122.84 µs and online at 123.06/123.02 µs. It supports
timing variation as the cause of that single-round gap. The repeat is
reported separately and does not replace the main 100-shape measurement.

The selector's dtype and FP16 golden CTests passed after the code change.
BF16 remains opt-in: the BASE objective is still approximate (533.88 µs
versus 599.28 µs for the measured `16×16×65408` source BASE tile), and
other BF16 family rates remain provisional. The result supports the
source-tile decision on this boundary-focused dataset, not general
default enablement.
