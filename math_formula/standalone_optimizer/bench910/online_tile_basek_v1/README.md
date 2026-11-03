# Online BASE tile injection, 16×16×102400 FP16

Device 4 of the Ascend 910B3, CANN 9.0.0. The experimental branch
`codex/matmul-v3-mip-selector` carries `MMV3_ONLINE_TILE=1` as an opt-in
BASE K-tile test. The normal online selector is enabled with
`MMV3_ONLINE_SELECTOR=1`. An isolated copy of the host package was used for
the tile arm, while the earlier family-only package remained unchanged.

The benchmark audited correctness, checked device 4 for other running
processes before and after profiling, and used two paired rounds with eight
kernel samples per arm per round. The table shows the median of the round
medians. The unmodified master and family-only online arms provide controls.

| Arm | Mode | baseM×baseN×baseK | stepKa/Kb | Kernel µs | Max relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Master | 3, deterministic Split-K | 128×128×128 | 3/3 | 42.00 | 0 |
| Online family only | 0, BASE | 16×16×64 | 64/64 | 936.87 | 0 |
| Online with model BASE K tile | 0, BASE | 16×16×1024 | 4/4 | 858.11 | 0 |

The model-proposed `baseK=1024` passes source tiling and correctness checks.
Changing to it lowers BASE time by 8.4%, but BASE remains 20.4 times slower
than deterministic Split-K. The reduced model predicts 30.05 µs for this
BASE tile, 28.6 times below its measured time. Both BASE tilings have the
same L1 K span (`baseK×stepKa=4096`), so changing the Cube K tile alone
does not solve the missing read cost.

## Why the BASE estimate is too low

The reduced online objective uses the generic `gm_read` profile. At the
selected `baseK=1024`, its 30.05 µs estimate decomposes into 6.20 µs
launch/head-A conversion and a 23.85 µs BASE stage. That stage is the
maximum of 3.56 µs Cube service and 23.85 µs read/output service. The
read/output number uses 100 `baseK` packets, 0.10962 µs fixed latency
per operand and packet, plus 1.77 µs for operand/output bytes. The read
rate was measured across 20 AICs with larger packets; it does not include
the cost of reading B's narrow ND rows on one useful output core.

The emitted source has `stepKa=stepKb=4`, so it moves both operands in
25 L1 K packets of 4096 elements. B is a 102400×16 ND matrix: each packet
reads 4096 rows of 32 bytes. An isolated one-core `DataCopy` ND→NZ probe
using that exact packet geometry gives:

| Operand | 1 packet | 13 packets | 25 packets | Slope per packet |
| --- | ---: | ---: | ---: | ---: |
| A | 4.90 µs | 13.48 µs | 21.52 µs | 0.69 µs |
| B | 37.97 µs | 441.54 µs | 844.43 µs | 33.60 µs |

An independent repeat of B's 25-packet point gave 843.05 µs. B's
source-shaped read time is therefore close to the 858.11 µs BASE kernel
time. The probe uses serialized packets and checks a completion token;
it is evidence for the bottleneck, not an exact decomposition of the
MatMulV3 kernel. The kernel profiler independently reports MTE2 as 74.6%
of the tile arm's AIC activity versus 0.9% MAC activity (medians across
eight warm tasks). These are AIC activity ratios, not percentages of the
whole kernel's elapsed time.

The existing optional source-step packet model includes ND row counts,
but its large-tile, 20-AIC coefficients predict approximately 165 µs for
this tile. That still misses the observed one-core narrow-row service.
The next model change should use a geometry- and occupancy-aware B read
service, calibrated with one-core 16-wide packet measurements, and score
the tile that MatMulV3 will actually emit before selecting the family.

Raw audit, tiling, and timing results are in [summary.json](summary.json).
The [probe source and measurements](mte2_probe/) and
[kernel counters](op_metrics.json) preserve this diagnosis.
The device-side run directory is
`/home/xianfei/solvebase_bench_20261025/cases/online_tile_basek_v1`.
