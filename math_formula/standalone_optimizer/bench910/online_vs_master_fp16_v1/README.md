# Online selector versus clean `master`: FP16 on 910B3

This experiment compares the online direct-search family/path selector at
`3270828b` with clean `master` at `b2cdb471`. It measures the **executed
MatMulV3 device kernel**, not the reduced model objective or AscendCL event
bracket. The selector chooses the implementation family/path; the source host
tiler still chooses all block and buffer tiles.

## Setup

- Ascend 910B3, device 4; installed CANN 9.0.0; FP16 ND/ND/ND, NN, no bias.
- The clean-master worktree was
  `/home/xianfei/dev/ops-nn-sribd-master-bf16` (the name predates the decision
  to use FP16). Its only source modification bypasses the missing optional
  legacy cache hook for the isolated `aclop` benchmark. Selection logic is
  unchanged.
- Both custom package trees contain the same files except the host tiling
  library and its `liboptiling.so` alias. The online arm sets
  `MMV3_ONLINE_SELECTOR=1`; the master arm leaves it unset.
- Each case had a correctness/tiling audit followed by `msprof` task-time
  measurement: 10 warmups and 12 measured kernel calls per round. The five
  shapes where the selected path changed had three paired rounds, alternating
  arm order; the eight equal-path controls had one round. All 26 audits
  succeeded; maximum relative output error was 0.000445. Device-4 idle checks
  found no interference flags.

The full [dataset](dataset.json), [per-case summary](summary.json),
[CSV](summary.csv), [round-level kernel durations and tilings](measurements.json),
[provenance](provenance.json), and [runner](online_vs_master_fp16.py) are saved
here. The original `msprof` profiles and CANN tiling logs remain under
`/home/xianfei/solvebase_bench_20261025/cases/online_vs_master_fp16_v1` on 910.

## Results

Times are microseconds. Each time is the median of the available round
medians. Speedup is master time divided by online time; values above 1 favor
the online selector. Modes: 0 = BASE, 3 = deterministic Split-K, 5 = single
core NK33 Split-K, 6 = single core GM-to-L1 Split-K.

| M×N×K | Master → online mode | Master µs | Online µs | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 16×16×102400 | 3 → 0 | 42.01 | 937.28 | 0.045× |
| 64×64×65536 | 3 → 3 | 45.45 | 42.96 | 1.058× |
| 128×128×32768 | 3 → 3 | 32.76 | 30.64 | 1.069× |
| 1792×768×27392 | 3 → 3 | 384.39 | 387.95 | 0.991× |
| 2048×768×32768 | 3 → 0 | 744.18 | 547.08 | 1.360× |
| 2560×1024×36864 | 3 → 6 | 1021.17 | 838.94 | 1.217× |
| 1024×2560×36864 | 6 → 6 | 814.64 | 837.55 | 0.973× |
| 2048×2048×4096 | 0 → 0 | 139.60 | 140.56 | 0.993× |
| 4096×4096×8192 | 0 → 0 | 1002.26 | 997.37 | 1.005× |
| 4096×4096×32768 | 6 → 6 | 5638.65 | 5614.18 | 1.004× |
| 49152×384×1536 | 5 → 0 | 247.10 | 269.71 | 0.916× |
| 2560×1024×73728 | 3 → 0 | 2661.19 | 3264.46 | 0.815× |
| 8192×4096×16384 | 0 → 0 | 4166.19 | 4136.54 | 1.007× |

Five paths changed: two wins and three losses. Their paired-round speedup
ranges were 0.044–0.047×, 1.337–1.449×, 1.204–1.289×, 0.908–0.986×, and
0.812–0.823× in table order. Across all 13 shapes the geometric-mean speedup
is **0.806×**; without the severe 16×16 case it is 1.026×. This dataset is a
designed coverage set, not a production workload distribution, so neither
aggregate should be interpreted as expected deployment speedup.

## What the failures show

The 16×16×102400 model predicts BASE at 30.05 µs versus deterministic
Split-K at 37.00 µs. The source BASE path performs the full K chain on one
useful spatial Cube core (16×16 output); deterministic Split-K distributes K
work. Measured time is 937.28 versus 42.01 µs. The reduced BASE service rate
is far too optimistic at this occupancy and tile size.

The model compares its **optimized tile within each family**, while the host
executes a different source tile after receiving only the family/path choice.
For 49152×384×1536, the model's mode-5 geometry is 128×128, but the source
mode-5 path uses a 2464×384 single-core output tile. The model estimates
BASE 416.10 µs versus mode 5 at 785.42 µs, yet the measured source paths are
269.71 µs for BASE and 247.10 µs for mode 5. This is a path-cost and tile
feasibility mismatch, not a solver issue.

For 2560×1024×73728, the model gives BASE 3006.17 µs and deterministic
Split-K 3035.62 µs, a narrow modeled margin. The measured source paths are
3264.46 and 2661.19 µs respectively. The K>65535 conversion and source
geometry need to be scored using the actual emitted path. By contrast,
2048×768×32768 and 2560×1024×36864 benefit when the source default uses
only a few spatial tiles and the selected BASE or mode-6 path exposes more
spatial parallelism.

**Conclusion:** this online selector is useful as an experiment but is not a
safe general replacement for the master selection yet. The next model change
should score each family's *source-emitted* tiling rather than an independent
optimized geometry, and represent effective per-core throughput for small
M/N and long K. Re-run this dataset, especially the five changed-path cases,
after those constraints are added.
