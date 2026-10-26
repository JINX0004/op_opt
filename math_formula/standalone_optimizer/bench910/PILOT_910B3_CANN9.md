# First 910B3 / CANN 9.0.0 SolveBase pilot

The [portable case records](pilot_results.json) come from device 0 of an
Ascend910B3 host on 2026-10-25. The driver reported version `25.3.rc1`;
`compiler/version.info` reported CANN `9.0.0` (package timestamp
`20260428_134817545`). The installed platform API returned the
[resource record](hardware_910b3.json): 20 AICs, 40 AIVs, 524,032-byte L1,
65,536-byte L0A/L0B, 131,072-byte L0C, 196,352-byte UB, and
201,326,592-byte L2. Device 0 had no running process at the initial check.

The runner called installed `MatMulV3` directly. Each case used contiguous
FP16 ND inputs, no transpose or bias, preallocated buffers, 10 profiler
warmups, and 8 measured kernel tasks. A periodic nonuniform input pattern
was checked across **every** output element; all recorded maximum relative
errors were zero. CANN logs confirmed tiling key `65536`, BASE, no A/B
ND→NZ conversion, no full-load or Fixpipe optimization, and a 20-block
launch. The profiler's per-kernel task time is the measurement here.

| Shape M×N×K | singleCore M×N | base M×N×K | Useful output tiles | Median kernel time | Measured range |
| --- | --- | --- | ---: | ---: | ---: |
| 256×256×1024 | 32×128 | 32×128×128 | 16 | 8.292 µs | 8.140–8.456 µs |
| 512×512×1024 | 112×128 | 112×128×128 | 20 | 10.830 µs | 10.540–19.940 µs |
| 1024×1024×4096 | 128×256 | 128×256×64 | 32 | 48.072 µs | 45.700–49.400 µs |

The 512 case has a 19.94 µs outlier; retain the raw durations and repeat it
before estimating uncertainty. A repeated older 256 case had a median of
8.64 µs with uniform inputs, and another patterned 256 case had 8.20 µs.
This spread is larger than a single-case profiler range, so these numbers
are a functional pilot, not a stable performance profile.
The complete profiler exports and CANN logs remain on the host under
`/home/xianfei/solvebase_bench_20261025/cases/`; the checked-in JSON keeps
each selected kernel duration and realized tiling without large raw logs.

The installed CANN source is **not identical** to this checkout's
`mat_mul_v3` source. SHA-256 for the installed
`matmul_v3_base_tiling.cpp` and `mat_mul_v3.cpp` is, respectively,
`f5fe01f30fffca5cc96e9a528f110d71d466be86355e2c27b2d3dc63bd39250b`
and `4b3635d18bcc075de8bc393a5d8e9954eeba947f832d2f2272b1117768930593`.
The corresponding checkout hashes at commit
`79b011574b9cf59d714a34811327ded1d2fbc90d` were
`1bf60f1aba7d8110c9bff2197d4660f78631ab643fa818f20c9762349a130593`
and `3abcdab5f60c3dd214a1b14a06b41ec47ef70a1cd22a3db95c753fd1c4b5a3fe`.
Thus these timings validate the *installed CANN implementation path* and
its emitted geometry. They do not establish the performance of this
checkout's source or of alternate tiles proposed by the optimizer.
The separate [source-matched follow-up](SOURCE_MATCHED_910B3_CANN9.md)
uses the remote `ops-nn-sribd` operator supplied by the user.

An intentionally excluded 750×650×2048 test selected `nd2nzB=1`, key 0,
and a `MIX_AIC` kernel. `run_case.sh` now stops after parsing its tiling,
before profiling, for such a case. The INFO-log event bracket was about
500–600 µs even for kernels below 50 µs; it includes host submission and
logging gaps, so it is retained only as a diagnostic.

The pilot does not identify the primitive rates or latencies in the
`SolveBase` objective. A measured profile requires isolated Cube,
GM→L1 read, and matching output/Fixpipe probes, then held-out tile-ranking
checks. Source-controlled spatial tiling is also required to compare
alternate plans at a fixed shape. Keep the existing illustrative profile until
those measurements are available.
