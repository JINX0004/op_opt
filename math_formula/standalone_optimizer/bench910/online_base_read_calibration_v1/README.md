# Online BASE B-read calibration on 910B3

This experiment investigates the online selector's BASE estimate for FP16
`M=N=16, K=102400` and tests a measured correction on device 4. The generic
selector estimated about 30.05 µs for BASE and selected mode 0, whereas the
source-backed BASE kernel took about 937 µs. The source's B ND→NZ copy is
the dominant cost for this narrow-N case. Mode 3 is deterministic multi-core
Split-K.

For performance relative to **default MatMulV3 selection**, see the separate
[20-shape direct comparison](LATEST_VS_DEFAULT.md). The generic-versus-narrow
figures below compare two online profiles, not online against default.

## Primitive measurement and fit

`probe_kernel.cpp` executes source-shaped A or B `DataCopy` packets into L1;
`probe_runner.cpp`, `sweep.py`, and `task_sweep.py` collect event and task
sweeps. One 16-wide B packet spans 4096 K rows (128 KiB of FP16 input). A
single AIC takes about 844 µs for 25 B packets; 25 A packets take about
21.5 µs. The source-backed BASE kernel with the injected `baseK=1024` tile
took 858.11 µs. The AIC activity sample attributed 74.6% to MTE2 and 0.9%
to MAC. These measurements identify B read service as the main missing
term for this case.

`fit.py` fits a source-step packet law from packet-count slopes, rather than
from a single full-kernel time:

\[
t_A(F,c)=\alpha_A(c)+F\beta_A(c),\qquad
t_B(F,R,c)=\alpha_B(c)+F\beta_B(c)
 +\mathbf1_{\mathrm{rowBytes}<64}R\rho_B(c).
\]

Here `F` is source bytes per packet, `R` is the number of ND source rows,
and `c` is the occupied AIC count. Each fixed and byte coefficient
interpolates linearly between measured one-AIC and 20-AIC endpoints by
`u=(c-1)/19`; the narrow-row coefficient uses
`rho_B(c)=rho_B(1)+u²[rho_B(20)-rho_B(1)]`. The quadratic interpolation is
an empirical description of the measured intermediate-occupancy points,
not a hardware law. The fitted coefficients are in [fit.json](fit.json),
with raw data in [task_sweep.jsonl](task_sweep.jsonl) and
[event_sweep.jsonl](event_sweep.jsonl). On 18 held-out packet cases, the
largest absolute relative packet-slope error was 7.64%; the 16-wide B
packets at 4 and 10 AICs were within 0.6%. The packet fit predicts
864.24 µs for the 16×16×102400 injected BASE tile.

The online `NarrowB910B3()` profile uses only the measured B narrow-row
coefficient and keeps the existing generic BASE read cost otherwise. The
`RowAware910B3()` profile additionally uses the fitted A/B packet fixed
and byte coefficients. Set `MMV3_ONLINE_ROW_READ_PROFILE=1` for the narrow
profile, or `=full` for the broader diagnostic profile. Both require
`MMV3_ONLINE_SELECTOR=1`. BF16 stays on the generic profile. Set these
before starting the process because decisions are cached by shape.

## Source-backed family-selection checks

For each arm and shape, the benchmark audited emitted source tiling and
correctness, checked that device 4 was idle, then alternated arm order in
two rounds with eight kernel samples per arm per round. The values below
are medians of the two round medians, in µs. `MMV3_ONLINE_TILE` was unset,
so the test changed family/path selection only; the source still emitted
the actual tile. All correctness audits passed.

The first six shapes compare generic and the **full packet** profile. The
full-profile package was built before `=1` was narrowed, so its `=1`
means the current `=full` profile. Raw tiling, round timings, and audit
details are in [online_row_read_fp16_summary.json](online_row_read_fp16_summary.json).

| M×N×K | Generic mode / µs | Full mode / µs | Generic/full |
| --- | ---: | ---: | ---: |
| 16×16×102400 | 0 / 934.08 | 3 / 42.53 | 21.96× |
| 1792×768×27392 | 3 / 387.23 | 0 / 390.01 | 0.993× |
| 2560×1024×36864 | 6 / 829.75 | 0 / 1180.79 | 0.703× |
| 1024×2560×36864 | 6 / 842.29 | 0 / 1218.32 | 0.691× |
| 4096×4096×32768 | 6 / 5620.14 | 0 / 4494.47 | 1.250× |
| 2048×2048×4096 | 0 / 140.03 | 0 / 139.78 | 1.002× |

The two large regressions rule out making the full packet fit the general
online default. A possible cause is that the isolated probes do not model
the cache and source assignment behavior of those large kernels; this
remains a hypothesis.

The next seven held-out shapes compare generic with the **narrow B-row**
profile from the isolated narrow package. Raw results are in
[online_narrow_b_fp16_summary.json](online_narrow_b_fp16_summary.json).

| M×N×K | Generic mode / µs | Narrow mode / µs | Generic/narrow |
| --- | ---: | ---: | ---: |
| 32×16×102400 | 0 / 945.40 | 3 / 50.12 | 18.86× |
| 64×16×102400 | 0 / 971.65 | 3 / 62.31 | 15.59× |
| 128×16×65536 | 0 / 650.01 | 3 / 50.71 | 12.82× |
| 256×16×32768 | 0 / 405.42 | 3 / 32.76 | 12.38× |
| 16×32×102400 | 0 / 129.19 | 3 / 40.72 | 3.17× |
| 32×32×102400 | 0 / 136.36 | 3 / 46.18 | 2.95× |
| 64×32×65536 | 0 / 97.03 | 3 / 45.90 | 2.11× |

For N=16, the measured source BASE tile uses `baseN=16`, matching the
narrow-row condition. For N=32, the model's winning hypothetical BASE
candidate uses `baseN=16`, but forced source BASE emits `baseN=32`.
Those N=32 tests demonstrate an improved family choice empirically; they
do not validate the narrow-row term as the causal cost of the emitted BASE
tile. The remaining structural issue is to score the tile that the source
will emit for each family, or inject and validate the optimized tile.

## Reproduction and artifacts

The experiment uses the remote branch `codex/matmul-v3-mip-selector` in
`/home/xianfei/dev/ops-nn-sribd-mip-selector` and its FP16 `mat_mul_v3`
source, built against the installed CANN 9.0.0. The device-4 packages are
`/home/xianfei/solvebase_bench_20261025/ops_nn_mip_row_package` (full
packet comparison, host library SHA-256
`a4bb560bd56f4512b92ac10d9db0f9b6c6cba51cee236dc54c372972a96341f8`)
and `/home/xianfei/solvebase_bench_20261025/ops_nn_mip_narrow_package`
(narrow comparison, host library SHA-256
`092693da5d9818375738c688e167e87b51f0f08cf84e4a9ad2233ecc89495cd5`).
The paired scripts [run_row_compare.py](run_row_compare.py) and
[run_narrow_compare.py](run_narrow_compare.py) use the benchmark harness
under `/home/xianfei/solvebase_bench_20261025`; their remote raw case
directories are `cases/online_row_read_fp16_v1` and
`cases/online_narrow_b_fp16_v1`.
