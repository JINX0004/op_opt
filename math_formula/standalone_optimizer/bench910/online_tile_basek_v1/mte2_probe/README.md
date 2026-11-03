# One-core BASE read probe

This is an isolated copy of the earlier `l1_step_read` probe configured for
device 4, a 16-wide operand tile, and a 4096-element L1 K step. The
runner checks a completion token for each launch. `run_probe.py` profiles
one, 13, and 25 serialized A and B packets with eight task samples each.

On 910B3 with CANN 9.0.0, configure with `-DCMAKE_BUILD_TYPE=Release`,
`-DSOC_VERSION=Ascend910B3`, and `-DASCEND_CANN_PACKAGE_PATH=<CANN root>`.
The Release build type is required by this CANN version's AscendC object
merge script. The remote run used a short build path, `/tmp/mmv3mte2`.

Measurements are in [results.json](results.json), with one independent
B/25 repeat in [repeat.json](repeat.json). The complete remote profiler
output is under
`/home/xianfei/solvebase_bench_20261025/cases/online_tile_basek_v1/mte2_measure`.
