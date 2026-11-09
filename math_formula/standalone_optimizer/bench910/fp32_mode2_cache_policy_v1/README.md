# FP32 mode-2 A/B cache-hint factorial experiment on 910B3

## Question

The earlier serialized copy probes overpredict the 192-KiB A-row-pitch
penalty of the actual MatMulV3 mode-2 kernel. Some probes also used an
A-cache-disable heuristic that differs from the device source. This
experiment checks the hint mismatch and measures the four A/B hint
combinations **inside the actual kernel** at otherwise identical work.

## Fixed workload and intervention

The four source-audited shapes in
[fixed_work_summary.json](fixed_work_summary.json) keep logical
`K=48384`, B's logical pitch, the emitted tile, and 20 AICs fixed.
Only A's physical row pitch varies among 48896, 49152, and 49408
FP32 elements (191, 192, and 193 KiB). The source-built host tiler
is the same one used by the
[fixed-work baseline](../fp32_mode2_in_context_pitch_v1/README.md).

For these tiles, the actual device source leaves **both A and B at
normal L2 caching**. Its A-disable condition requires the NKM path and
`stepN * baseN == N`; its B-disable condition requires the MKN path and
`stepM * baseM == M`. Neither applies here. The isolated
[device-header patch](patch_kernel.py) adds a research-only policy
switch, read from padding after row 0 of A. The padding is outside the
logical matrix. Policy 0 leaves both normal; 1 disables A; 2 disables
B; 3 disables both. All policies use the **same device binary**, host
tiler, shape, tile, and data apart from the marker. The
[benchmark](fp32_cache_bench.cpp) checks output correctness and reports
the selected policy. The original fixed-work package is interleaved as
a control.

A second isolated [source variant](patch_no_normal.py), reproduced by
[build_no_normal.sh](build_no_normal.sh), omits explicit
`CACHE_MODE_NORMAL` calls for policy 0. It compiled to a byte-identical
FP32 kernel binary: both variants have SHA-256
`4b129c876a3f385d5477c2bb0a7775c806a4decdd3d660d41108dbc476a2adea`
([variant hashes](no_normal_package_hashes.txt)). Thus those calls do
not explain any policy-0 versus original-package timing difference.
The marker read and diagnostic code still distinguish the patched
binary from the original kernel, whose SHA-256 is
`873dd8452eaf248ee81153ed10f64db03393985f97cabf70a1ecb1ad03d8d4d6`.

The [build script](build.sh) compiles the isolated kernel against the
machine's CANN 9.0.0 installation. To shorten the diagnostic build,
[slim_tiling_key.py](slim_tiling_key.py) restricts a separate generated
source copy to tiling key 65568, which all four audited shapes emit.
[make_overlay.sh](make_overlay.sh) replaces only its FP32 kernel binary
and metadata in a copy of the baseline package. [run.py](run.py)
audits output, tile, host-tiler
origin, and pitch override for each of the 48 shape × pitch × policy
combinations before timed profiling. Four balanced rounds run on
devices 4–7. `msprof` records task elapsed time and AIC pipe counters.
The original package runs once per shape, pitch, and round as an
interleaved control. [analyze.py](analyze.py) computes center-pitch
excess and 2×2 factorial effects. These are diagnostic experiments;
the online selector is unchanged.

For each shape and policy `ab`, where `a` disables A and `b` disables
B, define the center-pitch excess in µs as

`E_ab = T_ab(49152) − [T_ab(48896) + T_ab(49408)] / 2`.

The within-binary A effect is `E_10 − E_00`, the B effect is
`E_01 − E_00`, and their interaction is
`E_11 − E_10 − E_01 + E_00`. These are calculated both from
per-pitch medians and separately for each paired round. Pipe counters
represent overlapping busy times, so they are analyzed separately
from elapsed kernel time.

## Matched-hint copy-probe control

Before the device-kernel intervention, [probe_hint_control.py](probe_hint_control.py)
reran the existing packet-barrier A+B copy probe with normal A and B
cache hints and with A disabled. It used three balanced rounds on the
same devices, shapes, pitches, and tiles. The
[raw records](probe_hint_records.json) and
[medians](probe_hint_summary.json) show:

| M×N | Normal A/B center excess, µs | A disabled center excess, µs |
| --- | ---: | ---: |
| 2816×1024 | 1025.6 | 1016.4 |
| 3072×1024 | 1125.5 | 1118.9 |
| 2048×2048 | 655.4 | 671.9 |
| 2816×1536 | 962.2 | 968.8 |

Thus the previous A-hint mismatch changes the probe's center excess
by at most 17 µs, far less than the several-hundred-microsecond gap
between the probe and real kernel. The hint mismatch should still be
documented, but it does not explain that gap.

## Actual-kernel results

The [48 audits](audit_records.json) all emitted the same mode-2 tile as
their fixed-work source baseline, including key 65568, and had zero
reported relative output error. The [192 patched-kernel profiles](records.json)
and [48 interleaved original-package profiles](control_records.json)
span four balanced rounds. [Analysis](analysis.json) contains every
per-pitch median, round-level center excess, AIC counter, and factorial
effect. [Package hashes](package_hashes.txt), [provenance](provenance.json),
and the [run log](run.log) record the binaries and invocation order.

The table gives center-pitch excess in µs, computed from the four-round
median at each pitch. `Original` is the concurrently measured source
package; the four numbered columns use the patched binary.

| M×N | Original | 0: normal A/B | 1: A disabled | 2: B disabled | 3: both disabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2816×1024 | 497 | 484 | 498 | 460 | 427 |
| 3072×1024 | 568 | 529 | 569 | 578 | 672 |
| 2048×2048 | 83 | 77 | 73 | 77 | 84 |
| 2816×1536 | 421 | 450 | 422 | 429 | 475 |

The resulting within-binary factorial effects are:

| M×N | A disable | B disable | A×B interaction |
| --- | ---: | ---: | ---: |
| 2816×1024 | +13 | −24 | −47 |
| 3072×1024 | +39 | +48 | +54 |
| 2048×2048 | −4 | 0 | +11 |
| 2816×1536 | −28 | −21 | +73 |

No hint choice consistently removes the 192-KiB penalty. On the
2048×2048 tile, all four policies have 73–84 µs excess. On
3072×1024, disabling both hints increases the four-round-median
excess from 529 to 672 µs; on 2816×1024 it decreases it from 484 to
427 µs. Even the largest median difference, 143 µs, is much smaller
than the roughly 529-µs gap between the matched-hint copy probe's
1126-µs excess and the original kernel's earlier 597-µs excess for
3072×1024. Cache-hint selection alone does not explain that gap.

The four **paired-round** A effects range from −107 to +108 µs for
2816×1024 and from −170 to +347 µs for 3072×1024. Their B and
interaction ranges also cross zero for every shape (see
[analysis.json](analysis.json)). The original-package center excess
itself ranges from 474 to 689 µs across rounds for 3072×1024. Hence
the table is descriptive, not a significance claim for small A/B
differences. The patched policy-0 binary also has a different
round-level center excess from the original package, ranging from
−326 to +242 µs on that shape. The byte-identical no-normal variant
rules out the explicit normal calls as the reason, but the extra
padding-marker read and changed device code remain a confound for
direct patched-versus-original comparisons. The within-patched-binary
factorial is the relevant hint test.

The AIC `MTE2` busy-time excess changes with the policies, but not in
a common direction: 2816×1024 goes from 447 µs at policy 0 to 344
µs at policy 3, while 3072×1024 goes from 402 to 643 µs. These
overlapping busy counters cannot be summed into kernel elapsed time.
The remaining gap calls for a model of the source's packet issue,
buffer waits, and overlap; a cache-hint-only rate correction is not
supported by these measurements.
