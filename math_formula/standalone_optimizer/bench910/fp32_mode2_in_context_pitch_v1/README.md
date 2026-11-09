# FP32 mode-2 in-context A-pitch experiment on 910B3

## Question

The [copy-only holdout](../fp32_mode2_transfer_holdout_v1/README.md) and
[synchronization sweep](../fp32_mode2_sync_holdout_v1/README.md) overstate
the 192-KiB A-row-pitch penalty in several MatMulV3 mode-2 geometries.
Does the penalty remain when the **actual** MatMulV3 producer/consumer
pipeline runs the same logical work at all three pitches?

## Controlled change

The isolated [source patch](patch_source.py) makes `Ka` the physical A
row pitch while retaining `Kb=48384` as logical K for the mode-2 loop
count and B shape. The source kernel's `MatmulImpl::IterateAll` then
performs L1→L0 operand transfers, Cube work, Fixpipe output, and buffer
reuse in its normal schedule. The patch is restricted to FP32, mode 2,
`K=48384`, and A pitches 48896/49152/49408 elements (191/192/193 KiB).
It is an experiment only; no product tiling or selector code is changed.

The [benchmark](fp32_bench.cpp) allocates A with the selected physical
pitch, writes the same logical M×K values into each row, and retains a
logical M×K tensor descriptor. B is always K×N. It checks every C output
against an analytic FP32 reference. The [runner](run.py) requires the
same emitted tile at all three pitches, checks the research source in
the CANN tiling log and the `Ka/Kb` override, and uses `msprof` device
task time rather than the host event bracket.

The intended five held-out geometries are M×N = 2816×1024,
3072×1024, 2048×2048, 2816×1536, and 8192×4096. Four pass the
correctness gate. Each accepted pitch gets four rounds in alternating
order, with five warmups and five profiled launches per round.
The center excess is the 192-KiB median minus the mean of the 191- and
193-KiB medians. Compared with the earlier source-kernel holdout, this
test holds logical K and B footprint fixed while changing A pitch.

## Reproduction

On 910, copy the source tree into an isolated directory, apply
`patch_source.py` to that copy, and build it with CANN 9.0.0. The
[overlay script](make_overlay.sh) copies the prior research package to
a new isolated package and replaces only the patched host tiling
library and FP32 ND/ND device binary and metadata. This allows the
experiment to start once that variant compiles; the other variants
are not used. Compile `fp32_bench.cpp` against the same CANN. Run
`run.py --package PACKAGE --rounds 4` to audit and profile the four
accepted geometries. The large 8192×4096 case is retained as
`--geometry 4` for debugging but excluded by default. The
[summary](summary.json), [per-run records](results/),
[package fingerprints](package_fingerprint.json), and
[overlay hashes](overlay_hashes.txt) preserve the timing and build
evidence.

## Result

All 12 audits for the four accepted geometries passed with zero
reported FP32 output error. Each geometry emitted the same mode-2 tile
at all three pitches. The 48 accepted `msprof` runs contain five timed
MatMulV3 device tasks each. Center excess below is computed from the
median of four round medians at each pitch. The prior full-kernel
column uses its original K=48896/49152/49408 sweep, and the copy-only
column uses the packet-barrier probe from the synchronization sweep.
All values are µs.

| M×N | Core M×N×K | Fixed-work in-context excess | Prior full-kernel excess | Copy-only excess |
| --- | --- | ---: | ---: | ---: |
| 2816×1024 | 144×1024×256 | **417** | 439 | 987 |
| 3072×1024 | 160×1024×256 | **597** | 585 | 1159 |
| 2048×2048 | 208×1024×256 | **78** | 69 | 706 |
| 2816×1536 | 144×1536×256 | **415** | 442 | 967 |

The four-round fixed-work result is within 9–28 µs of the prior
full-kernel excess in each accepted geometry, while the serialized
copy probe exceeds it by 552–628 µs. This rules out the prior test's
changing logical K and B footprint as the main reason for the
copy-probe overestimate in these four geometries. It supports modeling
the actual producer/consumer schedule and overlap rather than assigning
the serialized copy time directly to the kernel critical path. It does
**not** identify a unique wait edge or a universal transfer rate.
The four individual center-excess rounds are 225–522, 436–675,
59–99, and 317–503 µs respectively, in table order. They are repeat
observations, not confidence intervals; the first two geometries show
material run-to-run variation.

## Wide-N validity limit

The 8192×4096×48384 audit faults on device with an out-of-range MTE
address when `Ka=48896` and `Kb=48384`; see the
[fault log](large_8192x4096_fault.log). Its emitted tile is
824×2048×192, `stepM=3`, 20 AICs. The same K and shape run correctly
with `Ka=Kb=48384` in both the baseline research package and the
patched binary. The patched path also runs correctly at
8192×1024 and 8192×2048 with padded A, while 8192×3072 faults. The
passing 8192×2048 tile has `singleCoreM=416`, `stepM=3`, hence two
inner-M panels; the failing 8192×3072 tile also has two panels. The
failure therefore involves more than panel count alone, and the exact
out-of-bounds operand has not been isolated. The three-panel
8192×4096 timing must not be compared with the old source or copy
probe until the research stride path is fixed and output-checked.

No online estimator, primitive parameter, or family selector was
changed by this experiment.
