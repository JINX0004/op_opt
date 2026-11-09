# FP32 mode-2 A/B producer-consumer follow-up on 910B3

## Question

The [packet event graph](../fp32_mode2_timeline_v1/README.md) uses one
A+B copy-probe duration per K packet and distributes it across A and B
by bytes. Does a controlled GM→L1→L0 handoff reveal separate A and B
pitch behavior, and does it reduce the gap to the real MatMulV3 kernel?

## Controlled probe

The [AscendC kernel](probe_kernel.cpp) uses the four source-audited
fixed-work geometries from the [in-context experiment](../fp32_mode2_in_context_pitch_v1/README.md):
K=48384, physical A pitches 48896/49152/49408 FP32 elements,
20 AICs, and the same per-core tiles. Its historical A cache-hint
heuristic disables caching when `singleCoreN == N`, whereas the actual
device source leaves both A and B at normal caching for these four
tiles. A matched-hint copy-probe control is recorded in the
[cache-policy follow-up](../fp32_mode2_cache_policy_v1/README.md).
The A and B GM→L1 copies use the source
tile's ND→NZ copy parameters. A consuming mode waits for the copy,
loads a fixed 32-KiB NZ prefix from L1 to L0, then waits before
reusing that L1 destination. Modes are:

| Mode | Operation |
| ---: | --- |
| 0 | A+B copies with a barrier after each K packet; same pattern as the earlier packet-barrier probe |
| 1 | A copy, then L0A consumption and L1 reuse wait |
| 2 | B copies, then L0B consumption and L1 reuse wait |
| 3 | A+B copies, with L0A/L0B consumption and L1 reuse waits |

The probe has **no Cube or Fixpipe work**. Its 32-KiB L0 consumption is
smaller than the source's full operand sequence. The explicit waits
serialize each copy/consume pair, so this measures a controlled
producer/consumer route rather than `MatmulImpl`'s hidden double-buffer
schedule. The [C++ runner](probe_runner.cpp) checks an output token on
every AIC. [run.py](run.py) ran three pitch-order-balanced rounds on
devices 4–7, with three warmups and five timed launches per run. Its
[raw measurements](records.json), [three-round medians](summary.json),
and [binary SHA-256](binary_sha256.txt) are preserved.

The 2048×2048 tile's A-only mode 1 fails the token check on both
devices 4 and 6; see the [failure log](invalid_2048_a_only.log).
Its modes 0, 2, and 3 pass. No timing from the failed mode is used.
One 2816×1024 B-only round was roughly 670 µs faster than its two
repeats despite no A access; the three-round median prevents that
outlier from being misread as a pitch effect.

## Measured center-pitch excess

Center excess is the 192-KiB median minus the mean of the 191- and
193-KiB medians, in µs. The actual MatMulV3 column is the earlier
fixed-work, output-checked source measurement.

| M×N | A-only + consume | B-only + consume | A+B copy only | A+B + consume | Actual MatMulV3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2816×1024 | 162 | 11 | 1018 | 1018 | 417 |
| 3072×1024 | 142 | 5 | 1139 | 1082 | 597 |
| 2048×2048 | invalid | 0 | 677 | 670 | 77 |
| 2816×1536 | 162 | −3 | 968 | 961 | 415 |

The B-only code never reads A, so its pitch-independent timing is a
useful negative control. The combined A+B penalty is much larger than
the sum of the isolated A/B penalties for the three valid A-only
geometries. Thus byte-proportional allocation of a single combined
probe duration cannot represent the pitch interaction. Adding the
L1→L0 handoff changes the combined pitch excess by at most 58 µs;
it does **not** reproduce the real kernel's 77–597-µs response.

The [probe profiler](profile.py) confirms L0 consumption is present:
for the 2816×1024 tile at 191 KiB, AIC MTE1 busy time is approximately
0, 16, 234, and 249 µs in modes 0–3 respectively. Mode 3's MTE2 busy
time is about 2806 µs and mode 0's about 2749 µs. These are
overlapping busy counters, not additive elapsed stages. The profiler
records are in [profile_summary.json](profile_summary.json).

## In-context MatMulV3 profiler check

The [source profiler](profile_source.py) ran the already-audited
fixed-work research package at all three pitches, twice per pitch,
on the same four devices. Each profile uses three warmups and three
profiled tasks. It checks the FP32 output before accepting a run. The
host tiling library and benchmark SHA-256 values match the earlier
[package fingerprint](../fp32_mode2_in_context_pitch_v1/package_fingerprint.json).
The 24 checked [source records](source_profile_records.json) and
[stage-delta analysis](source_profile_analysis.json) are preserved;
[analyze_source.py](analyze_source.py) reproduces the table below.
Numbers are 192-KiB center excess in µs, based on the median of the
two profile rounds at each pitch.

| M×N | Full kernel elapsed | AIC total | MTE2 busy | MTE1 busy | MAC busy | Fixpipe busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2816×1024 | 417 | 385 | 281 | 8 | 0 | 41 |
| 3072×1024 | 597 | 542 | 462 | 6 | 0 | 73 |
| 2048×2048 | 77 | 48 | 53 | 0 | 0 | 5 |
| 2816×1536 | 415 | 411 | 265 | 9 | 0 | 47 |

The elapsed column is the earlier four-round fixed-work measurement;
the stage columns are this separate two-round profile and have their
own run-to-run variation. Stage counters overlap and cannot be summed
to reconstruct elapsed time. They show that the pitch anomaly remains
associated with MTE2 in the actual kernel, but its MTE2 busy excess
is only 53–462 µs versus 677–1139 µs for the independent A+B copy
probe. MAC work stays flat, as expected for fixed logical work.
Fixpipe busy also rises on the three larger-excess geometries despite
unchanged output bytes, consistent with an interaction or downstream
wait; the counters do not identify the exact wait edge.

## Model consequence

The copy-only and consuming probes both expose an A+B transfer
interaction at 192 KiB. The real kernel exposes a geometry-dependent
fraction while Cube, Fixpipe, and internal buffering proceed. A
separate A rate plus B rate, or a constant wait per packet, would
discard the observed interaction. The event evaluator and online
selector remain unchanged. The next model step requires actual
`MatmulImpl` issue/consume dependencies or an in-context primitive
that includes Cube and output; these stage counters and the controlled
probe do not identify them uniquely.

## Reproduce

On 910 with the matching CANN 9.0.0 installation, configure with
`-DCMAKE_BUILD_TYPE=Release` (the installed CANN merge step requires
it), `-DSOC_VERSION=Ascend910B3`, and
`-DASCEND_CANN_PACKAGE_PATH=/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0`.
Build [CMakeLists.txt](CMakeLists.txt), copy the audited fixed-work
`summary.json` to `fixed_work_summary.json` beside the scripts, and
run `run.py`, then `profile.py`. For the actual source check, reuse the
[prior research package](../fp32_mode2_in_context_pitch_v1/README.md)
and run `profile_source.py`, then `analyze_source.py` locally. The scripts
resume from checked result files and reject occupied devices.
