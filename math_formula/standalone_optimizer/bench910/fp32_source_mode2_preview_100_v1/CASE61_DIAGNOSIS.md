# FP32 `8192×4096×32768`: deterministic Split-K ranking error

The paired [100-case record](records.json) has case 61 on device 5.
Both arms passed the output check, used 20 AICs, and have `nd2nzA=B=0`.
The source emits:

| Arm | Family | Core M×N×K | Base M×N×K | Workspace | Model µs | Paired kernel µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Master | Single-core split-K mode 2 | 824×2048×192 | 128×128×64 | 148 MiB | 32294.85 | 33588.44 |
| Online | Deterministic split-K | 2816×384×192 | 128×128×64 | 185 MiB | 32240.51 | 36406.66 |

The model gives deterministic a **54.34-µs** advantage (0.17%). It
actually takes **2818.22 µs longer** (8.4%). Mode-2 error is 1293.59 µs;
deterministic error is 4166.15 µs. This is a ranking error caused mainly
by underestimating the deterministic path. The solver's deterministic
candidate is the same `2816×384×192` tile that the source emits, so
this case has no deterministic tiling-search mismatch. The same source
mode-2 tile measured 33.6–33.9 ms
in earlier device runs.

## Why the reduced objective is optimistic here

The deterministic source uses NK order: `singleCoreN=384` makes 11
outer N panels (ten 384-column panels plus one 256-column tail).
Each panel divides 171 K slices of 192 elements across 20 AICs, so the
busiest AIC receives nine slices, or 1728 K elements. The source loops
over these panels, writes FP32 workspace partials, signals the vector
reduction, and reuses one of two workspace buffers. Starting with the
third panel it executes `WaitEvent`, `SyncAll`, and a full pipe barrier
before reusing a buffer; each panel also ends with a full pipe barrier.
See `MatMulMultiCoreSplitKDivide` in
`mat_mul_v3/op_kernel/mat_mul_deterministic_splitk_kernel.h`.

The current `SplitStageUs` scores each full 384-column panel at
**3019.90 µs Cube** versus **2872.95 µs GM-read/startup**; the modeled
vector stage is only **26.08 µs**. Thus ten full panels are classified
as Cube-bound, with just 4.9% between the modeled Cube and input
branches. The reduced objective overlaps operand reads with Cube and
the previous panel's reduction with the next panel. It does not
explicitly represent source buffer-reuse waits, workspace read/write
service, cache policy, or the per-panel MTE2/Fixpipe dependency chain.
Its resulting 32240.51-µs score is essentially the sum of Cube
critical-path terms plus a small final reduction.

Two reversed-order [PipeUtilization profiles](case61_pipe_summary.json)
were collected from the same master and online packages with five
kernel tasks per arm and round. The table shows medians across rounds;
stage busy times overlap and cannot be added to estimate runtime.

| Reported AIC counter | Mode 2 µs | Deterministic µs |
| --- | ---: | ---: |
| Total | 33755 | 36376 |
| MAC | 30239 | 29919 |
| MTE1 | 13663 | 12759 |
| MTE2 | **9808** | **20599** |
| Fixpipe | **11804** | **19375** |

The deterministic path has similar reported MAC activity but much
more MTE2 and Fixpipe activity. Its total AIC time exceeds reported
MAC time by about 6.46 ms, versus 3.52 ms for mode 2. These counters
support an exposed transfer/output/synchronization gap in the reduced
timeline. They do not isolate which dependency accounts for each
microsecond. The 128-KiB FP32 A row pitch at `K=32768` may amplify
transfer service, but this deterministic tile has not yet been tested
at fixed work with neighboring physical pitches.

The next model experiment should time the same deterministic source
tile at neighboring A row pitches and measure its workspace reduction
and buffer-wait path per outer panel. The score should then use those
services in an explicit max-plus panel recurrence rather than assuming
the current 26-µs reduction is fully hidden.
