# Source-step MTE2 packet model for plain BASE

The 128×256 and 256×128 BASE tiles tie under the old byte-only objective,
yet the measured 2048×2048×4096 kernels take 139.792 and 161.178 µs.
Huawei's nominal L1→L0A/L0B rates alone favor the opposite orientation.
The host tiler also changes the **GM/L2→L1 K-step packet geometry**:

| baseM×baseN | `stepKa` / `stepKb` | A ND packet | B ND packet | A/B packets per output tile at K=4096 |
| --- | ---: | ---: | ---: | ---: |
| 128×256 | 8 / 4 | 128×512 | 256×256 | 8 / 16 |
| 256×128 | 4 / 8 | 256×256 | 512×128 | 16 / 8 |

These are source-emitted values from `CalL1Tiling()` and the checked
tiling records. A rows follow M, whereas B rows follow K. All four full
packets contain 128 KiB, but their ND burst length, pitch, and row count
differ. The earlier [`directional_read`](../directional_read/README.md)
probe copied only one `baseK=64` slice, so its 79.7 µs directional read
gap was not a sound estimate of this kernel's actual L1-step service.

`probe_kernel.cpp` performs repeated ND→NZ copies into A1 or B1 at 20 AICs.
`sweep.py` randomizes five copy counts over three rounds, records the
profiler's AIC task time and checks a result token. The inputs fit L2. The
source-step slope is fitted to the median of the three run medians at each
copy count; launch/setup cost is the regression intercept, not part of the
packet slope. [`l1_step_read_raw.jsonl`](l1_step_read_raw.jsonl) contains all
60 points. [`analyze.py`](analyze.py) regenerates
[`l1_step_read_analysis.json`](l1_step_read_analysis.json).

| packet | measured µs per 128-KiB packet |
| --- | ---: |
| A 128×512 | 0.8234 |
| B 256×256 | 0.7793 |
| A 256×256 | 0.7793 |
| B 512×128 | 1.1136 |

At 20 AICs, the mean serial A+B read service is 121.96 µs for 128×256
and 136.81 µs for 256×128. The busiest AIC owns seven of the 128 output
tiles, so its corresponding read service is approximately 133.39 versus
149.64 µs. These are **primitive service demands**, not kernel makespans;
the Matmul engine can overlap MTE2, MTE1, Cube, and output work.

## Optional packet objective

The standalone solver now accepts four *optional* `Hardware::primitives`:

| key | unit | meaning |
| --- | --- | --- |
| `base_mte2_a`, `base_mte2_b` | `bytes` | fixed packet time and per-AIC requested-byte rate |
| `base_mte2_a_rows`, `base_mte2_b_rows` | `rows` | per-AIC ND row rate; fixed time must be zero |

All four must be supplied together, and `MipTileDomain::base_core_equals_base`
must be true. This path is only for the checked plain BASE source branch.
The default profile remains the previous generic objective. Split-K and
conversion still use `gm_read`; supplying this profile does not alter those
branches. `SourceBaseL1Steps()` follows the host source's L1-depth and
alignment calculation, and the solver evaluates each legal baseK because
K-step packet costs are not monotone in baseK. For AIC $q$, let $S_q$ be its
spatial base-packet count; $U^A_q$ the sum of actual M extents over those
packets; $U^B_q$ the sum of actual N extents. Let $h_A,h_B$ be the derived
source `stepKa`, `stepKb`, and $P_A=\lceil K/(h_A\,baseK)\rceil$,
$P_B=\lceil K/(h_B\,baseK)\rceil$. With measured packet intercepts
$\alpha_A,\alpha_B$, byte rates $R_A,R_B$, and ND row rates $Q_A,Q_B$:

$$
R_q=S_q(P_A\alpha_A+P_B\alpha_B)
 +\frac{2K U^A_q}{R_A}+\frac{2K U^B_q}{R_B}
 +\frac{P_A U^A_q}{Q_A}+\frac{K S_q}{Q_B}.
$$

The optional reduced objective is

$$
T=L+\max_q\max\left\{\frac{2K E_q}{R_C},
 R_q+\frac{2MN}{R_W}+O_q\lambda_W\right\}.
$$

The row and byte terms use actual spatial tails; the packet intercept
counts full and final K-step packets equally. Cube/MTE2 overlap is a `max`
approximation. MTE1 is still absent: the Huawei 256/128 B/cycle paths are
nominal service limits and do not specify the internal issue timeline.

`packet_model.py` fits the three per-packet coefficients
$\alpha+\beta\cdot\mathrm{KiB}+\gamma\cdot\mathrm{ND\ rows}$ from this sweep
and the prior baseK-slice sweep, with **no MatMul kernel-time fit**.
`sweep.py --heldout --rounds 3` measures four additional 64-KiB shapes.
The held-out relative slope errors are +18.3%, +6.9%, +4.3%, and −3.0%; the
first is too large to treat the compact primitive law as universally
calibrated. The coefficients and all errors are in
[`packet_model_analysis.json`](packet_model_analysis.json).

[`model_check.cpp`](model_check.cpp) injects those primitive coefficients
into `ReducedObjectiveUs` for the seven previously measured source-compatible
whole-kernel shapes; [`model_check.csv`](model_check.csv) has every prediction
and measurement. The profile ranks 128×256 ahead of 256×128 in **all seven**
cases, without fitting any whole-kernel time. Across the 14 individual
time estimates, median absolute relative error is 2.01% and maximum is
4.49%. For 2048²×4096 it yields 135.521 versus 164.175 µs, compared with
measured 139.792 versus 161.178 µs. The estimated gap is 28.65 µs versus
the observed 21.39 µs. These seven cases test one tile pair and one
hardware/software installation; they do not establish selection accuracy
over other tile sizes, implementations, cache regimes, or core counts.
Broader source-compatible device comparisons are needed before adopting
the packet profile as the default selector.

A later [12-shape source-backed selection experiment](../PACKET_BASE_OPTIMIZER_VS_DEFAULT_910B3.md)
found 11 exact default matches and one different plan. On the different
`(6144,384,8192)` shape, the model predicted 256×128 would be 2.4% faster,
but six paired device rounds measured it **15.94% slower** (paired 95%
interval 15.24–16.65% slower). This is a clear failure on a tail-heavy
shape; the current optional profile should not be used for general BASE
selection until the source block/L2 scheduling is modeled.

Build the optional diagnostic with the standalone CMake target and run
`mmopt_l1_step_read_check`. To repeat the primitive sweep, build this
directory's `CMakeLists.txt` against CANN and run `sweep.py` from this
directory. The raw profiler task times are preserved in the JSONL files;
profiler output folders are intentionally not committed.
