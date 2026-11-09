# FP32 single-core Split-K mode 5 on 910B3

This study enables the source's $K=1536$, $M\ge49152$, $N=384$ NKM
route for same-type FP32 NN/ND/ND/ND on the experimental
`codex/matmul-v3-mip-selector` branch. Clean master retains its
FP16/BF16-only small-K gate and chooses BASE on these shapes. The
research package changes only that host gate and uses the existing
mode-5 kernel. The FP32 product online selector remains disabled.

## Device measurements

All runs used device 4, CANN 9.0.0, the exact FP32 input pattern, five
warmups and five measured kernels under `msprof`. The output
check reported zero error. The local [profile script](profile.py) and
individual [records](49152x384x1536/mode5/result.json) preserve the
emitted tilings and profiler CSV paths on 910.

| Shape | Arm | Emitted mode and tile | Kernel median, µs | Model score, µs |
| --- | --- | --- | ---: | ---: |
| 49152×384×1536 | clean master | BASE, 128×256×1536 | 823.344 | 820.094 |
| 49152×384×1536 | source mode 5 | 5, 2464×384×192 | 825.420 | 877.029 |
| 73728×384×1536 | clean master | BASE, 128×256×1536 | 1216.336 | 1228.802 |
| 73728×384×1536 | source mode 5 | 5, 3688×384×192 | 1241.696 | 1308.764 |
| 73728×384×1536 | N=128 probe | 5, 3688×128×192 | 1467.980 | 1308.764 |

The model ranks BASE ahead of the source mode-5 tile on both shapes,
matching the measurements. Its mode-5 source-tile estimates are 5–6%
high. The narrow-N tile is 18.2% slower than the source tile, although
the current linear byte/Cube objective scores them equally. The
source-compatible FP32 mode-5 domain therefore fixes
`singleCoreN=384` and searches M partitions and occupied AICs.
The narrow-N profiler shows much more MTE2 activity; a broader N
domain needs a per-panel A-transfer model.

## Provenance and scope

- Clean master package host-library SHA-256:
  `b5b886221f087406f4c2c7c39cb6bb27a8254ac3e55e571779b0a38b82a871ad`.
- FP32 mode-5 research package host-library SHA-256:
  `60ea1bbc2e82a763fcfa26e0e8c8a0d4e5f3ae183736aa1472c8ccaaa4bf7cf1`.
- The N=128 probe package adds a benchmark-only environment switch and
  has host-library SHA-256
  `64464e5f368084f6c484814bf118937ddc9dc273ddeddccf6ae967b2eb91ec2c`.
- Raw profiles and stdout remain on 910 under
  `/home/xianfei/solvebase_bench_20261025/cases/fp32_mode5_v1`.
  The research packages are isolated under the same benchmark root.
- The current selector preflight is NN/ND/ND/ND. The source's other
  FP32 mode-5 route requires transposed B and is not modeled here.

The [mode-5 formulation](../../SOLVE_SC_SPLITK_MODE5_FORMULATION.md)
gives the direct-output objective and the restricted search domain.
