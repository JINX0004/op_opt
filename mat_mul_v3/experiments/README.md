# Split-K path experiments

Two shape datasets, chosen so auto-tiling on Atlas A2/A3 (24 AIC) should pick:

| Dataset | File | Target path | Shape idea |
|---|---|---|---|
| A `same_core` | `datasets/same_core_k.csv` | **BASE** and **single-core split-K** | M/N large enough to fill cubes; K is inner on **the same cube** |
| B `multi_core` | `datasets/multi_core_k.csv` | **Deterministic multi-core split-K** | M/N too small to fill 24 cores; K is split **across cubes** |

Atomic `MULTI_CORE_SPLIT_K` is not in dataset B: auto-tiling never selects it (AOE repo only).

## Shape gates used

Dataset A **BASE** rows: `K < 27392` and `ceil(M/128)*ceil(N/128)` large, so neither split-K template wins.

Dataset A **single-core split-K** rows: `K >= 27392` and `N >= 3072` so M/N still fill cores and the `singleCoreN < 512` / `N in [896, 2048]` rejects do not fire. Single-core split-K is tried **before** deterministic K-split.

Dataset B: `K >= 24*384` (9216) and `ceil(M/128)*ceil(N/128) < 12`, or fp32 `M,N <= 64` with `K >= 6144`, or `transA` with mid-K. N is kept small so single-core split-K fails first.

Layouts: `trans_a=0, trans_b=0` means A is `(M,K)`, B is `(K,N)`.

## Run

Dry-run (print shapes and the approximate classifier):

```bash
python3 mat_mul_v3/experiments/run_mm_bench.py --dataset both --backend dryrun
```

On NPU with torch_npu:

```bash
python3 mat_mul_v3/experiments/run_mm_bench.py --dataset same_core --backend torch --device npu:0 --out /tmp/same_core.csv
python3 mat_mul_v3/experiments/run_mm_bench.py --dataset multi_core --backend torch --device npu:0 --out /tmp/multi_core.csv
```

Confirm the real template with host logs (`tiling enable state is SingleCoreSplitK` / `DeterministicMultiCoreSplitK`). The Python classifier is an approximation of those gates, not the tiling key.
