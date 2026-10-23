K is the **reduction axis** of the multiply — the dimension that is summed over, not a spatial tile of the output.

For `C = op(A) @ op(B)`:

| Tensor | Shape | Role |
|---|---|---|
| `op(A)` | **(M, K)** | left matrix |
| `op(B)` | **(K, N)** | right matrix |
| `C` | **(M, N)** | output |

Each output element is:

\[
C_{mn} = \sum_{k=0}^{K-1} A_{mk}\, B_{kn}
\]

So:

- **M** is rows of A / rows of C  
- **N** is cols of B / cols of C  
- **K** is the **shared inner length**. It does not appear in C’s shape; it only lengthens the sum.

Which physical dim of the *stored* input is K depends on transpose:

- `transA = false`: A is stored **(M, K)** → K is A’s last dim  
- `transA = true`: A is stored **(K, M)** → K is A’s first dim  
- `transB = false`: B is stored **(K, N)** → K is B’s first dim  
- `transB = true`: B is stored **(N, K)** → K is B’s last dim  

In `aclnnMm`, `self @ mat2` with no transpose is `self (M, K)` and `mat2 (K, N)`. Those two K sizes must match.

That is why “split-K” is different from splitting M or N: M/N split assigns **disjoint C tiles** to cubes; K-split assigns **partial sums of the same C tile**, which then have to be added.


Shape is the main input to that choice, but it is not a direct “this size → this kernel” switch, and **atomic** multi-core K-split is not chosen that way at all.

Auto-tiling tries templates **in order and keeps the first that fits** (`DoSelectTiling()`):

1. Full-load / L2 **BASE** (spatial M/N split; each cube still owns all of K)
2. **Single-core split-K** — `DoSingleCoreSplitKTiling()`
3. **Deterministic** multi-core split-K — `DoDeterministicMultiCoreSplitKTiling()`

So two large-K shapes can still miss both split-K paths if BASE already succeeded.

**Single-core split-K** is shape-driven, mainly:

- `K >= 27392`, or a few special MKN / TLB / mata cases
- then extra rejects: `singleCoreN < 512`, load-balance `avgRatio < 0.7`, etc.

**Deterministic multi-core K-split** is also shape-driven, mainly when K is large enough to fill cores **and** M/N cannot:

- `K >= aicNum * 384` and `mCnt * nCnt < aicNum / 2`
- or small M,N (`<= 64`) with large K (fp32)
- or `K >= 27392` after the earlier checks

**Atomic** `MULTI_CORE_SPLIT_K` is different: `DoSelectTiling()` never picks it. It only appears from the AOE/repo knowledge bank (`GetTilingFromRepo()`), and even then it must be fp32, `transA=false`, `transB=true`, and a **single round** (`mCnt * nCnt * kCnt <= aicNum`).

Other gates besides size: dtype, transpose, alignment, L2, chip (`supportL0c2out`), bias.

Practical picture: **size decides BASE vs single-core split-K vs deterministic K-split.** Atomic K-split is a repo special case, not the size-based sibling of single-core split-K.

### Default working behavior
A cube almost never loads a full `M×K` / `K×N`. It always loads **small blocks of both M/N and K**. Templates only change **who iterates which loop**.

What one cube actually holds in L0:

- **L0A:** `baseM × baseK`
- **L0B:** `baseK × baseN`
- **L0C:** `baseM × baseN` (partial K sum)

Typical sizes: `baseM = baseN = 128`, `baseK = 64` (fp32) or `128` (fp16). That is already M/N **and** K chunked.

Around that, there are more tiles:

| Level | What is chunked | Example |
|---|---|---|
| L0 (`baseM/N/K`) | M, N, **and K** on every path | 128×128×64 |
| L1 (`stepM`, `stepKa`, …) | several L0 blocks of **M and K** | 3×3 “MK33”: `stepM=3`, `stepKa=3` |
| One `Iterate` (`singleCoreM/N/K`) | the window the cube API covers | BASE: `singleCoreK = K`; split-K: `singleCoreK = stepKa×baseK` |
| Across cubes | **M/N** (BASE / single-core split-K) or **K** (multi-core split-K) | `usedCoreNum` |

Single-core split-K is the explicit “both” case at the **outer** loop: cores split M/N, and each cube also loops K chunks (`kCnt`, `innerLoopM` / `innerSingleCoreM`).

BASE still chunks K **inside** `Iterate()` as `baseK`; it just does not expose an outer K-split across GM/workspace.

So: **loading both K and M/N in chunks is the default cube behavior.** Split-K only means those K chunks are a **core-level / GM-level** loop, not only an L0 loop.

### BASE / single-core split-K
On the **BASE** path that is the loop nest:

1. **Outer:** split **C** along M/N (across cubes, then extra M/N tiles / L2 tiles on this cube).
2. **Inner:** on that cube, walk **K** in `baseK` (and L1 `stepKa`) until that C tile is done.

K stays on the **same core** so the partial sum never leaves **L0C**. A and B stream in along K; C is written to GM **once**. That avoids a cross-core add (atomic or vector reduce), which is the usual win — not extra parallelism on K.

`iterateOrder` only swaps whether M or N is the outer **spatial** loop. K remains the reduction inner loop.

Single-core split-K uses the same idea (M/N across cores, K on one cube), but the K loop is **outside** the cube API and partials can hit GM. BASE keeps that K loop **inside** `Iterate()`.

Yes. That is **multi-core split-K** (and Stream-K on arch35).


### multi-core K-split
On **multi-core K-split**, K is a **parallel axis across cubes**:

| Path | What each cube gets | How C is finished |
|---|---|---|
| `DETERMINISTIC_SPLIT_K` | a slice of K for the **same** C tile | AIV reduces workspace partials in order |
| `MULTI_CORE_SPLIT_K` (atomic) | `(m, n, k)` from `GetBlockIdx()` | `GetTensorC(..., atomic)` after a sync |
| Stream-K (arch35) | a K stream / split-K tile | mix reduce or on-the-fly combine |

Each cube still tiles **its** K slice as `baseK` in L0. The difference is that **different cubes’ K chunks belong to the same C elements**, so they must add. That is the case BASE avoids by keeping K on one core.

Auto-tiling picks this when **K is large enough to fill cores and M/N is not** (`DoDeterministicMultiCoreSplitKTiling`). Atomic is repo-only, not that size cascade.

Shape datasets for the two regimes are in [`experiments/`](../experiments/README.md): `same_core_k.csv` (BASE / single-core split-K) and `multi_core_k.csv` (deterministic multi-core K-split).