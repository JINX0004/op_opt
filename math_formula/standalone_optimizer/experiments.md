### Step 1. Decision on choosing the right implementation

There is hard coded rule in mat_mul_v3 to decide which implementation (base or deterministic_split_k) is chosen:

ceil(M/128) * ceil(N/128) < 10, K >= 7680, and the layout is not “A not transposed, B transposed”. then select deterministic_split_k.

We have an analytical method to estimate the best run time for a given shape with the best tiling decision for the two implementations. 


The purpose is to prove our method is better than the default way of implementation selection in mat_mul_v3.

### Step 2. Part 1: Cases where both ways of decision making aligns

use our methods to estimate the run time of BASE, DETERMINISTIC_SPLIT_K seperatly to check these shapes ceil(M/128) * ceil(N/128) < 10, K < 7680

our estimation is the same as the mat_mul_v3 that BASE is better than DETERMINISTIC_SPLIT_K.

### Step 3. Part 2: Contradictions between two decision methods

use our methods to find the shapes subject to ceil(M/128) * ceil(N/128) < 10, K >= 7680, and BASE is still better than DETERMINISTIC_SPLIT_K, we found following shapes with estimations,

In a 648-shape scan of that box, BASE is ahead on 502 of them. At K = 7680 and K = 8192 it is ahead on all 108 shapes.

Shape BASE (µs)	Split-K (µs)
128×128×7680 10.377 11.137
16×1152×7680 15.388 31.675
96×256×12288 19.311 19.336
16×1152×16384 29.599 49.383
64×512×32768 49.005 49.423
Split-K first passes BASE at K = 12288, on 128×128 and 128×256. By K = 65536 it leads 74 of the 108 shapes, and BASE still leads the other 34, mostly the thin ones. The objective ignores transpose, so NN, TN, and TT share these scores.

4. we selected the two shapes where the difference is significant. The two shapes are 16×1152×7680 and 16×1152×16384

then we evaluate these shapes by controlling the selection of implementation strategy in mat_mul_v3, and profiling the run time on 910b3,

All three cases finished for FP16 16×1152×7680 on device 1. On this shape the default cascade selects deterministic split-K, so that run uses the same kernel as the forced one. Forcing the base path is faster.


| | Default cascade | Forced base | Forced deterministic split-K |
|---|---|---|---|
| Split-core mode | 3 | 0 | 3 |
| Tiling key | `0x10030` | `0x10000` | `0x10030` |
| Kernel median | **38.64 µs** (38.54–38.84) | **21.46 µs** (21.26–21.46) | **36.82 µs** (36.26–38.60) |
| Max abs error | 0 | 0 | 0 |


The cascade takes deterministic split-K because M and N fill only 9 of 20 cores and K is exactly 7680, which is the K threshold in SupportMultiSplitK(). The base kernel still finishes in about 21.5 µs.

All three cases finished for FP16 16×1152×16384 on device 1. The default cascade again selects deterministic split-K. Forcing the base path is still faster.

|  | Default cascade | Forced base | Forced deterministic split-K |
| --- | --- | --- | --- |
| Split-core mode | 3 | 0 | 3 |
| Tiling key | `0x10030` | `0x10000` | `0x10030` |
| Kernel median | **51.18 µs** (50.78–52.02) | **39.22 µs** (38.80–39.42) | **51.66 µs** (51.46–52.02) |
| Max abs error | 0 | 0 | 0 |

K is just over twice the previous 7680 case. The base kernel rose from 21.46 µs to 39.22 µs, and deterministic split-K rose from about 37 µs to about 51 µs.

### Step 4. More cases show much better performance by using BASE instead of Deterministic_split_k in our method

At K = 7680, Split-K is at least 40% slower than BASE on 34 shapes in that box, counting M×N and N×M as one shape. 16×1152×7680 itself is 105.8% slower (15.388 µs versus 31.675 µs).

The same gap shows up whenever one side stays small. With a side of 16 or 32, Split-K is about twice BASE:

| Shape | BASE (µs) | Split-K (µs) | Split-K slower |
|---|---:|---:|---:|
| 16×16 | 4.729 | 10.020 | 112% |
| 16×32 | 4.862 | 10.086 | 108% |
| 16×64 | 5.127 | 10.219 | 99% |
| 16×128 | 5.657 | 10.484 | 85% |
| 16×256 | 6.718 | 14.518 | 116% |
| 16×384 | 8.518 | 17.711 | 108% |
| 16×512 | 9.314 | 20.500 | 120% |
| 16×768 | 13.399 | 25.938 | 94% |
| 16×1024 | 14.725 | 29.763 | 102% |
| 16×1152 | 15.388 | 31.675 | 106% |
| 32×32 | 5.127 | 10.153 | 98% |
| 32×64 | 5.657 | 10.286 | 82% |
| 32×128 | 6.718 | 10.551 | 57% |
| 32×256 | 8.254 | 14.652 | 78% |
| 32×384 | 9.315 | 17.912 | 92% |
| 32×512 | 10.377 | 20.769 | 100% |
| 32×768 | 14.198 | 26.341 | 86% |
| 32×1024 | 15.791 | 30.300 | 92% |
| 32×1152 | 16.588 | 32.280 | 95% |

### Step 5. Onboard testings with these shapes

All 17 shapes finished for FP16 with K fixed at 7680. On every shape the default cascade selects deterministic split-K, and the forced base path is faster. Every run passed the correctness check.

Kernel median time in microseconds, five timed launches after 20 warmup launches, device 1:

| Shape | Default | Forced base | Forced deterministic split-K |
|---|---:|---:|---:|
| 16×32 | 24.04 | **13.10** | 22.38 |
| 16×64 | 23.52 | **16.08** | 25.82 |
| 16×128 | 24.30 | **17.62** | 23.26 |
| 16×256 | 32.52 | **19.00** | 25.36 |
| 16×384 | 28.48 | **19.24** | 27.88 |
| 16×512 | 28.62 | **20.74** | 28.08 |
| 16×768 | 33.58 | **20.86** | 32.92 |
| 16×1024 | 36.04 | **22.60** | 36.56 |
| 16×1152 | 38.38 | **21.38** | 40.26 |
| 32×64 | 23.36 | **15.92** | 23.44 |
| 32×128 | 23.64 | **17.58** | 23.18 |
| 32×256 | 26.92 | **19.02** | 26.14 |
| 32×384 | 26.06 | **20.58** | 27.80 |
| 32×512 | 29.68 | **22.12** | 30.94 |
| 32×768 | 35.94 | **22.62** | 34.18 |
| 32×1024 | 36.48 | **23.60** | 40.26 |
| 32×1152 | 38.88 | **21.98** | 40.58 |

