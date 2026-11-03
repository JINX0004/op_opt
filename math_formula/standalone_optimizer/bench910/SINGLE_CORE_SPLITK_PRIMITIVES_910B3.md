# FP16 single-core Split-K primitive probes on 910B3

These are **exploratory path coefficients**, measured on device 0 with
CANN 9.0.0 and the source-matched `mat_mul_v3` checkout at
`b2cdb471de09d93eb79b34812c1ebe4e938ae10c`. The isolated remote
build is `/home/xianfei/singlecore_calib_20261030`. They are available
as optional inputs to `SolveReducedMip`; they are not installed as
defaults because an independent end-to-end validation is still needed.

## Method

- Each point uses 10 warmup launches and the median of eight `msprof`
  `task_time(us)` values. Conditions are randomized within rounds and
  run sequentially on the same device.
- [The AIC probe](primitives/probe_kernel.cpp) holds one FP16 128³ Cube
  result in L0C and compares FP32 Fixpipe plain store (mode 5) with
  `SetAtomicAdd<float>()` (mode 6) at equal packet count and size. Twenty
  AICs write separate output regions. The sweep checked the first FP32
  element was at least one 128-wide Cube result. The runner now checks
  the exact accumulated value; a post-sweep atomic smoke passed.
- [The AIV probe](splitk_primitives/probe_kernel.cpp.in) loads FP32 into
  UB once, repeats `Cast(..., CAST_RINT)` 1–64 times, then writes FP16.
  Forty AIVs run, and the first/last output elements are checked.
- [The mode-6 copy probe](singlecore_copy/probe_kernel.cpp) issues the
  source's ND→NZ A `[128,k_packet]` or B `[k_packet,128]` copy with
  `k_packet=384/512` and source row pitch. Twenty AICs serialize 1–256
  packets. The token checks completion; the copied L1 payload is not
  numerically checked. The L2-resident source has K=6144 and occupies
  12 MiB per operand. A K=49152 source occupies exactly 192 MiB per
  operand, the device's reported L2 size; its 64-copy runs access less
  than that. The K=73728 source occupies 288 MiB per operand, with
  192/256-copy cases that touch beyond L2.

All raw task times, including every repetition, are in
[`singlecore_calibration_raw.jsonl`](singlecore_calibration_raw.jsonl),
[`singlecore_copy/copy_raw.jsonl`](singlecore_copy/copy_raw.jsonl), and
[`singlecore_copy/copy_above_l2_288_raw.jsonl`](singlecore_copy/copy_above_l2_288_raw.jsonl).
The analyzers are [`analyze_singlecore_calibration.py`](analyze_singlecore_calibration.py)
and [`singlecore_copy/analyze.py`](singlecore_copy/analyze.py).

## Measured services

The FP32 **plain-store** mode fits
`task_us = intercept + packets * lambda_store + bytes * beta_store`.
Across 36 conditions, `lambda_store = 0.2432 µs/packet` and
`beta_store = 2.925e-5 µs/byte`, an effective requested-byte rate of
34,190 bytes/µs per AIC in the 20-AIC probe. The median absolute
residual is 0.300 µs. The by-round byte slopes are
2.879–2.956e-5 µs/byte. The 3.332 µs intercept includes launch and
probe setup and is **not** charged per packet. Optional `sc_fp32_store`
replaces the generic FP16-output `gm_write` primitive for the FP32
partial-store term of modes 2/5 and the first mode-6 K packet. The
[device-4 family check](MODE6_OUTPUT_COST_CHANGE_910B3.md) showed this
serialized, distinct-address slope cannot be used for every repeated
mode-6 workspace update. A separate source-in-context
`sc_fp32_store_steady` **total atomic-output** demand was subsequently
measured in the [device-4 output study](MODE6_OUTPUT_MEASUREMENT_910B3.md).

The atomic comparison fits

`T_atomic - T_store = intercept + packets * lambda_atomic + bytes * beta_atomic`.

The pooled 36 paired conditions give `lambda_atomic = 0.0541 µs/packet`
and `beta_atomic = 9.379e-6 µs/byte`, equivalent to an **incremental**
requested-byte rate of 106,618 bytes/µs. The median absolute residual
is 0.292 µs. By-round byte slopes are 8.934–9.999e-6 µs/byte; the
packet intercept varies from 0.0163 to 0.0788 µs, so the byte term is
better identified than the fixed term. The global fit also has a
0.156 µs intercept. The optional solver field `sc_atomic_extra` uses
only the packet and byte terms, and applies them after K slice 0.

For the resident cast, the difference between 64 and 16 repeats is
divided by 48 at each vector length. The nine derived conditions fit
`T_cast = 0.0156 µs + elements / 116,718 elements/µs` per AIV cast.
The per-round element rates span 70,764–294,992 elements/µs. Small
repeat counts are noisy and this probe does not isolate source flag
handoff or workspace GM reads/writes. Treat `sc_cast` as a sensitivity
parameter until a source-shaped AIC→AIV panel experiment validates it.

The L2-resident copy table reports the median of per-round incremental
packet slopes from 16 to 64 serial copies. Each 384 packet contains
98,304 bytes and each 512 packet 131,072 bytes. Linear
`lambda + bytes/rate` fits through those two packet sizes; the rates are
*effective requested-byte* rates for a 20-AIC run, not physical HBM rates.

| Operand | 384 packet (µs) | 512 packet (µs) | fitted λ (µs) | fitted rate (bytes/µs) |
| --- | ---: | ---: | ---: | ---: |
| A | 0.675 | 0.796 | 0.312 | 270,810 |
| B | 0.874 | 1.158 | 0.023 | 115,414 |

The 192 MiB allocation is a **near-L2 diagnostic**, not an above-L2
calibration: its 64-copy runs access less than 192 MiB across the unique
source slices. Their slopes are in the raw data. Several third-round
64-packet L2 points are outliers; the table uses medians.

With the 288 MiB allocation, the 192-copy cases visit the full source
K range and the 256-copy cases revisit its beginning after that sweep.
The 20-AIC task medians are:

| Operand | K packet | 192 copies (µs, two rounds) | 256 copies (µs, two rounds) |
| --- | ---: | ---: | ---: |
| A | 384 | 140.4, 133.6 | 179.3, 180.4 |
| A | 512 | 163.6, 159.7 | 213.4, 213.2 |
| B | 384 | 393.8, 394.2 | 507.5, 445.7 |
| B | 512 | 500.4, 440.1 | 648.3, 685.7 |

The B path rises sharply and varies enough that a single above-L2
`lambda + bytes/rate` fit is not defensible from these two rounds.
The probe tests A and B separately, so it also does not identify their
concurrent contention or source prefetch overlap.

## Using the optional solver fields

For an **L2-resident FP16 sensitivity run**, add these primitives to
the caller's `Hardware::primitives` map (units match `PrimitiveTiming`):

```cpp
h.primitives["sc_fp32_store"] = {"bytes", 0.2432, 34190.0};
h.primitives["sc_atomic_extra"] = {"bytes", 0.0541, 106618.0};
h.primitives["sc_cast"] = {"elements", 0.0156, 116718.0};
h.primitives["sc_gm_to_l1_a"] = {"bytes", 0.3124, 270810.0};
h.primitives["sc_gm_to_l1_b"] = {"bytes", 0.0227, 115414.0};
```

For the measured source-in-context FP16 mode-6 atomic output demand on
device 4, also supply:

```cpp
h.primitives["sc_fp32_store_steady"] = {"bytes", 0.0, 91000.0};
```

`sc_fp32_store` replaces the generic FP32 partial-store estimate for
modes 2/5 and mode 6's first K packet. Later mode-6 packets use
`sc_fp32_store_steady` if supplied, otherwise generic `gm_write` plus
`sc_atomic_extra` as an exploratory proxy. The measured steady value
already includes atomic output cost and must not be added to
`sc_atomic_extra`. Without a steady profile, `sc_atomic_extra` adds an
increment over generic `gm_write` for later K slices. `sc_cast` is added
to the AIV panel service; the A/B fields replace
the generic read term for mode 6 only. Omit both A/B fields together
to retain the generic read model. This primitive profile
is **not** a calibrated whole-kernel model: existing BASE read/write
profiles are still exploratory, the cast coefficient varies between
rounds, and the reduced recurrence does not model shared GM contention
or the exact mode-6 copy/MAC dependency graph. Validate selected tilings
against source-executed kernels on held-out FP16 shapes before claiming
runtime accuracy or a speedup.

To repeat the sweeps after building the three probe directories with the
installed CANN package and sourcing its `set_env.sh`:

```bash
python3 singlecore_calibration.py --rounds 3
python3 singlecore_copy/sweep.py --rounds 3 --regime l2
python3 singlecore_copy/sweep.py --rounds 2 --regime above_l2_288
python3 analyze_singlecore_calibration.py
python3 singlecore_copy/analyze.py
```
