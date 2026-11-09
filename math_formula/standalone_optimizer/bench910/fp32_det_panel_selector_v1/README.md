# FP32 panel timeline in the MatMulV3 host selector

This retired experiment compared a research host overlay with the existing
online selector at `cdeb680f` and clean master on the same 100 FP32 shapes.
The host overlay SHA-256 was
`4a771dd841a78b651f8a19cd95e06be93cd5b2d7015f77d201061d68087086d3`.

The overlay rescored the same deterministic candidate the existing solver
proposed and kept the existing family ranking and source tiling flow.
It passed the host's MK/NK order explicitly because the solver's
deterministic candidate does not carry the source stepM/stepN values.
The audit captured both deterministic scores and the emitted kernel tile.
The timing comparison measures device kernel execution for the chosen
family and tile.

The test audited the experimental package on each shape with
an output check. For a changed emitted tile or focused case, it profiled
each distinct audited tiling on the same device in forward and reverse
arm order, with five timed kernel tasks per round. Two arms with the
same audited tile used the same fresh measurement. For the other identical
tiles, the analysis reused the paired timings from the
[prior comparison](../fp32_source_mode2_preview_100_v1/README.md).

## Results on 910B3

All 100 output audits passed and logged both deterministic scores.
The new formula changed 15 families. The 15 changed cases and two
additional focused cases received 76 fresh profiles across devices 4–7.
The other 83 cases reused existing measurements after tile parity.

| Comparison | Cases | Geometric mean reference / new | New more than 5% faster | New more than 5% slower |
| --- | ---: | ---: | ---: | ---: |
| Existing online selector | 100 | 0.9816× | 1 | 6 |
| Existing online selector, family changed | 15 | 0.8836× | 1 | 6 |
| Master | 100 | 1.0108× | 20 | 9 |

The new selector's geometric mean kernel time was 1.87% longer than
the existing selector's on this dataset. Its geometric mean speedup
over master was 1.0108×, versus about 1.0297× for the existing selector.

| Shape | Existing → new family | Existing µs | New µs | Master µs |
| --- | --- | ---: | ---: | ---: |
| 8192×4096×32768 | Deterministic → mode 2 | 36444.40 | 33757.93 | 33757.93 |
| 2560×768×32768 | Deterministic → mode 2 | 2027.88 | 4655.47 | 2027.88 |
| 1664×768×32768 | Deterministic → mode 2 | 1337.07 | 2505.58 | 1337.07 |
| 2560×1024×49152 | Deterministic → mode 2 | 3919.27 | 5153.91 | 3919.27 |

For `2560×768×32768`, the deterministic score rises from 1915.17 to
2095.76 µs. This puts it above the existing source-mode-2 score of
2073.12 µs, even though that mode-2 tile measures 4655.47 µs. The
result exposes sensitivity to error in both family scores. The
experimental formula and its host overlay were retired.

[All cases](comparison.csv), [family changes](family_changes.csv),
[summary](summary.json), and [raw records](records.json) are available.
Run `python3 analyze.py records.json` from this directory to regenerate
the tables.
