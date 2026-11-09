# FP32 online mode-2 source-tile preview on 910B3

## Change under test

The online host now previews MatMulV3's source mode-2 tiler before scoring
FP32 implementation families. It restores the host tiling state after the
preview. The scorer replaces its hypothetical best mode-2 geometry with
the previewed tile; if the source path cannot emit mode 2, that family is
not compared. The score cache includes the preview tile fields. The model
preflight has a source-emitted FP32 mode-2 flag because the online host
deliberately bypasses its ordinary `singleCoreN >= 512` heuristic while
forcing that family. FP16/BF16 selection and the FP32 primitive profile
are unchanged.

On `2560×768×32768`, the old selector scored a hypothetical
`128×768×256` mode-2 tile at 1885.87 µs but MatMulV3 emitted
`128×256×256`. Scoring the previewed source tile gives 2073.12 µs,
above the deterministic split-K score of 1915.17 µs. A focused device-4
[audit and profile](focused_2560x768x32768.json) confirm that the changed online host emits deterministic
`2560×384×192` with zero output error. Its first kernel profile is
2027.74 µs versus 2020.38 µs for clean master; the emitted tiling
matches master.

The model and host compile on the 910 branch
`codex/matmul-v3-mip-selector`. `mmopt_dtype_tests` and
`mmopt_fp16_golden` pass. The fixed [100-shape dataset](../fp32_source_base_100_v1/dataset.json)
was rerun with [run.py](run.py), clean master, and the modified online
host on devices 4–7. Each case audits correctness and unforced branch
parity. Changed tilings receive two reversed-order rounds; unchanged
tilings receive one. Each arm has five warmups and five timed `msprof`
kernel tasks. The [analysis script](analyze.py) compares the new decision
with master and the [previous 100-case selector](../fp32_source_base_100_v1/README.md).

The temporary host package SHA-256 is
`2df4299101331a24044cb1c15a8f6487b22d0593f79aabe0af1a451272576b3a`.

## Mode-2 transfer model check

The current FP32 mode-2 timing formula remains unchanged in this host.
The [earlier exact-tile diagnosis](../fp32_base_formula_v2/MODE2_GAP_DIAGNOSIS.md)
shows that an isolated 128-KiB A-row-pitch MTE2 penalty explains about
70% of the 816-µs underestimation for the selector's `128×768×256`
tile. Its remaining error depends on geometry and overlap.

Two structural read-service changes were tested offline against the
[32 audited source-tile profiles](../fp32_mode2_sweep_v1/README.md):

| Diagnostic formula | Mean absolute relative error, 32 cases |
| --- | ---: |
| Current FP32 mode-2 pipeline | 3.22% |
| Count A and B once per source inner-M/K panel, with current read rate | 17.18% |
| Same source traffic, treating the 3.71-MB/µs rate as shared by 20 AICs | 6.08% |

The source traffic correction is necessary for a faithful formulation,
but the current aggregate L2-hot read rate and overlap expression have
compensating errors. Neither diagnostic change improves the earlier
profiles, so neither was deployed. A source-shaped, occupancy-aware
MTE2 service with buffer dependencies must be validated before
replacing the present online score.

## Full 100-case result

All 100 [raw records](records.json) passed the output check (maximum
relative error 0), produced five kernel tasks per profile, and matched
clean master on the unforced host tiling. The 100-case [analysis](analysis.json)
and [comparison table](comparison.csv) report paired master/online kernel
runtime. A ratio greater than 1 means online is faster.

| Cases | Count | Geometric mean master/online | Online >5% faster | Master >5% faster |
| --- | ---: | ---: | ---: | ---: |
| All | 100 | 1.0297 | 22 | 7 |
| Different family from master | 31 | 1.1007 | 20 | 6 |
| Same family as master | 69 | 0.9993 | 2 | 1 |

The shape-bootstrap 95% interval for the all-case geometric mean is
**1.0114–1.0513**. It describes variation across these 100 sampled shapes;
it does not include device or production-workload uncertainty. The earlier
online selector had a 1.0166 geometric mean on its separate 100-case run.
Only nine online family decisions changed between the two versions; their
[details are recorded](selection_changes.csv). Times below are from the
new paired master/online run, not a paired old/new run.

| M×N×K | Previous online → new online | Master µs | New online µs | Master/new |
| --- | --- | ---: | ---: | ---: |
| 4608×64×27520 | deterministic → mode 2 | 549.64 | 491.32 | 1.119 |
| 2048×1024×32512 | deterministic → mode 2 | 2110.26 | 2273.63 | 0.928 |
| 3072×1536×33024 | mode 2 → deterministic | 5266.19 | 4871.70 | 1.081 |
| 2560×768×32768 | mode 2 → deterministic | 2020.38 | 2027.74 | 0.996 |
| 2304×1024×36864 | deterministic → mode 2 | 2631.24 | 2910.69 | 0.904 |
| 3072×2048×32768 | mode 2 → deterministic | 7094.18 | 6732.75 | 1.054 |
| 8192×4096×32768 | mode 2 → deterministic | 33588.44 | 36406.66 | 0.923 |
| 4096×64×27392 | deterministic → mode 2 | 492.55 | 455.08 | 1.082 |
| 4096×64×27456 | deterministic → mode 2 | 485.20 | 482.47 | 1.006 |

The target `2560×768×32768` no longer chooses the hypothetical mode-2
tile. The new online and master both emit deterministic `2560×384×192`;
the 7.36-µs difference is run variation with identical kernel tiling.
The earlier online mode-2 run measured 4654.32 µs in a separate run.

The three newly worse decisions above reveal the remaining family-score
error. The [exact source-tile score check](score_source_tiles.cpp) uses the
same FP32 hardware profile as online; its [inputs](tile_input.txt) and
[outputs](tile_scores.txt) are retained. On `2048×1024×32512`, the
mode-2/deterministic scores are 2019.14/2040.78 µs, a 1.1% predicted
mode-2 advantage, but mode 2 is 7.7% slower in the paired measurement.
At `2304×1024×36864` the predicted mode-2 advantage is only 0.14%,
while it measures 10.6% slower. At `8192×4096×32768` deterministic
scores 32240.51 versus mode 2 at 32294.85 µs, a 0.17% predicted
advantage, while it measures 8.4% slower. A [separate AIC pipeline
profile](CASE61_DIAGNOSIS.md) shows that the deterministic path has much
more MTE2 and Fixpipe activity despite similar MAC activity. The source-tile preview fixes
the emittability mismatch; these narrow score margins expose the
remaining mode-2 input-service and overlap modeling error. A reliable
margin rule would require a separate uncertainty estimate; none is
introduced here.
