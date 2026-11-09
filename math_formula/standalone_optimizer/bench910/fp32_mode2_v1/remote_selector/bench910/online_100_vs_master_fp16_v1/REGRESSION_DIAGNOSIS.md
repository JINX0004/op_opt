# Investigation of two apparent large-M BASE regressions

The 100-shape comparison initially measured online slower than clean
master for 65536×256×1536 and 47104×384×1536. Two reversed-order rounds
still showed 4.76% and 4.05% slowdowns, respectively. Both arms used
mode 0, key 65536, the same visible Cube tiling, 20 cores, and the same
workspace size. This follow-up isolates the selector path, package, and
measurement method.

## Controlled comparisons

Each row below is a four-round paired experiment. Each timing arm uses
ten warmups and eight profiled MatMulV3 kernel calls per round. The
runner alternates arm order between rounds and checks device idleness.
All output audits passed. Ratios are the median of each arm's four
round times, first arm divided by second arm; values above 1 mean the
second arm was faster. Raw per-round records are linked in each row.

| Shape | Comparison | First arm µs | Second arm µs | Ratio |
| --- | --- | ---: | ---: | ---: |
| 65536×256×1536 | [Online package: selector off / on](selector_path_device6.json) | 245.394 | 246.125 | 0.997× |
| 47104×384×1536 | [Online package: selector off / on](selector_path_device7.json) | 248.034 | 249.304 | 0.995× |
| 65536×256×1536 | [Master / same master package](master_self_device4.json) | 244.635 | 239.785 | 1.020× |
| 47104×384×1536 | [Master / same master package](master_self_device5.json) | 234.645 | 234.780 | 0.999× |
| 65536×256×1536 | [Master / online package, selector off](master_vs_online_disabled_device6.json) | 248.909 | 247.260 | 1.007× |
| 47104×384×1536 | [Master / online package, selector off](master_vs_online_disabled_device7.json) | 240.361 | 235.430 | 1.021× |

The selector-on/off pairs use the **same online binary**; the first arm
omits `MMV3_ONLINE_SELECTOR`, and the second enables it. The master
self-controls use the **same master binary** and the same environment
for both arms. The selector-disabled comparisons use the two distinct
packages but source selection in both. Each pair runs on one device;
absolute times across separate rows or devices should not be compared.

## What the source and profiles show

- The two packages have identical device kernel files. `diff -qr` found
  differences only in the two host tiling-library copies. Both arms
  launched `MatMulV3_ND_ND_ND_ND_FP16_FP16_FP16_FP16_high_performance_65536`
  with 20 cores. The tiler logs recorded the same L2 cache flag (`2`).
- The online branch's `DoL2CacheTiling` has the source `ALL` path's L2
  block-grouping fallback for online mode 0. No `CalcTile` update was
  logged for either shape. The visible tiling and first 136 bytes of
  the logged device tiling payload matched. The profiler log prints
  only 136 of the 280 payload bytes, so this does not prove every
  hidden field is identical.
- The benchmark's `run_case.sh` starts a fresh `base_aclnn_bench`
  process for each timed arm. Its eight steady-state kernel calls
  within a process are tight, but whole-process medians vary. For
  65536×256×1536, four selector-off runs of the **same online package**
  ranged from 225.394 to 255.254 µs; selector-on runs ranged from
  229.816 to 258.606 µs. The master self-control also showed a 2.0%
  difference between identical binaries in four-round medians.
- In the earlier L2-fallback experiment, the patched online package
  measured **2.1% faster** than master for 65536×256×1536 over four
  rounds. The later selector-disabled comparison also reversed the
  slowdown: online was 0.7% and 2.1% faster for the two shapes.

## Conclusion

These measurements do **not** support a reproducible slowdown caused
by the online selector for either shape. The selector-on/off tests are
within 0.5%, while identical-package and distinct-process controls
show differences of the same size as the original 4–5% gap. The
immediate cause of the reported gap is sensitivity of the current
fresh-process benchmark to run-to-run state. The records do not
distinguish device clock state from allocation or memory-placement
effects, so the physical source of that variation remains unresolved.

The diagnostic controls above did not replace the original
[100-case aggregate](analysis_before_rerun.json). Subsequently, the
user requested a fresh [eight-round rerun](README.md) of both shapes
against clean master on their original devices. Those results favor
online by 2.38% and 1.02% and replace only the two affected rows in the
[updated aggregate](analysis.json). A future comparison of small kernel differences should include a
same-package null control and record device clocks and buffer addresses
alongside repeated process-level timings.
