# Device-4 FP32 output probe

This probe measures an **output-only** 20-AIC Fixpipe stream at source
mode-6 spatial tile dimensions. It checks an FP32 atomic sum and a
completion token on every run. The source kernel's in-context Fixpipe
counter is extracted separately by `extract_source_pipeline.py`; see the
[measurement report](../MODE6_OUTPUT_MEASUREMENT_910B3.md) before using
either rate in the solver.

Build against the same CANN package as the source MatMulV3 benchmark:

```bash
CANN_ROOT=/path/to/cann-9.0.0
source "$CANN_ROOT/set_env.sh"
cmake -S . -B build -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
  -DSOC_VERSION=Ascend910B3 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python3 sweep.py --runner build/mode6_output_probe \
  --output output_sweep_device4.jsonl --device 4 --rounds 2
```

To extract the source counters from a completed family-selection run:

```bash
python3 extract_source_pipeline.py /path/to/splitk_family_selection_device4_v1 \
  source_pipeline_device4.jsonl
python3 analyze.py > analysis_device4.json
```

The saved `holdout_2048x4096x32768_device4.json` summarizes two separate
source-kernel runs with their emitted tiling, output checks, task times, and
pipeline counters. Its full remote `msprof` directory is under
`/home/xianfei/mode6_output_probe_20261030/source_holdout` on 910.
