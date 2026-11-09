#!/usr/bin/env python3
"""Repeat the 30-shape master/online comparison with FP32 source BASE scoring."""

import argparse
import importlib.util
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
BASELINE = ROOT / 'fp32_online_30_new_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_online_30_runner', BASELINE)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)

PACKAGE = ROOT / 'ops_nn_fp32_source_base_fix_package'
HOST_SHA = 'b49ce309d8760e2cab3ddaf0eb286684ead170ca775fb8811b4ae1db61925c75'
experiment.runner.OUT = ROOT / 'cases/fp32_source_base_fix_v1'
experiment.runner.PACKAGES.update(research_default=PACKAGE, online=PACKAGE)
experiment.runner.HASHES.update(research_default=HOST_SHA, online=HOST_SHA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, choices=(4, 5, 6, 7))
    parser.add_argument('--max-cases', type=int)
    args = parser.parse_args()
    experiment.check_package()
    experiment.worker(args.device, args.max_cases)
