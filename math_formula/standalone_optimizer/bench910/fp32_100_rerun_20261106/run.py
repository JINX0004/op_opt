#!/usr/bin/env python3
"""Fresh 100-case FP32 comparison using the 2026-11-06 selector binary."""

import importlib.util
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_100_source_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

runner.HERE = Path(__file__).resolve().parent
runner.OUT = ROOT / 'cases/fp32_100_rerun_20261106'
runner.SELECTOR_HASH = 'da3be68fbd02f6cec69fb1ff89ab12c9b9523ec59209e9624c7b984894954778'

original_environment = runner.environment


def environment(arm, model_mode, log_level):
    if arm == 'proposed' and model_mode == 2:
        env = original_environment('research_default', model_mode, log_level)
        env['MMV3_RESEARCH_SPLITK_FAMILY'] = 'single'
        return env
    if model_mode not in (0, 2, 3):
        raise RuntimeError(f'no source-backed force control for FP32 mode {model_mode}')
    return original_environment(arm, model_mode, log_level)


runner.environment = environment

if __name__ == '__main__':
    runner.main()
