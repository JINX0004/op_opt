#!/usr/bin/env python3
"""Analyze the fresh FP32 rerun with the original paired protocol."""

import importlib.util
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/analyze.py'
if not SOURCE.exists():
    SOURCE = Path(__file__).resolve().parent.parent / 'fp32_100_source_family_v1/analyze.py'
spec = importlib.util.spec_from_file_location('fp32_100_source_analyzer', SOURCE)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)

analyzer.HERE = Path(__file__).resolve().parent
analyzer.RAW = ROOT / 'cases/fp32_100_rerun_20261106'

if __name__ == '__main__':
    analyzer.main()
