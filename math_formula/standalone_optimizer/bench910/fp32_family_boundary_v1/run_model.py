#!/usr/bin/env python3
"""Build and score the FP32 boundary dataset with the remote reduced model."""

import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/xianfei/dev/ops-nn-sribd-mip-selector/experimental/mip_selector')
ROOT = Path('/home/xianfei/solvebase_bench_20261025')
OUT = ROOT / 'cases/fp32_family_boundary_v1'
MODEL = PROJECT / 'model'


def build(program, source):
    subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(PROJECT), '-I', str(MODEL),
                    str(source), str(MODEL / 'mip_solver.cpp'),
                    str(MODEL / 'plan_finalizer.cpp'), '-o', str(HERE / program)],
                   check=True)


def main():
    dataset = json.loads((HERE / 'dataset.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cmake', '--build', str(PROJECT / 'build'), '--target',
                    'mmopt_selector', '-j', '4'], check=True)
    build('family_scores', HERE / 'family_scores.cpp')
    build('score_source', PROJECT / 'bench910/fp32_plain_v1/score_source.cpp')
    shapes = ''.join('%d %d %d\n' % tuple(case['shape']) for case in dataset)
    model_shapes = ''.join('%d %d %d fp32\n' % tuple(case['shape']) for case in dataset)
    (OUT / 'model_decisions.csv').write_text(subprocess.run(
        [str(PROJECT / 'build/mmopt_selector')], input=model_shapes,
        capture_output=True, text=True, check=True).stdout)
    (OUT / 'family_scores.csv').write_text(subprocess.run(
        [str(HERE / 'family_scores')], input=shapes,
        capture_output=True, text=True, check=True).stdout)
    subprocess.run(['python3', str(HERE / 'summarize.py')], check=True)


if __name__ == '__main__':
    main()
