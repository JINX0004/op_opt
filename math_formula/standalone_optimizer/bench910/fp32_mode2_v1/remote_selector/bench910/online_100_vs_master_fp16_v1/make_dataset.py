#!/usr/bin/env python3
"""Extend the 72-shape FP16 master comparison to 100 unique shapes."""

import ast
import json
from pathlib import Path


HERE = Path(__file__).parent
PREVIOUS = HERE.parent / 'online_expanded_vs_master_fp16_v1'


def previous_64():
    module = ast.parse((PREVIOUS / 'run.py').read_text())
    assignment = next(node for node in module.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'SHAPES'
                              for target in node.targets))
    return [list(shape) for shape in ast.literal_eval(assignment.value)]


NEW = [
    # Narrow outputs, conversion boundary, and long K.
    [16, 64, 65536], [32, 128, 65536], [64, 128, 73728],
    [128, 256, 102400], [256, 256, 65536], [512, 512, 65536],
    [1024, 128, 102400],
    # Family-transition neighborhood and orientation.
    [1664, 768, 32768], [1920, 768, 32768],
    [2176, 768, 32768], [2432, 768, 32768],
    [2304, 1024, 32768], [2816, 1024, 49152],
    [1024, 2432, 32768],
    # The source mode-5 window and its upper BASE boundary.
    [47104, 384, 1536], [51200, 384, 1536],
    [63488, 384, 1536], [77824, 384, 1536],
    [83968, 384, 1536], [94208, 384, 1536],
    [65536, 448, 1536],
    # Larger M/N workloads, including swapped orientations.
    [3072, 8192, 8192], [8192, 3072, 8192],
    [12288, 2048, 16384], [2048, 12288, 16384],
    [16384, 8192, 4096], [8192, 16384, 4096],
    [10240, 10240, 4096],
]


def main():
    shapes = previous_64() + json.loads(
        (PREVIOUS / 'mode5_boundary_shapes.json').read_text()) + NEW
    if len(shapes) != 100 or len({tuple(shape) for shape in shapes}) != 100:
        raise ValueError('expected exactly 100 distinct shapes')
    if any(len(shape) != 3 or min(shape) <= 0 or
           any(value % 16 for value in shape) for shape in shapes):
        raise ValueError('all dimensions must be positive multiples of 16')
    (HERE / 'dataset.json').write_text(json.dumps(shapes, indent=2) + '\n')
    print('wrote 100 unique FP16 shapes')


if __name__ == '__main__':
    main()
