#!/usr/bin/env python3
"""Reuse supported FP16 shapes and add FP32 family-boundary coverage."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
FP16 = BENCH / 'online_100_vs_master_fp16_v1/dataset.json'
BOUNDARY = BENCH / 'fp32_family_boundary_v1/dataset.json'

EXTRA = [
    [16, 16, 6080], [16, 16, 6144], [16, 16, 6208],
    [64, 64, 6080], [64, 64, 6144], [64, 64, 6208],
    [128, 128, 7616], [128, 128, 7680], [128, 128, 7744],
    [256, 256, 7616], [256, 256, 7680], [256, 256, 7744],
    [512, 512, 27328], [512, 512, 27392], [512, 512, 27456],
    [2048, 64, 27328], [2048, 64, 27392],
    [4096, 64, 27328], [4096, 64, 27392], [4096, 64, 27456],
]


def main():
    fp16 = json.loads(FP16.read_text())
    supported = [shape for shape in fp16 if shape[2] <= 65535 and
                 shape[2] % 64 == 0 and shape[1] <= 65535 and
                 (shape[1] % 64 == 0 or shape[1] in (16, 32, 48, 96)) and
                 shape != [8192, 16384, 4096]]
    boundary = {tuple(row['shape']) for row in json.loads(BOUNDARY.read_text())}
    assert len(supported) == 80 and len(EXTRA) == 20
    assert all(tuple(shape) in boundary for shape in EXTRA)
    shapes = supported + EXTRA
    assert len(shapes) == 100 and len({tuple(shape) for shape in shapes}) == 100
    (HERE / 'dataset.json').write_text(json.dumps(shapes, indent=2) + '\n')
    (HERE / 'groups.json').write_text(json.dumps({
        'fp16_supported': [0, 80], 'fp32_boundary': [80, 100]}, indent=2) + '\n')
    print('wrote 100 unique model-supported FP32 shapes')


if __name__ == '__main__':
    main()
