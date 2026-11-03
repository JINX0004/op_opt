#!/usr/bin/env python3
"""Export compact verified timing evidence without copying msprof archives."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.experiment / 'timing').rglob('case.json')):
        directory = path.parent
        if (directory / 'interference.flag').exists():
            raise RuntimeError(f'interference flagged: {directory}')
        for phase in ('before', 'after'):
            snapshot = (directory / f'npu_smi_{phase}.txt').read_text()
            if f'No running processes found in NPU {args.device}' not in snapshot:
                raise RuntimeError(f'device busy at {phase}: {directory}')
        case = json.loads(path.read_text())
        tiling = json.loads((directory / 'tiling.json').read_text())
        profile = json.loads((directory / 'kernel_time.json').read_text())
        if (case['device'] != args.device or case['kernel_count'] != 8 or
                case['split_core_mode'] != tiling['tilingEnableSplitCore'] or
                profile['kernel_count'] != 8 or case['max_relative_error'] > 0.02):
            raise RuntimeError(f'invalid case: {directory}')
        rows.append({'round': int(directory.parent.parent.name.removeprefix('r')),
                     'shape': case['shape'], 'arm': directory.name,
                     'case': case, 'tiling': tiling, 'profile': profile})
    print('\n'.join(json.dumps(row, sort_keys=True) for row in rows))


if __name__ == '__main__':
    main()
