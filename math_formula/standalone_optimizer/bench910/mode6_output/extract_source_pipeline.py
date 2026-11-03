#!/usr/bin/env python3
"""Extract source MatMulV3 device-4 pipeline counters from msprof output."""

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    'Task Duration(us)', 'aicore_time(us)', 'aic_mac_time(us)',
    'aic_mte1_time(us)', 'aic_mte2_time(us)', 'aic_fixpipe_time(us)',
    'aic_scalar_time(us)',
)
SHAPES = (
    '1792x768x27392', '2048x768x27392',
    '1792x768x32768', '2048x768x32768',
    '2048x2048x27392', '2048x2048x32768',
    '2048x3072x27392', '2048x3072x32768',
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    records = []
    for shape in SHAPES:
        arm = 'single' if int(shape.split('x')[1]) == 768 else 'default'
        for round_index in (0, 1):
            directory = args.experiment_dir / 'timing' / f'r{round_index}' / shape / arm
            case = json.loads((directory / 'case.json').read_text())
            assert case['split_core_mode'] == 6 and case['device'] == 4
            source_dir = Path(case['profile_csv']).parent
            summary_csv, = source_dir.glob('op_summary_*.csv')
            with summary_csv.open(newline='') as stream:
                rows = [row for row in csv.DictReader(stream)
                        if row['OP Type'] == 'MatMulV3'][-8:]
            assert len(rows) == 8
            medians = {metric: statistics.median(float(row[metric])
                          for row in rows) for metric in METRICS}
            assert abs(medians['Task Duration(us)'] - case['kernel_median_us']) < 0.001
            records.append({
                'shape': list(map(int, shape.split('x'))), 'round': round_index,
                'k_slices': case['k_slices'],
                'output_tiles': case['output_tiles'],
                'source_core_m': case['singleCoreM'],
                'source_core_n': case['singleCoreN'],
                'metrics_us': medians,
                'profile_csv': str(summary_csv),
            })
    args.output.write_text(''.join(json.dumps(r, sort_keys=True) + '\n'
                                   for r in records))
    print(f'wrote {len(records)} source profile rows to {args.output}')


if __name__ == '__main__':
    main()
