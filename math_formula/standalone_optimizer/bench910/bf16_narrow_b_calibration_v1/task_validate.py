#!/usr/bin/env python3
"""Confirm narrow-B packet slopes using AI Core task times, not host events."""

import csv
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/bf16_narrow_b_calibration_v1')
BINARIES = {
    'fp16': '/tmp/mmv3mte2sweep/build/l1_step_read_probe',
    'bf16': '/tmp/mmv3bf16narrow/build/l1_step_read_probe',
}


def require_idle():
    snapshot = subprocess.run(['npu-smi', 'info'], text=True,
                              capture_output=True, check=True).stdout
    if 'No running processes found in NPU 4' not in snapshot:
        raise RuntimeError('device 4 occupied')


def main():
    output = ROOT / 'task_sweep.jsonl'
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row['dtype'], row['cores'], row['copies']))
    for cores in (1, 20):
        for copies in (1, 25):
            for dtype in ('bf16', 'fp16'):
                if (dtype, cores, copies) in done:
                    continue
                require_idle()
                label = f'{dtype}_B16_K4096_c{cores}_p{copies}'
                directory = ROOT / 'profiles' / label
                directory.mkdir(parents=True, exist_ok=True)
                cmd = ['msprof', f'--output={directory}', '--task-time=on',
                       '--ai-core=on', BINARIES[dtype], 'B', '16', '4096',
                       '16', '16', '102400', str(copies), str(cores), '10', '8']
                with (directory / 'stdout.log').open('w') as stdout, \
                        (directory / 'stderr.log').open('w') as stderr:
                    run = subprocess.run(cmd, stdout=stdout, stderr=stderr,
                                         timeout=300)
                require_idle()
                if run.returncode:
                    raise RuntimeError(f'{label} exited {run.returncode}: '
                                       + (directory / 'stderr.log').read_text()[-1000:])
                files = list(directory.glob('PROF_*/mindstudio_profiler_output/task_time_*.csv'))
                if not files:
                    raise RuntimeError(f'{label}: no task CSV')
                with files[-1].open(newline='') as stream:
                    times = [float(row['task_time(us)']) for row in csv.DictReader(stream)
                             if row['kernel_name'].startswith('l1_step_read_kernel')
                             and row['kernel_type'] == 'AI_CORE']
                if len(times) < 8:
                    raise RuntimeError(f'{label}: only {len(times)} AI Core tasks')
                row = dict(dtype=dtype, operand='B', tile_axis=16, k_span=4096,
                           cores=cores, copies=copies,
                           task_times_us=times[-8:],
                           median_us=statistics.median(times[-8:]),
                           profile_csv=str(files[-1]))
                with output.open('a') as stream:
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
                done.add((dtype, cores, copies))
                print(label, f'{row["median_us"]:.3f} us', flush=True)


if __name__ == '__main__':
    main()
