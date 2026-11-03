#!/usr/bin/env python3
"""Profile exact one-core BASE L1 read packets on Ascend 910B3 device 4."""
import csv
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/online_tile_basek_v1/mte2_measure')
ROOT.mkdir(parents=True, exist_ok=True)
BINARY = Path('/tmp/mmv3mte2/build/l1_step_read_probe')
rows = []
for operand in ('A', 'B'):
    for copies in (1, 13, 25):
        label = f'{operand}_p{copies}'
        profile = ROOT / label
        snapshot = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True, check=True).stdout
        (ROOT / f'npu_smi_before_{label}.txt').write_text(snapshot)
        if 'No running processes found in NPU 4' not in snapshot:
            raise RuntimeError(f'device 4 occupied before {label}')
        cmd = ['msprof', f'--output={profile}', '--task-time=on', '--ai-core=on',
               str(BINARY), operand, '16', '4096', '16', '16', '102400',
               str(copies), '1', '10', '8']
        with (ROOT / f'{label}.stdout').open('w') as out, (ROOT / f'{label}.stderr').open('w') as err:
            proc = subprocess.run(cmd, stdout=out, stderr=err, timeout=240)
        if proc.returncode:
            raise RuntimeError(f'{label} failed rc={proc.returncode}: {(ROOT / f"{label}.stderr").read_text()[-1500:]}')
        snapshot = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True, check=True).stdout
        (ROOT / f'npu_smi_after_{label}.txt').write_text(snapshot)
        if 'No running processes found in NPU 4' not in snapshot:
            raise RuntimeError(f'device 4 occupied after {label}')
        files = list(profile.glob('PROF_*/mindstudio_profiler_output/task_time_*.csv'))
        if not files:
            raise RuntimeError(f'missing task profile: {label}')
        with files[-1].open(newline='') as source:
            times = [float(row['task_time(us)']) for row in csv.DictReader(source)
                     if row['kernel_name'].startswith('l1_step_read_kernel')
                     and row['kernel_type'] == 'AI_CORE']
        if len(times) < 8:
            raise RuntimeError(f'insufficient task rows: {label}: {times}')
        row = {'operand': operand, 'copies': copies, 'cores': 1,
               'tile_axis': 16, 'k_span': 4096, 'shape': [16, 16, 102400],
               'task_times_us': times[-8:], 'median_us': statistics.median(times[-8:]),
               'profile_csv': str(files[-1])}
        rows.append(row)
        (ROOT / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(label, f'{row["median_us"]:.3f} us', flush=True)
