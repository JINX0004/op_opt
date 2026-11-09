#!/usr/bin/env python3
"""Paired producer/consumer probe on the four correct fixed-work tiles."""

import hashlib
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
BINARY = HERE / 'build/mode2_consumer_probe'
PITCHES = (48896, 49152, 49408)
MODES = (0, 1, 2, 3)


def source_env():
    result = subprocess.run(
        ['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
         'bash', str(CANN)], capture_output=True, check=True,
    )
    return dict(item.decode().split('=', 1)
                for item in result.stdout.split(b'\0') if item)


def require_idle(device):
    info = subprocess.run(['npu-smi', 'info'], capture_output=True,
                          text=True, check=True).stdout
    if f'No running processes found in NPU {device}' not in info:
        raise RuntimeError(f'device {device} occupied')


def run_one(case, env):
    m, n, k = case['shape']
    tile = case['tile']
    device = case['device']
    assert k == 48384 and tile['block_dim'] == 20
    assert tile['stepM'] * tile['baseM'] >= tile['singleCoreM']
    directory = HERE / 'runs' / f'{m}x{n}_device{device}'
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / 'records.json'
    rows = json.loads(record_path.read_text()) if record_path.exists() else []
    # The 208x1024 A-only L0A path fails the output-token gate on both
    # devices 4 and 6. Keep the other independently checked modes; do not
    # treat a failed A-only timing as a measurement.
    case_modes = (0, 2, 3) if (m, n) == (2048, 2048) else MODES
    require_idle(device)
    for round_index in range(3):
        pitches = PITCHES if round_index == 0 else tuple(reversed(PITCHES))
        modes = case_modes if round_index == 0 else tuple(reversed(case_modes))
        for pitch in pitches:
            for mode in modes:
                if any(row['round'] == round_index and row['pitch'] == pitch
                       and row['mode'] == mode for row in rows):
                    continue
                cmd = [str(BINARY), str(pitch), str(k), str(m), str(n),
                       str(tile['singleCoreM']), str(tile['singleCoreN']),
                       str(tile['stepM'] * tile['baseM']),
                       str(tile['singleCoreK']),
                       str(int(tile['singleCoreN'] == n)), str(mode),
                       str(tile['block_dim']), str(device), '5']
                for attempt in range(3):
                    proc = subprocess.run(cmd, env=env, cwd=HERE,
                                          capture_output=True, text=True,
                                          timeout=600)
                    if proc.returncode == 0 or 'output token mismatch' not in proc.stderr:
                        break
                    with (directory / 'token_retries.log').open('a') as log:
                        log.write(f'{pitch} mode={mode} round={round_index} '
                                  f'attempt={attempt}: {proc.stderr.strip()}\n')
                if proc.returncode:
                    raise RuntimeError((cmd, proc.stdout[-1000:], proc.stderr[-2000:]))
                row = json.loads(proc.stdout.strip().splitlines()[-1])
                assert row['device'] == device and row['mode'] == mode
                row['round'] = round_index
                rows.append(row)
                record_path.write_text(json.dumps(rows, indent=2) + '\n')
                print(m, n, pitch, mode, round_index,
                      row['device_elapsed_us'], flush=True)
    require_idle(device)
    return rows


def main():
    cases = json.loads((HERE / 'fixed_work_summary.json').read_text())
    assert len(cases) == 4 and len({row['device'] for row in cases}) == 4
    env = source_env()
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(lambda case: run_one(case, env), cases))
    records = [row for group in groups for row in group]
    (HERE / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    (HERE / 'binary_sha256.txt').write_text(
        hashlib.sha256(BINARY.read_bytes()).hexdigest() + '\n')
    summary = []
    for case in cases:
        m, n, _ = case['shape']
        for mode in ((0, 2, 3) if (m, n) == (2048, 2048) else MODES):
            times = {
                pitch: statistics.median(
                    row['device_elapsed_us'] for row in records
                    if (row['m'], row['n'], row['pitch'], row['mode']) ==
                       (m, n, pitch, mode))
                for pitch in PITCHES
            }
            summary.append({
                'shape': case['shape'], 'mode': mode,
                'median_us': times,
                'center_excess_us': times[49152] -
                                    (times[48896] + times[49408]) / 2,
            })
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for row in summary:
        print(row['shape'][:2], 'mode', row['mode'],
              'excess', round(row['center_excess_us'], 1), flush=True)


if __name__ == '__main__':
    main()
