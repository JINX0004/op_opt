#!/usr/bin/env python3
"""Recheck the old copy probe with the actual source's normal A/B hints."""

import importlib.util
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'fp32_mode2_in_context_pitch_v1/run.py'
spec = importlib.util.spec_from_file_location('fixed', SOURCE)
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
BINARY = ROOT / 'fp32_mode2_sync_holdout_v1/build/panel_ab_probe'
PITCHES = (48896, 49152, 49408)


def one(case, env):
    m, n, k = case['shape']
    tile = case['tile']
    device = case['device']
    directory = HERE / 'probe_hint_runs' / f'{m}x{n}_device{device}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'records.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    fixed.require_idle(device)
    for round_index in range(3):
        pitches = PITCHES if round_index % 2 == 0 else tuple(reversed(PITCHES))
        hints = (0, 1) if round_index % 2 == 0 else (1, 0)
        for pitch in pitches:
            for hint in hints:
                if any(row['round'] == round_index and row['pitch'] == pitch and
                       row['a_cache_disable'] == hint for row in rows):
                    continue
                cmd = [str(BINARY), str(pitch), str(k), str(m), str(n),
                       str(tile['singleCoreM']), str(tile['singleCoreN']),
                       str(tile['stepM'] * tile['baseM']),
                       str(tile['singleCoreK']), str(hint), '1',
                       str(tile['block_dim']), str(device), '5']
                for attempt in range(3):
                    proc = subprocess.run(cmd, env=env, cwd=HERE, text=True,
                                          capture_output=True, timeout=600)
                    if proc.returncode == 0 or 'output token mismatch' not in proc.stderr:
                        break
                    with (directory / 'token_retries.log').open('a') as log:
                        log.write(f'{round_index} {pitch} {hint} {attempt}: '
                                  f'{proc.stderr.strip()}\n')
                if proc.returncode:
                    raise RuntimeError((cmd, proc.stdout[-500:], proc.stderr[-1000:]))
                row = json.loads(proc.stdout.strip().splitlines()[-1])
                row['round'] = round_index
                rows.append(row)
                path.write_text(json.dumps(rows, indent=2) + '\n')
                print(m, n, pitch, hint, round_index,
                      row['device_elapsed_us'], flush=True)
    fixed.require_idle(device)
    return rows


def main():
    cases = json.loads((HERE / 'fixed_work_summary.json').read_text())
    env = fixed.source_cann_env()
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(lambda case: one(case, env), cases))
    records = [row for group in groups for row in group]
    (HERE / 'probe_hint_records.json').write_text(json.dumps(records, indent=2) + '\n')
    result = []
    for case in cases:
        m, n, _ = case['shape']
        for hint in (0, 1):
            times = {pitch: statistics.median(
                row['device_elapsed_us'] for row in records
                if (row['m'], row['n'], row['pitch'], row['a_cache_disable']) ==
                   (m, n, pitch, hint)) for pitch in PITCHES}
            excess = times[49152] - (times[48896] + times[49408]) / 2
            result.append({'shape': case['shape'], 'a_cache_disable': hint,
                           'median_us': times, 'center_excess_us': excess})
            print((m, n), 'A-disable', hint, 'excess', round(excess, 1),
                  flush=True)
    (HERE / 'probe_hint_summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
