#!/usr/bin/env python3
"""Paired two-shape stage ladder and actual MatMulV3 fixed-work control."""

import hashlib
import importlib.util
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXED_SOURCE = ROOT / 'fp32_mode2_in_context_pitch_v1/run.py'
spec = importlib.util.spec_from_file_location('fixed', FIXED_SOURCE)
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
BINARY = HERE / 'build/mode2_stage_ladder'
SOURCE_BINARY = ROOT / 'fp32_mode2_cache_policy_v1/fp32_cache_bench'
SOURCE_PACKAGE = ROOT / 'fp32_mode2_in_context_pitch_v1/ops_nn_pitch_overlay_package'
PITCHES = (48896, 49152, 49408)
MODES = (0, 1, 2, 3)


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')


def invoke_probe(case, pitch, mode, round_index, env):
    m, n, k = case['shape']
    tile, device = case['tile'], case['device']
    path = HERE / 'runs' / f'{m}x{n}' / f'round{round_index}' / \
        f'pitch{pitch}_mode{mode}.json'
    if path.exists():
        return json.loads(path.read_text())
    args = [str(BINARY), str(pitch), str(k), str(m), str(n),
            str(tile['singleCoreM']), str(tile['singleCoreN']),
            str(tile['stepM'] * tile['baseM']), str(tile['singleCoreK']),
            str(mode), str(tile['block_dim']), str(device), '5']
    proc = subprocess.run(args, env=env, cwd=HERE, capture_output=True,
                          text=True, timeout=900)
    if proc.returncode:
        raise RuntimeError((args, proc.returncode, proc.stdout[-1000:],
                            proc.stderr[-2000:]))
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    if (row['pitch'], row['logical_k'], row['m'], row['n'], row['mode'],
            row['device'], row['cores']) != (pitch, k, m, n, mode, device, 20):
        raise RuntimeError(('wrong probe output', row))
    if mode == 3 and row['scratch_sample'] <= 0:
        raise RuntimeError(('Fixpipe did not write scratch', row))
    row['round'] = round_index
    save(path, row)
    return row


def invoke_source(case, pitch, round_index, env):
    m, n, k = case['shape']
    device = case['device']
    path = HERE / 'runs' / f'{m}x{n}' / f'round{round_index}' / \
        f'pitch{pitch}_source.json'
    if path.exists():
        return json.loads(path.read_text())
    args = [str(SOURCE_BINARY), '--api', 'v3cached', '--device', str(device),
            '--m', str(m), '--n', str(n), '--k', str(k),
            '--a-pitch', str(pitch), '--cache-policy', '0',
            '--warmup', '3', '--batches', '1', '--repetitions', '5']
    proc = subprocess.run(args, env=env, cwd=HERE, capture_output=True,
                          text=True, timeout=900)
    if proc.returncode:
        raise RuntimeError((args, proc.returncode, proc.stdout[-1000:],
                            proc.stderr[-2000:]))
    row = next(json.loads(line) for line in reversed(proc.stdout.splitlines())
               if line.startswith('{') and 'matmul_v3_event' in line)
    if (row['device'], row['m'], row['n'], row['k'], row['a_pitch'],
            row['cache_policy']) != (device, m, n, k, pitch, 0) or \
            row['max_relative_error'] > .02:
        raise RuntimeError(('wrong source output', row))
    row['round'] = round_index
    save(path, row)
    return row


def one_case(case, probe_env, source_env, rounds):
    m, n, _ = case['shape']
    device = case['device']
    fixed.require_idle(device)
    probes, controls = [], []
    for round_index in range(rounds):
        pitches = PITCHES if round_index % 2 == 0 else tuple(reversed(PITCHES))
        modes = MODES if round_index % 2 == 0 else tuple(reversed(MODES))
        for pitch in pitches:
            if round_index % 2 == 0:
                row = invoke_source(case, pitch, round_index, source_env)
                controls.append(row)
                print(m, n, pitch, 'source', round_index,
                      row['median_us'], flush=True)
            for mode in modes:
                row = invoke_probe(case, pitch, mode, round_index, probe_env)
                probes.append(row)
                print(m, n, pitch, mode, round_index,
                      row['device_elapsed_us'], flush=True)
            if round_index % 2 == 1:
                row = invoke_source(case, pitch, round_index, source_env)
                controls.append(row)
                print(m, n, pitch, 'source', round_index,
                      row['median_us'], flush=True)
    fixed.require_idle(device)
    return probes, controls


def center(values):
    return values[49152] - (values[48896] + values[49408]) / 2


def main():
    cases = [case for case in json.loads((HERE / 'fixed_work_summary.json').read_text())
             if tuple(case['shape'][:2]) in ((3072, 1024), (2048, 2048))]
    assert len(cases) == 2 and len({case['device'] for case in cases}) == 2
    probe_env = fixed.source_cann_env()
    source_env = fixed.environment(SOURCE_PACKAGE, 3)
    rounds = 4
    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(lambda case: one_case(case, probe_env,
                                                     source_env, rounds), cases))
    probes = [row for group, _ in groups for row in group]
    controls = [row for _, group in groups for row in group]
    assert len(probes) == 2 * 3 * 4 * rounds
    assert len(controls) == 2 * 3 * rounds
    save(HERE / 'records.json', probes)
    save(HERE / 'source_controls.json', controls)
    save(HERE / 'binary_hashes.json', {
        'probe': hashlib.sha256(BINARY.read_bytes()).hexdigest(),
        'source_benchmark': hashlib.sha256(SOURCE_BINARY.read_bytes()).hexdigest(),
        'source_host_tiler': hashlib.sha256((SOURCE_PACKAGE /
            'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so').read_bytes()).hexdigest(),
    })
    for case in cases:
        shape = case['shape']
        for mode in MODES:
            medians = {pitch: statistics.median(row['device_elapsed_us']
                for row in probes if row['m'] == shape[0] and row['n'] == shape[1]
                and row['mode'] == mode and row['pitch'] == pitch)
                for pitch in PITCHES}
            print(shape[:2], 'mode', mode, 'center', center(medians), flush=True)
        source_medians = {pitch: statistics.median(row['median_us']
            for row in controls if row['m'] == shape[0] and row['n'] == shape[1]
            and row['a_pitch'] == pitch) for pitch in PITCHES}
        print(shape[:2], 'source center', center(source_medians), flush=True)


if __name__ == '__main__':
    main()
