#!/usr/bin/env python3
"""Output-checked 2x2 A/B cache-hint experiment in actual mode-2 MatMulV3."""

import argparse
import glob
import hashlib
import importlib.util
import json
import re
import sqlite3
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'fp32_mode2_in_context_pitch_v1/run.py'
spec = importlib.util.spec_from_file_location('fixed_work_runner', SOURCE)
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
PACKAGE = HERE / 'ops_nn_cache_policy_package'
PRIOR_PACKAGE = ROOT / 'fp32_mode2_in_context_pitch_v1/ops_nn_pitch_overlay_package'
BINARY = HERE / 'fp32_cache_bench'
PITCHES = (48896, 49152, 49408)
POLICIES = (0, 1, 2, 3)
METRICS = ('aic_total_time', 'aic_mte1_time', 'aic_mte2_time',
           'aic_mac_time', 'aic_fixpipe_time', 'aic_scalar_time')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def args_for(m, n, device, pitch, policy, warmup, repetitions):
    return [str(BINARY), '--api', 'v3cached', '--device', str(device),
            '--m', str(m), '--n', str(n), '--k', str(fixed.K),
            '--a-pitch', str(pitch), '--cache-policy', str(policy),
            '--warmup', str(warmup), '--batches', '1',
            '--repetitions', str(repetitions)]


def invoke(case, pitch, policy, round_index, control=False):
    m, n, _ = case['shape']
    device = case['device']
    arm = 'prior_control' if control else f'policy{policy}'
    directory = HERE / 'results' / f'{m}x{n}' / arm / str(pitch) / \
                ('audit' if round_index is None else f'round{round_index}')
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    fixed.require_idle(device)
    audit = round_index is None
    env = fixed.environment(PRIOR_PACKAGE if control else PACKAGE,
                            1 if audit else 3)
    args = args_for(m, n, device, pitch, policy,
                    0 if audit else 3, 1 if audit else 3)
    cmd = args if audit else [
        'msprof', '--output=' + str(directory / 'msprof'),
        '--task-time=on', '--ai-core=on',
        '--aic-metrics=PipeUtilization', 'env'] + args
    with (directory / 'stdout.log').open('w') as stdout, \
         (directory / 'stderr.log').open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=HERE,
                              stdout=stdout, stderr=stderr, timeout=1800)
    fixed.require_idle(device)
    if proc.returncode:
        raise RuntimeError((directory, proc.returncode,
                            (directory / 'stderr.log').read_text()[-2500:]))
    event = fixed.harness_event(directory / 'stdout.log')
    if (event['device'] != device or event['dtype'] != 'fp32' or
            event['m'] != m or event['n'] != n or event['k'] != fixed.K or
            event['a_pitch'] != pitch or event['cache_policy'] != policy or
            event['max_relative_error'] > .02):
        raise RuntimeError(('correctness/provenance failure', directory, event))
    data = {'shape': [m, n, fixed.K], 'device': device,
            'pitch': pitch, 'policy': policy, 'round': round_index,
            'control': control, 'max_relative_error': event['max_relative_error']}
    if audit:
        parsed = subprocess.run(['python3', str(ROOT / 'parse_tiling_log.py'),
                                 '--pid', str(event['pid'])],
                                env=env, text=True, capture_output=True,
                                check=True)
        tile = json.loads(parsed.stdout)
        log = Path(tile['plog']).read_text(errors='replace')
        original = ROOT / 'fp32_mode2_in_context_pitch_v1/ops_nn_pitch_source'
        if (tile['tilingEnableSplitCore'] != 2 or
                not tile.get('tiling_source', '').startswith(str(original)) or
                not re.search(rf'fixed-work A pitch: logicalK={fixed.K} '
                              rf'Ka={pitch} Kb={fixed.K}', log)):
            raise RuntimeError(('incorrect family/host tiler', directory, tile))
        data['tile'] = {key: tile.get(key) for key in fixed.TILE_FIELDS}
    else:
        databases = glob.glob(str(directory / 'msprof/**/ai_core_op_summary.db'),
                              recursive=True)
        if len(databases) != 1:
            raise RuntimeError(('missing profiler database', directory, databases))
        with sqlite3.connect(databases[0]) as conn:
            columns = [row[1] for row in conn.execute('pragma table_info(ai_core_metrics)')]
            values = conn.execute('select * from ai_core_metrics order by task_id').fetchall()
        if len(values) < 3:
            raise RuntimeError(('missing AIC tasks', directory, len(values)))
        stage = {}
        for metric in METRICS:
            index = columns.index(metric)
            stage[metric] = statistics.median(float(row[index]) for row in values[-3:])
        data['metrics'] = stage
        parsed = subprocess.run([
            'python3', str(ROOT / 'bf16_online_vs_master/parse_profile.py'),
            str(directory / 'msprof'), '--last', '3',
            '--kernel-type', 'MIX_AIC'],
            text=True, capture_output=True, check=True)
        profile = json.loads(parsed.stdout)
        if profile['kernel_count'] != 3:
            raise RuntimeError(('wrong task count', directory, profile))
        data['device_median_us'] = profile['median_us']
    save(result_path, data)
    return data


def one_geometry(case, rounds):
    m, n, _ = case['shape']
    audits = [invoke(case, pitch, policy, None)
              for pitch in PITCHES for policy in POLICIES]
    tile = case['tile']
    if any(row['tile'] != tile for row in audits):
        raise RuntimeError(('tile changed', (m, n), audits))
    records = []
    controls = []
    for round_index in range(rounds):
        pitches = PITCHES if round_index % 2 == 0 else tuple(reversed(PITCHES))
        policies = POLICIES if round_index % 2 == 0 else tuple(reversed(POLICIES))
        for pitch in pitches:
            if round_index % 2 == 0:
                control = invoke(case, pitch, 0, round_index, True)
                controls.append(control)
                print(m, n, pitch, 'prior', round_index,
                      control['device_median_us'], flush=True)
            for policy in policies:
                row = invoke(case, pitch, policy, round_index)
                records.append(row)
                print(m, n, pitch, policy, round_index,
                      row['device_median_us'], flush=True)
            if round_index % 2 == 1:
                control = invoke(case, pitch, 0, round_index, True)
                controls.append(control)
                print(m, n, pitch, 'prior', round_index,
                      control['device_median_us'], flush=True)
    return records, controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=4)
    args = parser.parse_args()
    if args.rounds < 1:
        raise ValueError('--rounds must be positive')
    cases = json.loads((HERE / 'fixed_work_summary.json').read_text())
    if len(cases) != 4 or len({row['device'] for row in cases}) != 4:
        raise RuntimeError('expected four source-audited devices')
    prior = json.loads((ROOT / 'fp32_mode2_in_context_pitch_v1/package_fingerprint.json').read_text())
    host = PACKAGE / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
    if hashlib.sha256(host.read_bytes()).hexdigest() != prior['host_tiling_sha256']:
        raise RuntimeError('host tiling library differs from fixed-work baseline')
    save(HERE / 'provenance.json', {
        'package': str(PACKAGE), 'host_tiling_sha256': prior['host_tiling_sha256'],
        'benchmark_sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest(),
    })
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(lambda case: one_geometry(case, args.rounds), cases))
    save(HERE / 'records.json', [row for group, _ in groups for row in group])
    save(HERE / 'control_records.json',
         [row for _, control in groups for row in control])


if __name__ == '__main__':
    main()
