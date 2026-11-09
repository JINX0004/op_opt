#!/usr/bin/env python3
"""Measure actual mode-2 MatMulV3 with fixed logical K and varied A row pitch."""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = ROOT / 'fp32_mode2_in_context_pitch_v1'
CANN = Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
K = 48384
PITCHES = (48896, 49152, 49408)
GEOMETRIES = ((2816, 1024, 4), (3072, 1024, 5),
              (2048, 2048, 6), (2816, 1536, 7), (8192, 4096, 5))
TILE_FIELDS = ('singleCoreM', 'singleCoreN', 'singleCoreK', 'baseM', 'baseN',
               'baseK', 'stepM', 'stepN', 'stepKa', 'stepKb', 'depthA1',
               'depthB1', 'block_dim', 'nd2nzA', 'nd2nzB',
               'tiling_key', 'tilingEnableSplitCore', 'workspace_bytes')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def source_cann_env():
    result = subprocess.run(['bash', '-c', 'source "$1/set_env.sh" >/dev/null 2>&1 && env -0',
                             'bash', str(CANN)], capture_output=True, check=True)
    return dict(item.decode().split('=', 1) for item in result.stdout.split(b'\0') if item)


def environment(package, log_level):
    env = source_cann_env()
    for key in ('MMV3_ONLINE_SELECTOR', 'MMV3_ONLINE_TILE', 'MMV3_ONLINE_BF16_EXPERIMENT'):
        env.pop(key, None)
    env.update(BENCH_CUSTOM_OPP_PATH=str(package), ASCEND_CUSTOM_OPP_PATH=str(package),
               MMV3_SKIP_LEGACY_OP_CACHE='1', IGNORE_INFER_ERROR='1',
               MMV3_RESEARCH_SPLITK_FAMILY='single',
               ASCEND_GLOBAL_LOG_LEVEL=str(log_level))
    env['LD_LIBRARY_PATH'] = str(package / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    return env


def require_idle(device):
    info = subprocess.run(['npu-smi', 'info'], capture_output=True,
                          text=True, check=True).stdout
    if f'No running processes found in NPU {device}' not in info:
        raise RuntimeError(f'device {device} occupied')


def harness_event(path):
    for line in reversed(path.read_text().splitlines()):
        if line.startswith('{'):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get('kind') == 'matmul_v3_event':
                return data
    raise RuntimeError(f'no benchmark event in {path}')


def benchmark_args(m, n, device, pitch, warmup, repetitions):
    return [str(HERE / 'fp32_pitch_bench'), '--api', 'v3cached',
            '--device', str(device), '--m', str(m), '--n', str(n),
            '--k', str(K), '--a-pitch', str(pitch), '--warmup', str(warmup),
            '--batches', '1', '--repetitions', str(repetitions)]


def invoke(m, n, device, pitch, package, *, round_index=None):
    directory = (HERE / 'results' / f'{m}x{n}' / str(pitch) /
                 ('audit' if round_index is None else f'round{round_index}'))
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    require_idle(device)
    audit = round_index is None
    env = environment(package, 1 if audit else 3)
    args = benchmark_args(m, n, device, pitch, 0 if audit else 5, 1 if audit else 5)
    cmd = args if audit else ['msprof', '--output=' + str(directory / 'msprof'),
                              '--task-time=on', '--ai-core=on', 'env'] + args
    stdout_path = directory / 'stdout.jsonl'
    stderr_path = directory / 'stderr.log'
    with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
        proc = subprocess.run(cmd, env=env, cwd=HERE, stdout=stdout,
                              stderr=stderr, timeout=1800)
    require_idle(device)
    if proc.returncode:
        raise RuntimeError(f'{m}x{n} pitch={pitch} round={round_index} exit={proc.returncode}: '
                           + stderr_path.read_text()[-1600:])
    event = harness_event(stdout_path)
    if (event['device'] != device or event['dtype'] != 'fp32' or
            event['k'] != K or event['a_pitch'] != pitch or
            event['max_relative_error'] > .02):
        raise RuntimeError(f'correctness or provenance failure: {event}')
    result = {'shape': [m, n, K], 'a_pitch': pitch, 'device': device,
              'round': round_index, 'event': event}
    if audit:
        parsed = subprocess.run(['python3', str(ROOT / 'parse_tiling_log.py'),
                                 '--pid', str(event['pid'])], env=env,
                                text=True, capture_output=True, check=True)
        tile = json.loads(parsed.stdout)
        plog = Path(tile['plog']).read_text(errors='replace')
        pattern = rf'fixed-work A pitch: logicalK={K} Ka={pitch} Kb={K}'
        if (tile['tilingEnableSplitCore'] != 2 or
                not tile.get('tiling_source', '').startswith(str(HERE / 'ops_nn_pitch_source')) or
                not re.search(pattern, plog)):
            raise RuntimeError(f'wrong research kernel or tiling: {tile}')
        result['tile'] = {key: tile.get(key) for key in TILE_FIELDS}
        result['plog'] = str(tile['plog'])
    else:
        parsed = subprocess.run(['python3', str(ROOT / 'bf16_online_vs_master/parse_profile.py'),
                                 str(directory / 'msprof'), '--last', '5',
                                 '--kernel-type', 'MIX_AIC'],
                                text=True, capture_output=True, check=True)
        result['profile'] = json.loads(parsed.stdout)
        if result['profile']['kernel_count'] != 5:
            raise RuntimeError(f'incorrect kernel count: {result["profile"]}')
    save(result_path, result)
    return result


def one_geometry(m, n, device, package, rounds):
    audits = {pitch: invoke(m, n, device, pitch, package) for pitch in PITCHES}
    reference = audits[PITCHES[0]]['tile']
    if any(row['tile'] != reference for row in audits.values()):
        raise RuntimeError(f'tile differs across pitches for {m}x{n}')
    runs = {pitch: [] for pitch in PITCHES}
    for round_index in range(rounds):
        order = PITCHES if round_index % 2 == 0 else tuple(reversed(PITCHES))
        for pitch in order:
            row = invoke(m, n, device, pitch, package, round_index=round_index)
            us = row['profile']['median_us']
            runs[pitch].append(us)
            print(m, n, pitch, round_index, us, flush=True)
    medians = {pitch: statistics.median(values) for pitch, values in runs.items()}
    excess = medians[PITCHES[1]] - (medians[PITCHES[0]] + medians[PITCHES[2]]) / 2
    return {'shape': [m, n, K], 'device': device, 'tile': reference,
            'round_us': runs, 'median_us': medians, 'center_excess_us': excess,
            'max_relative_error': max(row['event']['max_relative_error']
                                      for row in audits.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    # Geometry 4 is retained as a diagnostic but excluded by default: the
    # research Ka != Kb path faults for its wide-N, three-panel tile.
    parser.add_argument('--geometry', type=int, nargs='*', default=list(range(4)))
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if args.rounds <= 0:
        raise RuntimeError('--rounds must be positive')
    package = args.package.resolve()
    lib = package / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
    if not lib.is_file():
        raise RuntimeError(f'host tiling library absent: {lib}')
    save(HERE / 'package_fingerprint.json', {'package': str(package),
         'host_tiling_sha256': hashlib.sha256(lib.read_bytes()).hexdigest(),
         'benchmark_sha256': hashlib.sha256((HERE / 'fp32_pitch_bench').read_bytes()).hexdigest()})
    assigned = {}
    for index in args.geometry:
        if index < 0 or index >= len(GEOMETRIES):
            raise RuntimeError(f'invalid geometry index {index}')
        assigned.setdefault(GEOMETRIES[index][2], []).append(index)

    def one_device(indices):
        return [(index, one_geometry(*GEOMETRIES[index], package, args.rounds)) for index in indices]

    with ThreadPoolExecutor(max_workers=len(assigned)) as pool:
        groups = list(pool.map(one_device, assigned.values()))
    rows = [row for _, row in sorted((item for group in groups for item in group))]
    save(HERE / 'summary.json', rows)


if __name__ == '__main__':
    main()
