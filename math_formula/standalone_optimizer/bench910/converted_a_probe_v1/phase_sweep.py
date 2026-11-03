#!/usr/bin/env python3
"""Four-device in-context AIC phase sweep for converted-A MatMulV3 kernels."""

import json
import os
import re
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
OUT = ROOT / 'converted_a_probe_v1/phase_sweep_v1'
EXE = ROOT / 'base_aclnn_bench'
PACKAGE = ROOT / 'ops_nn_phase_probe_install/vendors/phaseprobe_nn'
SOURCE = ROOT / 'ops_nn_phase_probe_source/matmul/mat_mul_v3'
CASES = [(65536, 'base'), (65536, 'split'),
         (73728, 'base'), (73728, 'split')]
MARKER = re.compile(r'MMV3_PHASE_(BASE|SPLITK) k=(\d+) wait_cycles=(\d+) cube_cycles=(\d+)')


def idle(device: int, directory: Path, name: str) -> None:
    output = subprocess.run(['npu-smi', 'info'], text=True,
                            capture_output=True, check=True).stdout
    (directory / f'npu_smi_{name}.txt').write_text(output)
    if f'No running processes found in NPU {device}' not in output:
        raise RuntimeError(f'device {device} occupied at {name}: {directory}')


def run(device: int, k: int, family: str, round_index: int) -> dict:
    directory = OUT / f'r{round_index}' / f'device{device}' / f'{family}_k{k}'
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    idle(device, directory, 'before')
    env = os.environ.copy()
    env.update({'IGNORE_INFER_ERROR': '1', 'ASCEND_GLOBAL_LOG_LEVEL': '1',
                'MMV3_SKIP_LEGACY_OP_CACHE': '1',
                'ASCEND_CUSTOM_OPP_PATH': str(PACKAGE),
                'LD_LIBRARY_PATH': f'{PACKAGE / "op_api/lib"}:{env.get("LD_LIBRARY_PATH", "")}'})
    for name in ('MMV3_RESEARCH_SPLITK_FAMILY', 'MMV3_ONLINE_SELECTOR',
                 'MMV3_ONLINE_ROW_READ_PROFILE', 'MMV3_ONLINE_TILE'):
        env.pop(name, None)
    if family == 'base':
        env['MMV3_ONLINE_SELECTOR'] = '1'
        env['MMV3_ONLINE_ROW_READ_PROFILE'] = '1'
    else:
        env['MMV3_RESEARCH_SPLITK_FAMILY'] = 'deterministic'
    cmd = [str(EXE), '--api', 'v3cached', '--device', str(device),
           '--m', '2560', '--n', '1024', '--k', str(k),
           '--warmup', '5', '--batches', '1', '--repetitions', '8']
    with (directory / 'stdout.log').open('w') as stdout, \
         (directory / 'stderr.log').open('w') as stderr:
        result = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise RuntimeError(f'benchmark failed rc={result.returncode}: {directory}')
    idle(device, directory, 'after')
    stdout = (directory / 'stdout.log').read_text()
    matches = [(int(m.group(2)), int(m.group(3)), int(m.group(4)))
               for m in MARKER.finditer(stdout)
               if m.group(1) == ('BASE' if family == 'base' else 'SPLITK')]
    if len(matches) < 8 or any(row[0] != k for row in matches):
        raise RuntimeError(f'phase markers missing or wrong K: {directory}')
    samples = matches[-8:]
    event_lines = [line for line in stdout.splitlines() if line.startswith('{"run_id":')]
    if len(event_lines) != 1:
        raise RuntimeError(f'expected one benchmark result: {directory}')
    event = json.loads(event_lines[0])
    if event['max_relative_error'] > .02:
        raise RuntimeError(f'output check failed: {directory}: {event["max_relative_error"]}')
    parsed = subprocess.run(['python3', str(ROOT / 'parse_tiling_log.py'),
                             '--pid', str(event['pid'])], text=True,
                            capture_output=True, check=True)
    tiling = json.loads(parsed.stdout)
    expected = ((0, 128, 256, 128, 256, 64) if family == 'base'
                else (3, 2560, 384, 128, 128, 128))
    actual = (tiling['tilingEnableSplitCore'], tiling['singleCoreM'],
              tiling['singleCoreN'], tiling['baseM'], tiling['baseN'],
              tiling['baseK'])
    if actual != expected or tiling['nd2nzA'] != 1 or tiling['nd2nzB'] != 0 \
            or str(SOURCE) not in tiling['tiling_source']:
        raise RuntimeError(f'wrong source tile: {directory}: {tiling}')
    row = {'device': device, 'k': k, 'family': family, 'round': round_index,
           'wait_cycles': [x[1] for x in samples],
           'cube_cycles': [x[2] for x in samples],
           'wait_median_cycles': statistics.median(x[1] for x in samples),
           'cube_median_cycles': statistics.median(x[2] for x in samples),
           'phase_total_median_cycles': statistics.median(x[1] + x[2] for x in samples),
           'event': event, 'tiling': tiling, 'log': str(directory / 'stdout.log')}
    result_path.write_text(json.dumps(row, indent=2) + '\n')
    print('result', device, k, family, round_index,
          row['wait_median_cycles'], row['cube_median_cycles'], flush=True)
    return row


OUT.mkdir(parents=True, exist_ok=True)
rows = []
for round_index in range(4):
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for index, (k, family) in enumerate(CASES):
            device = 4 + (index + round_index) % 4
            futures.append(pool.submit(run, device, k, family, round_index))
        for future in as_completed(futures):
            rows.append(future.result())
            (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
