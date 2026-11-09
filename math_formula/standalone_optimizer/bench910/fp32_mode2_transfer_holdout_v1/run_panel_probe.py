#!/usr/bin/env python3
"""Measure source-shaped A/B service for audited FP32 mode-2 geometries."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = ROOT / 'fp32_mode2_transfer_holdout_v1'
spec = importlib.util.spec_from_file_location('audit_tiles', HERE / 'audit_source_tiles.py')
audit_tiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_tiles)
runner = audit_tiles.runner
BINARY = HERE / 'build/panel_ab_probe'
PITCHES = audit_tiles.KS
LOGICAL_K = 48384  # divisible by both 192 and 256, fixed across pitches


def one_geometry(item, env):
    m, n, device = item
    dest = HERE / 'probe_runs' / f'{m}x{n}_device{device}'
    dest.mkdir(parents=True, exist_ok=True)
    record_path = dest / 'records.json'
    previous = json.loads(record_path.read_text()) if record_path.exists() else []
    if len(previous) == 6:
        return previous
    runner.require_idle(device, dest, 'before')
    audits = [runner.audit(k, [m, n, k], device, 'sweep', 2)['tiling']
              for k in PITCHES]
    keys = ('singleCoreM', 'singleCoreN', 'singleCoreK',
            'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'block_dim')
    if any(tuple(t[key] for key in keys) != tuple(audits[0][key] for key in keys)
           for t in audits[1:]):
        raise RuntimeError(('tiling changed across K', item, audits))
    tile = audits[0]
    if tile['tilingEnableSplitCore'] != 2 or tile['baseN'] != 128:
        raise RuntimeError((item, tile))
    core_m, core_n, packet_k = (tile['singleCoreM'], tile['singleCoreN'],
                                tile['singleCoreK'])
    inner_m = tile['stepM'] * tile['baseM']
    a_cache_disable = int(core_n == n)
    if LOGICAL_K % packet_k or inner_m > 384:
        raise RuntimeError((item, tile))
    rows = previous
    for round_index, order in enumerate((PITCHES, tuple(reversed(PITCHES)))):
        for pitch in order:
            if any(row['round'] == round_index and row['pitch'] == pitch
                   for row in rows):
                continue
            cmd = [str(BINARY), str(pitch), str(LOGICAL_K),
                   str(m), str(n), str(core_m), str(core_n),
                   str(inner_m), str(packet_k), str(a_cache_disable),
                   str(tile['block_dim']), str(device), '5']
            for attempt in range(3):
                proc = subprocess.run(cmd, env=env, cwd=HERE, text=True,
                                      capture_output=True, timeout=600)
                if proc.returncode == 0 or 'output token mismatch' not in proc.stderr:
                    break
                (dest / 'token_retries.log').open('a').write(
                    f'{pitch} round {round_index} attempt {attempt}: '
                    f'{proc.stderr.strip()}\n')
            if proc.returncode:
                raise RuntimeError((cmd, proc.stdout[-1000:], proc.stderr[-2000:]))
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            row['round'] = round_index
            rows.append(row)
            record_path.write_text(json.dumps(rows, indent=2) + '\n')
            print('probe', m, n, pitch, round_index,
                  row['device_elapsed_us'], flush=True)
    runner.require_idle(device, dest, 'after')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=('first', 'remaining', 'all'),
                        default='all')
    args = parser.parse_args()
    (HERE / 'binary_sha256.txt').write_text(
        hashlib.sha256(BINARY.read_bytes()).hexdigest() + '\n')
    env = runner.CANN_ENV.copy()
    first = (audit_tiles.GEOMETRIES[0], audit_tiles.GEOMETRIES[2],
             audit_tiles.GEOMETRIES[3])
    remaining = (audit_tiles.GEOMETRIES[1], audit_tiles.GEOMETRIES[4])
    groups = []
    if args.group in ('first', 'all'):
        with ThreadPoolExecutor(max_workers=3) as pool:
            groups.extend(pool.map(lambda item: one_geometry(item, env), first))
    if args.group in ('remaining', 'all'):
        groups.extend(one_geometry(item, env) for item in remaining)
    (HERE / f'probe_records_{args.group}.json').write_text(
        json.dumps([row for group in groups for row in group], indent=2) + '\n')
    if args.group == 'all':
        (HERE / 'probe_records.json').write_text(
            json.dumps([row for group in groups for row in group], indent=2) + '\n')


if __name__ == '__main__':
    main()
