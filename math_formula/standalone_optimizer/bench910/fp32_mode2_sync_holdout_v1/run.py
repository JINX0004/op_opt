#!/usr/bin/env python3
"""Paired 191/192/193-KiB copy-, packet-, and panel-barrier sweep."""

import hashlib
import importlib.util
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
HERE = ROOT / 'fp32_mode2_sync_holdout_v1'
PRIOR = ROOT / 'fp32_mode2_transfer_holdout_v1'
spec = importlib.util.spec_from_file_location('audit_tiles', PRIOR / 'audit_source_tiles.py')
audit_tiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_tiles)
runner = audit_tiles.runner
BINARY = HERE / 'build/panel_ab_probe'
LOGICAL_K = 48384
PITCHES = (48896, 49152, 49408)
MODES = (0, 1, 2)


def one_geometry(item, env):
    m, n, device = item
    dest = HERE / 'runs' / f'{m}x{n}_device{device}'
    dest.mkdir(parents=True, exist_ok=True)
    record_path = dest / 'records.json'
    rows = json.loads(record_path.read_text()) if record_path.exists() else []
    if len(rows) == 18:
        return rows
    runner.require_idle(device, dest, 'before')
    audits = json.loads((PRIOR / 'tile_audits.json').read_text())
    tiles = [row['tile'] for row in audits if row['shape'][:2] == [m, n]]
    keys = ('singleCoreM', 'singleCoreN', 'singleCoreK',
            'baseM', 'baseN', 'baseK', 'stepM', 'stepN', 'block_dim')
    if len(tiles) != 3 or len({tuple(t[key] for key in keys) for t in tiles}) != 1:
        raise RuntimeError(('tile changed', item, tiles))
    tile = tiles[0]
    core_m, core_n, packet_k = (tile['singleCoreM'], tile['singleCoreN'],
                                tile['singleCoreK'])
    inner_m = tile['stepM'] * tile['baseM']
    a_cache_disable = int(core_n == n)
    for round_index, pitches in enumerate((PITCHES, tuple(reversed(PITCHES)))):
        modes = MODES if round_index == 0 else tuple(reversed(MODES))
        for pitch in pitches:
            for sync_mode in modes:
                if any(row['round'] == round_index and
                       row['pitch'] == pitch and row['sync_mode'] == sync_mode
                       for row in rows):
                    continue
                cmd = [str(BINARY), str(pitch), str(LOGICAL_K),
                       str(m), str(n), str(core_m), str(core_n),
                       str(inner_m), str(packet_k), str(a_cache_disable),
                       str(sync_mode), str(tile['block_dim']), str(device), '5']
                for attempt in range(3):
                    proc = subprocess.run(cmd, env=env, cwd=HERE, text=True,
                                          capture_output=True, timeout=600)
                    if proc.returncode == 0 or 'output token mismatch' not in proc.stderr:
                        break
                    with (dest / 'token_retries.log').open('a') as log:
                        log.write(f'{pitch} mode {sync_mode} round {round_index} '
                                  f'attempt {attempt}: {proc.stderr.strip()}\n')
                if proc.returncode:
                    raise RuntimeError((cmd, proc.stdout[-1000:], proc.stderr[-2000:]))
                row = json.loads(proc.stdout.strip().splitlines()[-1])
                row['round'] = round_index
                rows.append(row)
                record_path.write_text(json.dumps(rows, indent=2) + '\n')
                print('probe', m, n, pitch, sync_mode, round_index,
                      row['device_elapsed_us'], flush=True)
    runner.require_idle(device, dest, 'after')
    return rows


def main():
    (HERE / 'binary_sha256.txt').write_text(
        hashlib.sha256(BINARY.read_bytes()).hexdigest() + '\n')
    env = runner.CANN_ENV.copy()
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(lambda item: one_geometry(item, env),
                               audit_tiles.GEOMETRIES[:4]))
    groups.append(one_geometry(audit_tiles.GEOMETRIES[4], env))
    (HERE / 'probe_records.json').write_text(
        json.dumps([row for group in groups for row in group], indent=2) + '\n')


if __name__ == '__main__':
    main()
