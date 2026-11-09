#!/usr/bin/env python3
"""Preserve online selector decisions from the source-backed CANN audit logs."""

import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_RAW = Path('/home/xianfei/solvebase_bench_20261025/cases/bf16_source_base_v1')
HERE = Path(__file__).resolve().parent
PATTERN = re.compile(r'Online MIP decision: mode=(\d+), estimated_us=([0-9.eE+-]+), '
                     r'shape=(\d+) x (\d+) x (\d+)')


def source_base_steps(shape, tile):
    """Mirror SourceBaseL1Steps for the plain 910B3 BF16 BASE branch."""
    k = shape[2]
    base_m, base_n, base_k = (tile[key] for key in ('baseM', 'baseN', 'baseK'))
    l1 = 524032 + 256
    packet_a, packet_b = 2 * base_m * base_k, 2 * base_n * base_k
    depth_a, depth_b = l1 // 2 // packet_a, l1 // 2 // packet_b
    if depth_a * packet_a + depth_b * packet_b > l1:
        if base_m <= base_n:
            depth_a //= 2
        else:
            depth_b //= 2
    step_a, step_b = depth_a // 2, depth_b // 2

    def align(step):
        if step * base_k >= k:
            return step
        span_bytes = step * base_k * 2
        quantum = 512 if span_bytes > 512 else 256 if span_bytes > 256 else 0
        if quantum and quantum % (base_k * 2) == 0:
            while step > 1 and (step * base_k * 2) % quantum:
                step -= 1
        return step

    step_a, step_b = align(step_a), align(step_b)
    if step_a >= step_b:
        step_a = step_a // step_b * step_b
    else:
        step_b = step_b // step_a * step_a
    return step_a, step_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    rows = []
    for record_path in sorted(args.raw.glob('device*/*/record.json')):
        record = json.loads(record_path.read_text())
        tiling = record['audits']['online']['tiling']
        log = Path(tiling['plog'])
        decisions = []
        for line in log.open(errors='replace'):
            match = PATTERN.search(line)
            if match:
                decisions.append(match)
        if len(decisions) > 1:
            raise RuntimeError(f'{record_path}: expected at most one online decision')
        if decisions:
            mode, estimated, m, n, k = decisions[0].groups()
            if ([int(m), int(n), int(k)] != record['shape'] or
                    int(mode) != tiling['tilingEnableSplitCore']):
                raise RuntimeError(f'{record_path}: logged decision differs from emitted tile')
        else:
            # Source branches that bypass DoSelectTiling still belong in the
            # 100-shape comparison, but must not be counted as MIP decisions.
            if record['changed_tiling']:
                raise RuntimeError(f'{record_path}: changed tile without an online decision')
            mode = tiling['tilingEnableSplitCore']
            estimated = ''
        if int(mode) == 0:
            expected = source_base_steps(record['shape'], tiling)
            actual = tiling['stepKa'], tiling['stepKb']
            if expected != actual:
                raise RuntimeError(f'{record_path}: source BASE K steps {actual} '
                                   f'differ from model steps {expected}')
        rows.append({'index': record['index'], 'shape': 'x'.join(map(str, record['shape'])),
                     'device': record['device'], 'mode': int(mode),
                     'online_decision': bool(decisions),
                     'estimated_us': float(estimated) if decisions else ''})
    rows.sort(key=lambda row: row['index'])
    if len(rows) != 100 or len({row['index'] for row in rows}) != 100:
        raise RuntimeError(f'incomplete decision audit: {len(rows)}/100')
    with (HERE / 'decision_audit.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print('validated', sum(row['online_decision'] for row in rows),
          'online decisions and', sum(not row['online_decision'] for row in rows),
          'source bypasses')


if __name__ == '__main__':
    main()
