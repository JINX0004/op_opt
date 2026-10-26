#!/usr/bin/env python3
"""Freeze a source-checked packet SolveBase selection before repeat timings."""

import argparse
import hashlib
import json
from pathlib import Path


SHAPES = [
    (512, 4096, 8192), (4096, 512, 8192),
    (1024, 4096, 4096), (4096, 1024, 4096),
    (2048, 6144, 4096), (6144, 2048, 4096),
    (1536, 3072, 8192), (3072, 1536, 8192),
    (256, 8192, 8192), (8192, 256, 8192),
    (384, 6144, 8192), (6144, 384, 8192),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--defaults', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shapes-file', type=Path,
                        help='JSON list of declared [M,N,K] shapes')
    args = parser.parse_args()
    shapes = [tuple(shape) for shape in
              json.loads(args.shapes_file.read_text())] \
        if args.shapes_file is not None else SHAPES
    defaults = {tuple(row['shape']): row for row in
                map(json.loads, args.defaults.read_text().splitlines())}
    if set(defaults) != set(shapes):
        raise ValueError('default audit does not cover declared shapes')
    audit = {}
    for line in args.audit.read_text().splitlines():
        parts = line.split(',')
        if len(parts) < 11 or parts[3] != 'BASE':
            raise ValueError(f'bad solver audit line: {line}')
        shape = tuple(map(int, parts[:3]))
        audit[shape] = {'tile': list(map(int, parts[4:10])),
                        'objective_us': float(parts[10])}
    if set(audit) != set(shapes):
        raise ValueError('solver audit does not cover declared shapes')
    rows = []
    for shape in shapes:
        default = defaults[shape]
        if default['status'] != 'eligible':
            rows.append({'shape': shape, 'status': 'ineligible',
                         'reason': default['reason']})
            continue
        default_tile = default['tile']
        optimizer_tile = audit[shape]['tile']
        rows.append({'shape': shape,
                     'status': 'same_tile' if default_tile == optimizer_tile
                               else 'selected',
                     'default_tile': default_tile,
                     'optimizer_tile': optimizer_tile,
                     'optimizer_objective_us': audit[shape]['objective_us']})
    result = {'declared_shapes': shapes,
              'default_audit_sha256': sha256(args.defaults),
              'solver_audit_sha256': sha256(args.audit),
              'rows': rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    for row in rows:
        print(tuple(row['shape']), row['status'],
              row.get('default_tile'), '->', row.get('optimizer_tile'))


if __name__ == '__main__':
    main()
