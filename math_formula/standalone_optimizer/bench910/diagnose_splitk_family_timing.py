#!/usr/bin/env python3
"""Decompose the historical reduced mode-6 objective at executed 910B3 tiles.

The four wide audited cases use one spatial tile per AIC and one M panel
per tile. This script mirrors ScMode6ReadUs/ScMode6MacUs/ScKChainUs in
mip_solver.cpp for exactly that geometry, and checks its total against the
archived native ReducedObjectiveUs CSV. The deterministic baseline retains
the old per-baseK startup so that CSV remains reproducible; the current
solver counts startup per L1 K step only when A is converted. Ablations
are sensitivity calculations.
"""

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).parent
AUDIT = HERE / 'splitk_family_selection_device4_v1/audit'
SUMMARY = HERE / 'splitk_family_selection_device4_v1/summary.json'
SCORES = HERE / 'splitk_family_source_geometry_scores.csv'

LAUNCH = 2.68
READ_RATE = 3710212.0
READ_FIXED = 0.109620
WRITE_RATE = 5229502.0
WRITE_FIXED = 0.162586
CUBE_RATE = 14719432.0
SC_STORE_FIXED = 0.2432
SC_STORE_RATE = 34190.0
SC_ATOMIC_FIXED = 0.0541
SC_ATOMIC_RATE = 106618.0
SC_A_FIXED = 0.3124
SC_A_RATE = 270810.0
SC_B_FIXED = 0.0227
SC_B_RATE = 115414.0
CAST_FIXED = 0.0156
CAST_RATE = 116718.0


def ceildiv(a, b):
    return (a + b - 1) // b


def axis_packets(shape, core, base):
    full, tail = divmod(shape, core)
    return full * ceildiv(core, base) + (ceildiv(tail, base) if tail else 0)


def deterministic(shape, tiling, *, zero_packet_fixed=False,
                  legacy_base_packet_startup=False):
    m, n, k = shape
    cm, cn = tiling['singleCoreM'], tiling['singleCoreN']
    bm, bn = tiling['baseM'], tiling['baseN']
    step, cores = tiling['singleCoreK'] // tiling['baseK'], tiling['block_dim']
    q = ceildiv(k, 128)
    p = ceildiv(q, step)
    rounds = ceildiv(p, cores)
    q_crit = min(q, rounds * step)
    k_crit = min(k, q_crit * 128)
    n_outer = m > n
    outer_shape, outer_core = (n, cn) if n_outer else (m, cm)
    extents = [outer_shape] if outer_core >= outer_shape else \
        [outer_core] * (outer_shape // outer_core) + \
        ([outer_shape % outer_core] if outer_shape % outer_core else [])
    cube_stages, read_stages, vector_stages = [], [], []
    for extent in extents:
        me, ne = (m, extent) if n_outer else (extent, n)
        mp = axis_packets(m, cm, bm) if n_outer else ceildiv(me, bm)
        np = ceildiv(ne, bn) if n_outer else axis_packets(n, cn, bn)
        spatial = mp * np
        cube = 2 * me * ne * k_crit / CUBE_RATE
        operand_bytes = 2 * k * (me * np + ne * mp)
        read_bytes = operand_bytes / READ_RATE
        read_groups = (q_crit if legacy_base_packet_startup or k <= 65535
                       else ceildiv(q_crit, step))
        read_fixed = 0 if zero_packet_fixed else 2 * spatial * read_groups * READ_FIXED
        shard = ceildiv(me * ne, min(2 * cores, me * ne))
        vector = .1 + (cores - 1) * (1.28 + shard / 1e6) + WRITE_FIXED
        cube_stages.append(cube)
        read_stages.append((read_bytes, read_fixed))
        vector_stages.append(vector)
    stages = [max(cube, sum(read)) for cube, read in zip(cube_stages, read_stages)]
    stage = stages[0]
    for index in range(1, len(stages)):
        stage += max(stages[index], vector_stages[index - 1])
    return {'panels': len(extents), 'cube_full_panel_us': cube_stages[0],
            'read_bytes_full_panel_us': read_stages[0][0],
            'read_fixed_full_panel_us': read_stages[0][1],
            'vector_full_panel_us': vector_stages[0],
            'total_us': LAUNCH + stage + vector_stages[-1]}


def mode6(m, n, k_total, k_packet, *, generic_store=False,
          no_atomic=False, full_copy_overlap=False):
    mb, nb = ceildiv(m, 128), ceildiv(n, 128)
    packets = ceildiv(k_total, k_packet)
    tail = k_total - (packets - 1) * k_packet
    store_fixed = WRITE_FIXED if generic_store else SC_STORE_FIXED
    store_rate = WRITE_RATE if generic_store else SC_STORE_RATE
    def store(mm, nn):
        return ceildiv(mm, 128) * ceildiv(nn, 128) * store_fixed + \
            4 * mm * nn / store_rate
    def atomic(mm, nn):
        if no_atomic:
            return 0.0
        return ceildiv(mm, 128) * ceildiv(nn, 128) * SC_ATOMIC_FIXED + \
            4 * mm * nn / SC_ATOMIC_RATE
    def copy(kk):
        return 2 * kk * m / SC_A_RATE + mb * SC_A_FIXED + \
               2 * kk * n / SC_B_RATE + nb * SC_B_FIXED
    def mac(kk, is_atomic):
        cube = 2 * m * n * kk / CUBE_RATE
        output = store(m, n) + (atomic(m, n) if is_atomic else 0)
        mm, nn = min(m, 128), min(n, 128)
        one_piece = 2 * mm * nn * kk / CUBE_RATE + store(mm, nn) + \
            (atomic(mm, nn) if is_atomic else 0)
        return max(cube, output, one_piece)
    copy_full = copy(k_packet)
    mac_plain = mac(k_packet, False)
    mac_atomic = mac(k_packet, True)
    overlap = 2 * m * min(n, 128) * k_packet / CUBE_RATE
    residual = 0 if full_copy_overlap else max(0, copy_full - overlap)
    tail_residual = 0 if full_copy_overlap else max(0, copy(tail) - overlap)
    chain = copy_full + mac_plain + (packets - 2) * (mac_atomic + residual) + \
        tail_residual + mac(tail, True)
    shard = ceildiv(m, 2) * n
    vector = .1 + 4 * shard / READ_RATE + 2 * shard / WRITE_RATE + \
        WRITE_FIXED + CAST_FIXED + shard / CAST_RATE
    return {
        'packets': packets,
        'copy_full_us': copy_full,
        'cube_full_us': 2 * m * n * k_packet / CUBE_RATE,
        'plain_store_us': store(m, n),
        'atomic_extra_us': atomic(m, n),
        'mac_atomic_us': mac_atomic,
        'copy_residual_us': residual,
        'vector_us': vector,
        'total_us': LAUNCH + chain + vector,
    }


def main():
    actual = {tuple(r['shape']): r for r in json.loads(SUMMARY.read_text())['records']}
    with SCORES.open() as stream:
        native = {(int(r['m']), int(r['n']), int(r['k']), int(r['mode'])):
                  float(r['source_geometry_model_us']) for r in csv.DictReader(stream)
                  if not r['source_geometry_model_us'].startswith('error')}
    rows = []
    for shape, record in actual.items():
        if shape[1] == 768:
            continue
        tiling = json.loads((AUDIT / 'x'.join(map(str, shape)) /
                             'single/result.json').read_text())['tiling']
        det_tiling = json.loads((AUDIT / 'x'.join(map(str, shape)) /
                                 'deterministic/result.json').read_text())['tiling']
        m, n, k = tiling['singleCoreM'], tiling['singleCoreN'], shape[2]
        assert tiling['tilingEnableSplitCore'] == 6
        assert tiling['stepM'] * 128 >= m
        assert ceildiv(shape[0], m) * ceildiv(shape[1], n) == tiling['block_dim']
        baseline = mode6(m, n, k, tiling['singleCoreK'])
        assert math.isclose(baseline['total_us'], native[shape + (6,)], abs_tol=1e-3)
        det_baseline = deterministic(
            shape, det_tiling, legacy_base_packet_startup=True)
        assert math.isclose(det_baseline['total_us'], native[shape + (3,)], abs_tol=1e-3)
        variants = {
            'no_atomic': mode6(m, n, k, tiling['singleCoreK'], no_atomic=True),
            'generic_store': mode6(m, n, k, tiling['singleCoreK'], generic_store=True),
            'copy_fully_hidden': mode6(m, n, k, tiling['singleCoreK'], full_copy_overlap=True),
            'generic_store_copy_hidden': mode6(m, n, k, tiling['singleCoreK'],
                                               generic_store=True, full_copy_overlap=True),
        }
        rows.append({'shape': shape, 'source_tile': [m, n],
                     'actual_mode6_us': record['single_us'],
                     'actual_deterministic_us': record['deterministic_us'],
                     'native_mode6_us': native[shape + (6,)],
                     'native_deterministic_us': native[shape + (3,)],
                     'baseline': baseline,
                     'deterministic_baseline': det_baseline,
                     'deterministic_grouped_startup_us': deterministic(
                         shape, det_tiling)['total_us'],
                     'deterministic_no_read_fixed_us': deterministic(
                         shape, det_tiling, zero_packet_fixed=True)['total_us'],
                     'ablations_us': {name: v['total_us'] for name, v in variants.items()}})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
