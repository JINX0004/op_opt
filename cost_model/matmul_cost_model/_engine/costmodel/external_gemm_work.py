"""Source-level work adapters for the fixed external GEMM validation harness.

These adapters do not infer a schedule from shape alone. They consume the
executed strategy and tile metadata of attention_matmul's frozen wrappers.
Source MMAD geometry remains a hypothesis until compared with PMU/lowering.
The legacy Cube L/II formula produces per-core active-service predictions,
never whole-kernel latency or a qualified practical ceiling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter
import math

from .calibration import CubePipelineFit


def ceildiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def catlass_split_factor(m, n, k, l1_tile, physical_aic=24):
    """Mirror the pinned SplitkMatmul::GetSplitkFactor for prediction freezing."""
    if any(type(v) is not int or v <= 0 for v in (m,n,k,physical_aic)):
        raise ValueError('positive integral launch geometry required')
    if l1_tile not in ([128,256,256], [16,16,4096]):
        raise ValueError('unreviewed CATLASS L1 tile')
    maximum = 2 if k <= 1024 else 4 if k <= 2048 else 8 if k <= 4096 else 16
    tiles = ceildiv(m,l1_tile[0])*ceildiv(n,l1_tile[1])
    split = max(1,min(physical_aic//tiles,maximum))
    if tiles < physical_aic:
        while split+1 <= maximum and ceildiv(tiles*split,physical_aic) >= ceildiv(tiles,physical_aic)*split:
            split += 1
    split = min(ceildiv(k,l1_tile[2]),split)
    if k > 8192:
        split = max(2,split)
    if k > 32768:
        split = max(4,split)
    return split


@dataclass
class CoreWork:
    core_id: int
    calls: Counter = field(default_factory=Counter)
    output_flushes: int = 0
    logical_a_read_bytes: int = 0
    logical_b_read_bytes: int = 0
    logical_output_write_bytes: int = 0

    def add_chunk(self, m, n, k, base, output_bytes, *, pad_mn=False):
        bm, bn, bk = base
        for mo in range(0, m, bm):
            tm = min(m - mo, bm)
            for no in range(0, n, bn):
                tn = min(n - no, bn)
                for ko in range(0, k, bk):
                    tk = min(k - ko, bk)
                    cm, cn = (ceildiv(tm,16)*16, ceildiv(tn,16)*16) if pad_mn else (tm,tn)
                    self.calls[(cm, cn, tk)] += 1
                self.output_flushes += 1
                self.logical_output_write_bytes += tm * tn * output_bytes
        # These are logical requests for this chunk, not cache-miss traffic.
        self.logical_a_read_bytes += m * k * 2
        self.logical_b_read_bytes += k * n * 2

    @property
    def cube_calls(self):
        return sum(self.calls.values())

    @property
    def cube_blocks(self):
        return sum(count * math.prod(ceildiv(v, 16) for v in shape)
                   for shape, count in self.calls.items())

    def predict_cube_cycles(self, fit: CubePipelineFit):
        return sum(count * fit.predict_cycles(math.prod(ceildiv(v, 16) for v in shape))
                   for shape, count in self.calls.items())

    def to_dict(self):
        return dict(core_id=self.core_id, cube_calls=self.cube_calls,
                    cube_blocks=self.cube_blocks,
                    mmads=[dict(m=m, n=n, k=k, count=count)
                           for (m, n, k), count in sorted(self.calls.items())],
                    output_flushes=self.output_flushes,
                    logical_a_read_bytes=self.logical_a_read_bytes,
                    logical_b_read_bytes=self.logical_b_read_bytes,
                    logical_output_write_bytes=self.logical_output_write_bytes)


def external_work(measurement: dict) -> dict:
    """Derive fixed-wrapper per-core work; reject unreviewed configurations."""
    r = measurement
    m, n, k = (r[key] for key in ('m', 'n', 'k'))
    if any(type(v) is not int or v <= 0 for v in (m, n, k)):
        raise ValueError('positive integral dimensions required')
    if (r['dtype'], r['output_dtype'], r['layout_a'], r['layout_b'],
            r['trans_a'], r['trans_b']) != ('float16', 'float16', 'row_major', 'row_major', False, False):
        raise ValueError('unreviewed numerical/layout contract')
    cores = {}
    def core(index):
        return cores.setdefault(index, CoreWork(index))
    caveats = ['MMAD shape is source-derived; lowering and hardware rounding require independent audit',
               'logical requests are not PMU transactions or physical HBM bytes',
               'Cube active service is not whole-kernel elapsed time']
    extra = {}
    if r['backend'] == 'catlass':
        if r['algorithm'] != 'example09_splitk':
            raise ValueError('unsupported CATLASS kernel')
        l1, l0 = r['l1_tile'], r['l0_tile']
        if (l1, l0) not in [([128,256,256],[128,256,64]), ([16,16,4096],[16,16,1024])]:
            raise ValueError('unreviewed CATLASS tile')
        gm, gn, gk = (ceildiv(v, b) for v, b in zip((m,n,k), l1))
        split, launched = r['splitk_factor'], r['physical_aic']
        if not 1 <= split <= gk or launched <= 0:
            raise ValueError('invalid split/launch geometry')
        # Exact SplitkGemmIdentityBlockSwizzle<3,0>: K slice then Zn M/N order.
        for s in range(split):
            begin = s * (gk // split) + min(s, gk % split)
            kt = gk // split + int(s < gk % split)
            chunk_k = min(kt * l1[2], k - begin * l1[2])
            for inner in range(gm * gn):
                block = inner // (3 * gn)
                rem = inner % (3 * gn)
                height = min(3, gm - block * 3)
                mi = block * 3 + rem % height
                ni = rem // height
                if block % 2:
                    ni = gn - 1 - ni
                tm, tn = min(l1[0], m-mi*l1[0]), min(l1[1], n-ni*l1[1])
                target = core((s * gm * gn + inner) % launched)
                # K chunk ends on L1 boundaries except final tail, and L0 divides L1.
                target.add_chunk(tm, tn, chunk_k, l0, 4, pad_mn=True)
        extra = dict(split_k_slices=split, partial_write_bytes=4*m*n*split,
                     reduction_read_bytes=4*m*n*split,
                     reduction_add_elements=(split-1)*m*n, output_cast_elements=m*n)
        source = 'catlass:SplitkGemmIdentityBlockSwizzle<3,0>/MmadAtlasA2Pingpong'
    elif r['backend'] == 'ops_nn':
        if r.get('tiling_policy') != 'manual_conservative_v1' or r['base_tile'] != [128,128,128]:
            raise ValueError('only frozen direct harness supported')
        if m > 128 or n > 128 or k % 16:
            raise ValueError('adapter scope is small-M/N, aligned-K benchmark domain')
        strategy, launched = r['actual_strategy'], r['block_dim']
        chunk = r['single_core'][2]
        chunks = ceildiv(k, chunk)
        if strategy == 'DETERMINISTIC_SPLIT_K':
            if chunk != 384 or r['steps'] != [3,1,3,3]:
                raise ValueError('only nondegenerate MK33 deterministic path reviewed')
            q, rem = divmod(chunks, launched)
            for ci in range(launched):
                start = ci*q + min(ci, rem)
                count = q + int(ci < rem)
                for index in range(start, start + count):
                    core(ci).add_chunk(m, n, min(chunk, k-index*chunk), r['base_tile'], 4)
            extra = dict(k_chunks=chunks, partial_write_bytes=4*m*n*chunks,
                         reduction_read_bytes=4*m*n*launched,
                         reduction_add_elements=(launched-1)*m*n, output_cast_elements=m*n,
                         atomic_update_chunks=max(0,chunks-launched))
        elif strategy in ('BASE', 'SINGLE_CORE_SPLIT_K'):
            if launched != 1:
                raise ValueError('unexpected small-M/N core geometry')
            if (strategy == 'BASE' and chunk != k) or (strategy == 'SINGLE_CORE_SPLIT_K' and chunk != min(k,512)):
                raise ValueError('unexpected K chunk')
            for index in range(chunks):
                core(0).add_chunk(m,n,min(chunk,k-index*chunk),r['base_tile'],
                                  2 if strategy == 'BASE' else 4)
            extra = dict(k_chunks=chunks, atomic_update_chunks=max(0,chunks-1),
                         output_cast_elements=0 if strategy == 'BASE' else m*n)
        else:
            raise ValueError('unsupported ops strategy')
        extra['nd2nz_a_logical_read_bytes'] = 2*m*k if r['nd2nz_a'] else 0
        extra['nd2nz_b_logical_read_bytes'] = 2*k*n if r['nd2nz_b'] else 0
        source = 'ops-nn:frozen BASE/PRELOAD_MK/MK33 host and kernel source'
        caveats.append('CANN inner Matmul lowering assumed baseK calls; verify instruction/work counters')
        if m == 1:
            caveats.append('M=1 may trigger GEMV lowering; 16-rounded Cube block cost unqualified')
    else:
        raise ValueError('unknown backend')
    return dict(schema_version=1, source_model=source, source_derived=True,
                counter_validated=False, m=m,n=n,k=k, nominal_flops=2*m*n*k,
                active_cube_cores=len(cores), cores=[v.to_dict() for _,v in sorted(cores.items())],
                total_cube_calls=sum(v.cube_calls for v in cores.values()),
                total_cube_blocks=sum(v.cube_blocks for v in cores.values()),
                extra_work=extra, caveats=caveats)


def legacy_cube_prediction(work: dict, fit: CubePipelineFit, frequency_mhz: float) -> dict:
    if not math.isfinite(frequency_mhz) or frequency_mhz <= 0:
        raise ValueError('positive frequency required')
    rows = []
    for c in work['cores']:
        calls, blocks = c['cube_calls'], c['cube_blocks']
        cycles = calls*fit.latency_cycles + (blocks-calls)*fit.issue_interval_cycles
        rows.append(dict(core_id=c['core_id'], cube_calls=calls, cube_blocks=blocks,
                         predicted_cube_active_cycles=cycles, predicted_cube_active_us=cycles/frequency_mhz))
    return dict(per_core=rows, sum_active_us=sum(v['predicted_cube_active_us'] for v in rows),
                max_core_active_us=max(v['predicted_cube_active_us'] for v in rows),
                whole_kernel_latency_us=None,
                qualification='external extrapolation of legacy per-call Cube service formula',
                frequency_mhz=frequency_mhz)
