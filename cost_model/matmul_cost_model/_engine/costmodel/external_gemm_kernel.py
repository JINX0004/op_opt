"""Implementation-conditioned *whole kernel* model for the external wrappers.

The source adapters own work counts. This module owns a finite-buffer schedule
and conditional service parameters. Parameters fitted to kernel events are
effective costs, NOT independently identified hardware service rates. In
particular, a predicted critical resource is a model attribution, not PMU proof.
All times are microseconds. No measured duration is consumed by predict().
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import json
from typing import Mapping

from .external_gemm_work import ceildiv


@dataclass
class Clock:
    next_issue: float = 0.0
    completion: float = 0.0
    service: float = 0.0
    count: int = 0

    def issue(self, ready: float, latency: float, interval: float | None = None):
        if interval is None:
            interval = latency
        if not all(math.isfinite(x) and x >= 0 for x in (ready, latency, interval)):
            raise ValueError('nonnegative finite times required')
        if interval > latency + 1e-12:
            raise ValueError('issue interval cannot exceed completion latency')
        start = max(ready, self.next_issue)
        end = start + latency
        self.next_issue = start + interval
        self.completion = max(self.completion, end)
        self.service += interval
        self.count += 1
        return end


@dataclass(frozen=True)
class KernelParameters:
    """Positive context coefficients; defaults are structural priors only."""
    entry_us: float = 5.0
    mte2_scale: float = 1.0
    scalar_scale: float = 1.0
    fixpipe_scale: float = 1.0
    conversion_scale: float = 1.0
    reduction_scale: float = 1.0
    shared_scale: float = 1.0
    tail_scale: float = 1.0
    vnchw_scale: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError('strictly positive finite model parameters required')


def cube_service(m: int, n: int, k: int, *, narrow_deep: bool = False) -> float:
    """Retain legacy Cube and the independently measured narrow-deep repair."""
    blocks = ceildiv(m,16)*ceildiv(n,16)*ceildiv(k,16)
    if narrow_deep and m == n == 16 and k == 1024:
        return (44.32853606050941 + (blocks-1)*2.0379445810361325)/1800.0
    return (44.225302439513385 + (blocks-1)*0.9643561962910648)/1800.0


def mte1_service(a_calls, b_calls, a_blocks, b_blocks):
    # Existing Mte1CallStructureFit v1, without recharging its fixed boundary
    # at every operation. Toolchain transfer remains a model assumption.
    return max(0.0, (3.344420062904378*(a_calls+b_calls)
                    +1.653977503506531*(a_blocks+b_blocks)
                    -2.7908475802762203*b_calls
                    +2.5581665447251276*b_blocks)/1800.0)


def fixpipe_service(m, n, *, atomic=False):
    # FixpipeIntrinsicFit v3/v4 shape-aware fallback, FP16/atomic differences
    # are conditional scale terms; not claimed as a new primitive calibration.
    mb, nb = ceildiv(m,16), ceildiv(n,16)
    return (137.66483372997993 + 45.7326969902116*(mb*nb-1)
            + 24.94214557963165*(nb-1))/1800.0 * (1.25 if atomic else 1.0)


def transfer_service(t, p):
    """A logical request model, not logical-bytes / physical HBM bandwidth.

    Row count and intrinsic issue count are separate. The tail factor covers
    narrow/unaligned source rows whose transaction and ND2NZ costs differ.
    """
    calls = t.get('calls', 1)
    rows = t.get('rows', 1)
    logical = t['bytes']
    row_bytes = t.get('row_bytes', logical/max(rows,1))
    base = 0.055*calls + logical/70000.0 + rows*0.0007
    tail = (rows*0.0025 + logical/45000.0) if row_bytes % 32 else 0.0
    return p.mte2_scale*(base+p.tail_scale*tail)


def simulate_core(tasks, p: KernelParameters, *, l1_depth=2, l0_depth=2,
                  l0c_depth=2, narrow_deep=False, trace=False,frontend_mode='queued'):
    """Finite-buffer source program, using completion for unknown consume times.

    A task owns one L0C output; each slab has GM->L1 transfers and ordered
    L0 chunks. The source emitters split larger output regions into such tasks.
    Scalar issues overlap earlier DMA/compute. L1/L0 credits prevent overwrite.
    No sum of component-active maxima is used as a kernel prediction.
    """
    if min(l1_depth,l0_depth,l0c_depth) <= 0:
        raise ValueError('buffer depths must be positive')
    if frontend_mode not in ('queued','blocking'):
        raise ValueError('unknown frontend assumption')
    clocks = {s:Clock() for s in ('scalar','mte2','mte1','cube','fixpipe')}
    l1_free = [0.0]*l1_depth
    l0_free = [0.0]*l0_depth
    l0c_free = [0.0]*l0c_depth
    slab_index = chunk_index = 0
    events = []
    total_requests = 0
    retained_operands = {}
    call_end_ready = 0.0
    for ti, task in enumerate(tasks):
        acc_ready = max(l0c_free[ti % l0c_depth],call_end_ready)
        task_control = clocks['scalar'].issue(0, p.scalar_scale*0.24)
        slabs = task['slabs']
        slab_start = slab_index
        loaded_slabs = {}

        def load_slab(index):
            nonlocal total_requests
            slab = slabs[index]
            si = (slab_start+index) % l1_depth
            ready = max(l1_free[si],task_control,call_end_ready)
            finished = ready
            operand_ends = []
            for t in slab['transfers']:
                cache_key = t.get('cache_key')
                if t.get('cache_hit',False):
                    if cache_key not in retained_operands:
                        raise ValueError('cached operand must have an earlier producer')
                    operand_ends.append(max(ready,retained_operands[cache_key]))
                    continue
                issue = clocks['scalar'].issue(ready if frontend_mode=='blocking' else 0,
                                               p.scalar_scale*0.012*t.get('calls',1))
                finished = clocks['mte2'].issue(max(issue,ready), transfer_service(t,p))
                operand_ends.append(finished)
                total_requests += t['bytes']
                if cache_key is not None:
                    retained_operands[cache_key] = finished
            loaded_slabs[index] = operand_ends

        for index in range(min(l1_depth,len(slabs))):
            load_slab(index)
        for index,slab in enumerate(slabs):
            si = slab_index % l1_depth
            loaded_a,loaded_b = loaded_slabs[index]
            loaded = max(loaded_a,loaded_b)
            for c in slab['chunks']:
                ci = chunk_index % l0_depth
                m,n,k = c['m'],c['n'],c['k']
                mb,nb,kb = ceildiv(m,16),ceildiv(n,16),ceildiv(k,16)
                ac,bc = c.get('a_calls',mb),c.get('b_calls',kb)
                # API counts are source counts, not scalar ISA instructions.
                issue = clocks['scalar'].issue(max(loaded,l0_free[ci]) if frontend_mode=='blocking' else 0,
                          p.scalar_scale*(0.022 + 0.001*(ac+bc)))
                moved_a = clocks['mte1'].issue(max(issue,loaded_a,l0_free[ci]),mte1_service(ac,0,mb*kb,0))
                moved = clocks['mte1'].issue(max(issue,loaded_b,l0_free[ci]),mte1_service(0,bc,0,nb*kb))
                l1_free[si] = moved
                issue = clocks['scalar'].issue(moved if frontend_mode=='blocking' else 0,p.scalar_scale*0.018)
                acc_ready = clocks['cube'].issue(max(issue,moved,acc_ready),
                                      cube_service(m,n,k,narrow_deep=narrow_deep))
                l0_free[ci] = acc_ready
                if c.get('barrier',False):
                    clocks['scalar'].issue(acc_ready if frontend_mode=='blocking' else 0,p.scalar_scale*0.015)
                if trace:
                    events.append(dict(task=ti,slab=slab_index,chunk=chunk_index,
                                       mte2_complete=loaded,mte1_complete=moved,
                                       mte2_a_complete=loaded_a,mte2_b_complete=loaded_b,
                                       mte1_a_complete=moved_a,mte1_b_complete=moved,
                                       cube_complete=acc_ready))
                chunk_index += 1
            slab_index += 1
            if index+l1_depth < len(slabs):
                load_slab(index+l1_depth)
        issued = clocks['scalar'].issue(acc_ready if frontend_mode=='blocking' else 0,p.scalar_scale*0.035)
        written = clocks['fixpipe'].issue(max(issued,acc_ready),p.fixpipe_scale*fixpipe_service(
                         task['m'],task['n'],atomic=task.get('atomic',False)))
        l0c_free[ti % l0c_depth] = written
        if task.get('call_end',False) or task.get('pipeline_barrier_after',False):
            # SDK End destroys operand/output queues. FreeAllEvent waits for
            # pending release events (MTE1->MTE2 and FIX->M); this is a device
            # dependency for the next call, not a global Scalar issue stall.
            call_end_ready = max(written,clocks['mte1'].completion)
            retained_operands.clear()
    return dict(body_us=max((c.completion for c in clocks.values()),default=0),
                resources={k:dict(service_us=c.service,last_complete_us=c.completion,
                                  issue_count=c.count) for k,c in clocks.items()},
                logical_request_bytes=total_requests,events=events)


def vector_phase(read_bytes,write_bytes,elements,tiles,cores,p,*,conversion,buffer_depth=2,
                 api_counts=None,path='generic'):
    """Batched AIV phase with MTE2/Vector/MTE3 overlap and fill/drain.

    Phase barriers are composed separately. Bytes are logical source requests.
    """
    if not (read_bytes or write_bytes or elements or tiles or (api_counts and any(api_counts.values()))):
        return dict(time_us=0.0,resources={})
    cores = max(1,cores)
    waves = max(1,ceildiv(tiles,cores))
    scale = p.conversion_scale if conversion else p.reduction_scale
    if conversion and path.startswith('vnchw'):
        scale *= p.vnchw_scale
    read = read_bytes/cores/35000.0
    write = write_bytes/cores/30000.0
    vec = elements/cores/(4500.0 if conversion else 10000.0)
    scalar = waves*(0.18 if conversion else 0.06)
    if conversion and api_counts is not None:
        a=api_counts
        # Source-counted conditional priors, not measured hardware peaks.
        read = read_bytes/cores/100000.0 + (.03*a.get('mte2_calls',0)+.002*a.get('mte2_bursts',0)
              +.012*p.tail_scale*a.get('unaligned_mte2_row_units',0)
              +.001*a.get('strided_mte2_bursts',0))/cores
        write = write_bytes/cores/100000.0 + (.04*a.get('mte3_calls',0)+.002*a.get('mte3_bursts',0)
                 +.001*a.get('strided_mte3_bursts',0))/cores
        vec = (.012*a.get('vector_copy_repeats',0)+.008*a.get('transdata_repeats',0)
               + a.get('duplicate_elements',0)/45000.0
               + .04*a.get('nonzero_vector_api_calls',sum(a.get(k,0) for k in
                                                  ('vector_copy_calls','transdata_calls','duplicate_calls'))))/cores
        scalar = (.003*a.get('scalar_address_assignments',0)+.02*a.get('scalar_loop_iterations',0)
                  +.02*sum(a.get(k,0) for k in ('set_flag_calls','wait_flag_calls','pipe_barrier_calls')))/cores
    service = dict(mte2=read,vector=vec,mte3=write,scalar=scalar)
    # For one tile all stages are on the critical path; later tiles pipeline.
    fill = (sum(service.values())-max(service.values()))/waves
    if buffer_depth == 1:
        fill = sum(service.values())-max(service.values())
    return dict(time_us=scale*(0.5+max(service.values())+fill),
                resources={k:v*scale for k,v in service.items()},
                active_cores=cores,tile_waves=waves)


def _vector_completion(program,p,phase):
    conversion = phase == 'conversion'
    if phase+'_cores' not in program:
        return vector_phase(**program[phase],p=p,conversion=conversion)
    rows=[]
    for row in program[phase+'_cores']:
        if 'segments' in row:
            segments=[vector_phase(**work,p=p,conversion=conversion) for work in row['segments']]
            rows.append(dict(core_id=row['core_id'],time_us=sum(s['time_us'] for s in segments),segments=segments))
        else:
            rows.append(dict(core_id=row['core_id'],**vector_phase(**row['work'],p=p,conversion=conversion)))
    return dict(time_us=max((row['time_us'] for row in rows),default=0.0),per_core=rows,
                active_cores=sum(row['time_us']>0 for row in rows))


def simulate_program(program, p: KernelParameters, *, trace=False):
    cached = {}
    cores = []
    for index,c in enumerate(program['cores']):
        key = program.get('_core_keys',list(range(len(program['cores']))))[index]
        if key not in cached:
            cached[key] = simulate_core(c['tasks'],p,
                  narrow_deep=program.get('narrow_deep',False),trace=trace,**program.get('buffers',{}))
        cores.append(dict(core_id=c['core_id'],**cached[key]))
    pre = _vector_completion(program,p,'conversion')
    post = _vector_completion(program,p,'reduction')
    core_max = max((c['body_us'] for c in cores),default=0.0)
    total_bytes = sum(c['logical_request_bytes'] for c in cores)
    # Shared request capacity for the warmed-cache protocol, NOT a HBM roof.
    shared = total_bytes/800000.0*p.shared_scale
    compute = max(core_max,shared)
    critical = max(cores,key=lambda x:x['body_us']) if cores else None
    return dict(kernel_time_us=p.entry_us+pre['time_us']+compute+post['time_us'],
                entry_us=p.entry_us,conversion=pre,matmul_us=compute,
                shared_request_bound_us=shared,critical_core_id=None if critical is None else critical['core_id'],
                reduction=post,per_core=cores,
                measurement_boundary='warmed batched ACL-event mean per launch; no host allocation/H2D/tiling',
                composition='entry + conversion barrier + max(per-core finite-buffer completion, shared request capacity) + reduction barrier',
                qualification='conditional whole-kernel estimate; service priors and phase overlap require external validation')


def _empty_vector():
    return dict(read_bytes=0,write_bytes=0,elements=0,tiles=0,cores=1)


def catlass_program(configuration):
    from .catlass_kernel_features import catlass_kernel_features
    features = catlass_kernel_features(configuration)
    cores = []
    for core in features['aic_cores']:
        tasks = []
        for task in core['tasks']:
            slabs = []
            for step in task['l1_steps']:
                transfers = [dict(bytes=t['logical_read_bytes'],calls=t['intrinsic_calls'],
                                  rows=t['rows'],row_bytes=t['row_bytes'],
                                  stride_bytes=t['source_row_stride_bytes'])
                             for t in (step['a_mte2'],step['b_mte2'])]
                chunks = [dict(m=c['m'],n=c['n'],k=c['k'],
                               a_calls=c['a_mte1']['intrinsic_calls'],
                               b_calls=c['b_mte1']['intrinsic_calls'],
                               barrier=bool(c['cube']['pipe_m_barriers']))
                          for c in step['l0_chunks']]
                slabs.append(dict(transfers=transfers,chunks=chunks))
            tasks.append(dict(m=task['m'],n=task['n'],slabs=slabs,atomic=False))
        cores.append(dict(core_id=core['core_id'],tasks=tasks))
    red = features['reduction']
    reduction = dict(read_bytes=red['reduction_read_bytes'],write_bytes=red['output_write_bytes'],
                     elements=red['reduction_add_elements']+red['output_cast_elements'],
                     tiles=red['task_count'],cores=max(1,red['active_aiv']))
    reduction_cores = []
    for core in features['aiv_cores']:
        totals = core['totals']
        reduction_cores.append(dict(core_id=core['core_id'],work=dict(
            read_bytes=totals.get('reduction_read_bytes',0),write_bytes=totals.get('output_write_bytes',0),
            elements=totals.get('reduction_add_elements',0)+totals.get('output_cast_elements',0),
            tiles=totals.get('task_count',0),cores=1,buffer_depth=1)))
    return dict(configuration=features['configuration'],cores=cores,
                conversion=_empty_vector(),reduction=reduction,
                reduction_cores=reduction_cores,
                buffers=dict(l1_depth=features['buffers']['l1_stages'],
                             l0_depth=min(features['buffers']['l0a_stages'],features['buffers']['l0b_stages']),
                             l0c_depth=features['buffers']['l0c_stages']),
                narrow_deep=configuration['l0_tile']==[16,16,1024],
                source_features=features)


def ops_program(configuration):
    from .ops_kernel_features import ops_kernel_features
    features = ops_kernel_features(configuration)
    phases = {row['phase_id']:row for row in features['phases']}
    cores = []
    for core in phases['matmul']['per_core']:
        tasks = []
        for chunk in core['chunks']:
            outputs = {}
            for step in chunk['l1_tiles']:
                key = (step['m_offset'],step['n_offset'])
                task = outputs.setdefault(key,dict(m=step['m'],n=step['n'],slabs=[],atomic=chunk['atomic']))
                transfers = []
                for operand,rows,columns in (('a',step['m'],step['k']),('b',step['k'],step['n'])):
                    nz = configuration['nd2nz_'+operand]
                    transfer = dict(bytes=step['mte2_'+operand+'_bytes'],calls=1,
                        rows=rows,row_bytes=(32 if nz else columns*2),
                        stride_bytes=(32 if nz else 2*configuration['k' if operand=='a' else 'n']))
                    if operand=='a' and 'mte2_a_cache_key' in step:
                        transfer['cache_key']=step['mte2_a_cache_key']
                        transfer['cache_hit']=step['mte2_a_cache_hit']
                    transfers.append(transfer)
                cubes = []
                for c in step['cube_tiles']:
                    for _ in range(c['count']):
                        cubes.append(dict(m=c['m'],n=c['n'],k=c['k'],barrier=False,
                            a_calls=c.get('mte1_a_api_calls',ceildiv(c['m'],16)),
                            b_calls=c.get('mte1_b_api_calls',ceildiv(c['k'],16))))
                task['slabs'].append(dict(transfers=transfers,chunks=cubes))
            chunk_tasks=list(outputs.values())
            if chunk_tasks:
                chunk_tasks[-1]['call_end']=chunk.get('call_end',False)
                chunk_tasks[-1]['pipeline_barrier_after']=bool(chunk.get('output_tile_barrier'))
            tasks.extend(chunk_tasks)
        cores.append(dict(core_id=core['core_id'],tasks=tasks))
    result = dict(configuration=features['configuration'],cores=cores,source_features=features,
                  buffers=dict(l1_depth=2,l0_depth=2,
                    l0c_depth=phases['matmul']['l0c_buffer_depth']),narrow_deep=False)
    for source,target in [('nd2nz','conversion'),('reduce_cast','reduction')]:
        result[target] = _empty_vector()
        rows = []
        for core in phases[source]['per_core']:
            if source=='nd2nz':
                segments=[]
                for group in core['conversion_groups']:
                    f=group['features']
                    api=dict(f)
                    api['unaligned_mte2_row_units']=0.0
                    api['strided_mte2_bursts']=api['strided_mte3_bursts']=0
                    for operation in group['operations']:
                        for d in operation['api_work']['mte2']:
                            step=d['burst_bytes']+d.get('source_stride_bytes',0)
                            # Source row alignment pressure, not a cache-line or
                            # hardware transaction count. Calibrated context term.
                            api['unaligned_mte2_row_units']+=d['bursts']*(1-math.gcd(step,32)/32)
                            if d.get('source_stride_bytes',0): api['strided_mte2_bursts']+=d['bursts']
                        for d in operation['api_work']['mte3']:
                            if d.get('destination_stride_bytes',0): api['strided_mte3_bursts']+=d['bursts']
                    segments.append(dict(read_bytes=f['mte2_bytes'],write_bytes=f['mte3_bytes'],
                         elements=f['nd2nz_elements'],tiles=f['control_iterations'],cores=1,
                         buffer_depth=group['buffer_depth'],api_counts=api,path=group['path']))
                rows.append(dict(core_id=core['core_id'],segments=segments))
                continue
            f = core['features']
            rows.append(dict(core_id=core['core_id'],work=dict(read_bytes=f['mte2_bytes'],
                        write_bytes=f['mte3_bytes'],
                        elements=f['nd2nz_elements']+f['cast_elements']+f['vector_add_elements'],
                        tiles=f['control_iterations'],cores=1,buffer_depth=2)))
        result[target+'_cores'] = rows
    result['phase_caveat'] = ('Conservative all-core phase barrier; SCSK aligned fast-cast may overlap '
                             'individual output tiles with later compute. CANN inner queue/II unresolved.')
    return result


def compile_program(configuration: Mapping):
    """Consume source configuration only; never infer a route from FLOPs."""
    if configuration['backend'] == 'catlass':
        program = catlass_program(configuration)
    elif configuration['backend'] == 'ops_nn':
        program = ops_program(configuration)
    else:
        raise ValueError('unsupported implementation')
    signatures = {}
    keys = []
    for core in program['cores']:
        key = json.dumps(core['tasks'],sort_keys=True,separators=(',',':'))
        if key not in signatures:
            signatures[key] = len(signatures)
        keys.append(signatures[key])
    program['_core_keys'] = keys
    return program


def manual_configuration(m,n,k,config_id,physical_aic=24):
    if config_id.startswith('catlass_'):
        from .catlass_kernel_features import manual_catlass_configuration
        return manual_catlass_configuration(m,n,k,config_id.removeprefix('catlass_'),physical_aic)
    from .ops_kernel_features import manual_ops_configuration
    variant = dict(ops_base='base',ops_scsk='scsk',ops_deterministic='deterministic').get(config_id)
    if variant is None:
        raise ValueError('unknown implementation/configuration')
    return manual_ops_configuration(m,n,k,variant,physical_aic)
