"""Output pitch and byte phases from the reviewed four-path source semantics.

Current FP16-input NN profile:host/workspace output bases are512B aligned.
Only attach static geometry;no timing,PMU or labels are read.
"""
def attach_output_geometry(row,program):
    kind=program['implementation'];m,n,k=program['shape']
    if m%16 or (n%16 and n!=24):raise ValueError('new FP32 output model reviewed for M%16=0,N%16=0 orN24')
    starts={}
    if kind=='catlass':
        ks=row['l1_tile'][2];gk=(k+ks-1)//ks;split=program['split_factor']
        starts={(i*(gk//split)+min(i,gk%split))*ks:i for i in range(split)}
        limit=m*n*split;ldc=n;element_bytes=4
    elif kind=='scsk':
        ldc=((n+63)//64)*64;element_bytes=4;limit=m*ldc
    elif kind=='deterministic':
        t=row['plan']['tcube']
        if t['singleCoreM']<m or t['singleCoreN']<n:raise ValueError('deterministic output model requires reviewed one-MN-tile policy')
        ldc=n;element_bytes=4;single_size=t['singleCoreM']*n
        limit=2*single_size*program['cores']
    elif kind=='base':ldc=n;element_bytes=2;limit=m*n
    else:raise ValueError('unknown implementation')
    summary=dict(element_bytes=element_bytes,ldc=ldc,base_alignment_bytes=512,
        output_allocation_elements=limit,unit_flag_enabled=kind in ('base','catlass'),
        per_core_phases=[],source_semantics='FP16 NN reviewed direct wrappers/pinned source')
    for ci,jobs in enumerate(program['work']):
        phases=[]
        for job in jobs:
            g=job['geometry'];mo,no=g['mo'],g['no']
            if kind=='catlass':offset=starts[g['ko']]*m*n+mo*n+no
            elif kind=='deterministic':
                assert mo==no==0 and g['m']==m and g['n']==n
                offset=2*ci*single_size
            else:offset=mo*ldc+no
            assert 0<=offset and offset+(g['m']-1)*ldc+g['n']<=limit
            g['output_ldc']=ldc;g['output_phase_bytes']=offset*element_bytes%512
            g['output_element_bytes']=element_bytes;phases.append(g['output_phase_bytes'])
        summary['per_core_phases'].append(sorted(set(phases)))
    program['native_output_geometry']=summary
    return program
