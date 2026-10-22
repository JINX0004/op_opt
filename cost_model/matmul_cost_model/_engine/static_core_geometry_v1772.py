"""Work distribution and initial direct-ND source-window phase proxies.

Address dispersion is an arithmetic input feature, not a claim about real
bank topology, page size, concurrency or measured hardware conflicts.
"""
from math import ceil,sqrt,log2
from matmul_cost_model._engine.costmodel.matmul_four_path_v590 import compile_plan
EXTRA_NAMES=('core_work_min_max_ratio','core_work_CV','core_work_uniformity',
    'A_last_source_min_fraction','A_last_source_weighted_fraction',
    'B_last_source_min_fraction','B_last_source_weighted_fraction',
    'A_first_window_reuse_fraction','B_first_window_reuse_fraction',
    'A_direct_ND_initial_phase512_excess','B_direct_ND_initial_phase512_excess',
    'A_direct_ND_initial_phase4096_excess','B_direct_ND_initial_phase4096_excess')

def extra_features(row):
    p=compile_plan(row);work=[js for js in p['work'] if js];m,n,k=row['shape'];core=[];allg=[]
    for jobs in work:
        gs=[j['geometry'] for j in jobs];allg+=gs
        core.append(sum(ceil(g['m']/16)*ceil(g['n']/16)*ceil(g['k']/16) for g in gs))
    mean=sum(core)/len(core);maximum=max(core);cv=sqrt(sum((w-mean)**2 for w in core)/len(core))/mean
    raw=[min(core)/maximum,cv,(sum(core)**2)/(len(core)*sum(w*w for w in core))]
    for op,axis in (('a','m'),('b','n')):
        values=[((g['k']-1)%(g['bk']*g['s'+op])+1)/(g['bk']*g['s'+op]) for g in allg]
        weights=[g[axis]*g['k'] for g in allg]
        raw.extend((min(values),sum(v*w for v,w in zip(values,weights))/sum(weights)))
    reuse={op:0. for op in ('a','b')};phase={(op,q):0. for op in ('a','b') for q in (512,4096)}
    for wave in range(max(map(len,work))):
        gs=[jobs[wave]['geometry'] for jobs in work if len(jobs)>wave]
        for op in ('a','b'):
            windows={};direct={}
            for g in gs:
                span=min(g['k'],g['bk']*g['s'+op])
                if op=='a':
                    key=(g['mo'],g['ko'],g['m'],span);address=512+row.get('source_a_offset_extra_bytes',0)+2*(g['mo']*k+g['ko'])
                else:key=(g['ko'],g['no'],span,g['n']);address=512+2*(g['ko']*n+g['no'])
                windows[key]=address
                if not g['nz'+op]:direct[key]=address
            reuse[op]=max(reuse[op],1-len(windows)/len(gs))
            unique=len(direct)
            if not unique:continue
            for q in (512,4096):
                counts={}
                for address in direct.values():
                    sector=(address%q)//128;counts[sector]=counts.get(sector,0)+1
                excess=max(0.,sum(v*v for v in counts.values())/(unique*unique)-1/unique)
                phase[op,q]=max(phase[op,q],excess)
    raw.extend((reuse['a'],reuse['b'],phase['a',512],phase['b',512],phase['a',4096],phase['b',4096]))
    assert len(raw)==len(EXTRA_NAMES) and all(v>=-1e-12 for v in raw)
    return [log2(1+max(0,v)) for v in raw]
