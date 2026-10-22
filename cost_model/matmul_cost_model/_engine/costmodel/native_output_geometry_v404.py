"""Nominal-wave output interval sharing,computed from source task assignment.

128/512B intervals are static address geometry,not asserted transactions/locks.
Nominal job index is not a measured simultaneous arrival schedule.
"""
from collections import Counter,defaultdict
from .native_output_geometry_v392 import attach_output_geometry as original_attach

def attach_output_geometry(row,program):
    program=original_attach(row,program);kind=program['implementation'];m,n,k=program['shape'];cohorts=defaultdict(list)
    if kind=='catlass':
        ks=row['l1_tile'][2];gk=(k+ks-1)//ks;split=program['split_factor']
        starts={(i*(gk//split)+min(i,gk%split))*ks:i for i in range(split)}
    for ci,jobs in enumerate(program['work']):
        for ji,job in enumerate(jobs):
            g=job['geometry'];ldc=g['output_ldc'];eb=g['output_element_bytes']
            if kind=='catlass':offset=starts[g['ko']]*m*n+g['mo']*n+g['no']
            elif kind=='deterministic':offset=2*ci*row['plan']['tcube']['singleCoreM']*n
            else:offset=g['mo']*ldc+g['no']
            assert offset*eb%512==g['output_phase_bytes']
            cohorts[ji].append((ci,g,offset))
    cache={}
    for jobs in cohorts.values():
        signature=tuple((ci,offset,g['m'],g['n'],g['output_ldc'],g['output_element_bytes']) for ci,g,offset in jobs)
        if signature not in cache:
            counts=[]
            for grain in (128,512):
                unions=[]
                for ci,g,offset in jobs:
                    union=set();eb=g['output_element_bytes']
                    for r in range(g['m']):
                        begin=(offset+r*g['output_ldc'])*eb;end=begin+g['n']*eb
                        union.update(range(begin//grain,(end-1)//grain+1))
                    unions.append(union)
                peers=Counter(line for union in unions for line in union)
                counts.append([sum(peers[line]-1 for line in union)/1000 for union in unions])
            cache[signature]=counts
        counts=cache[signature]
        for i,(ci,g,offset) in enumerate(jobs):g['output_peer128_k']=counts[0][i];g['output_peer512_k']=counts[1][i]
    program['native_output_geometry']['shared_interval_hypothesis']='Same nominal per-core job index;not measured arrival overlap.'
    return program
