"""Nominal B row-interval requester ownership, not a hardware bank model.

The128B intervals are static geometry proxies. Distinct clients are separate
from within-client row revisits. Repeated job-wave patterns are cached within
one plan; no timing or shape-performance entries are cached.
"""
from collections import Counter,defaultdict
from .first_source_pressure_v489 import attach_output_geometry as original_attach

def attach_output_geometry(row,program):
    program=original_attach(row,program)
    for core in program['work']:
        for job in core:job['geometry']['interval_b_peer_excess']=0.
    if program['active_cores']<=1:return program
    waves=defaultdict(list)
    for ci,core in enumerate(program['work']):
        for ji,job in enumerate(core):
            g=job['geometry']
            if not g['nzb']:waves[ji].append((ci,g))
    cache={}
    for jobs in waves.values():
        if len(jobs)<=1:continue
        anchor=min(2*(g['ko']*g['global_n']+g['no']) for ci,g in jobs)//128*128
        specs=tuple((ci,2*(g['ko']*g['global_n']+g['no'])-anchor,min(g['k'],g['bk']*g['sb']),2*g['n'],2*g['global_n']) for ci,g in jobs)
        if specs not in cache:
            clients={};distinct=Counter()
            for ci,start,rows,width,stride in specs:
                visits=Counter()
                for r in range(rows):
                    first=start+r*stride;visits.update(range(first//128,(first+width-1)//128+1))
                clients[ci]=visits;distinct.update(visits.keys())
            cache[specs]={ci:sum(count*(distinct[line]-1) for line,count in v.items())/sum(v.values()) for ci,v in clients.items()}
        for ci,g in jobs:
            # New feature only counts sharing missed by exact starting-address
            # equality. Own repeated row visits are already represented elsewhere.
            g['interval_b_peer_excess']=max(0.,cache[specs][ci]-max(0,g['same_address_b_clients']-1))
    return program
