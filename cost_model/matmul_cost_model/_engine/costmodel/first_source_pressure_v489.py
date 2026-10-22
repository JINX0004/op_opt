"""Static first B demand with C0-parity packed-row coalescing proxy;unqualified service.

Even-C0 contiguous full-width ND rows contribute one visit per touched interval per
client. Odd-C0 and strided subrows retain their separate row visits. This describes an
effective request hypothesis,not a verified cache transaction/arbiter.
"""
from collections import Counter
from .native_output_geometry_v404 import attach_output_geometry as original_attach
from .source_line_demand_v439 import _visits


def attach_output_geometry(row,program):
    program=original_attach(row,program);clients={}
    for ci,jobs in enumerate(program['work']):
        if not jobs:continue
        g=jobs[0]['geometry'];raw=_visits(g,'b',128)
        if raw is not None:clients[ci]=Counter({line:1 for line in raw}) if g['n']==g['global_n'] and (g['n']//16)%2==0 else raw
    total=Counter()
    for visits in clients.values():total.update(visits)
    for ci,jobs in enumerate(program['work']):
        if not jobs:continue
        v=clients.get(ci)
        pressure=sum(count*(total[line]-count) for line,count in v.items())/sum(v.values()) if v else max(program['active_cores']-1,0)
        jobs[0]['geometry']['first_b_peer_visits']=pressure
    return program
