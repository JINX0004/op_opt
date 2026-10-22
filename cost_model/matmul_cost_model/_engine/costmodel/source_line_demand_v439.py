"""Static ND row-interval demand by nominal job wave, not hardware transactions.

Repeated row visits and distinct clients are deliberately different. No timing,
PMU, target label or physical-address assumption enters this geometry audit.
NZ sources are explicitly omitted. This is not wired into a timing predictor.
"""
from collections import Counter,defaultdict


def _visits(g,operand,grain):
    if g['nza' if operand=='a' else 'nzb']:return None
    extent=min(g['k'],g['bk']*g['s'+operand])
    if operand=='a':rows,width,stride,start=g['m'],2*extent,2*g['global_k'],2*(g['mo']*g['global_k']+g['ko'])
    else:rows,width,stride,start=extent,2*g['n'],2*g['global_n'],2*(g['ko']*g['global_n']+g['no'])
    result=Counter()
    for r in range(rows):
        begin=start+r*stride;end=begin+width
        result.update(range(begin//grain,(end-1)//grain+1))
    return result


def analyze_source_demand(program,grains=(128,512),max_waves=None):
    cohorts=defaultdict(list)
    for ci,jobs in enumerate(program['work']):
        for ji,job in enumerate(jobs):
            if max_waves is None or ji<max_waves:cohorts[ji].append((ci,job['geometry']))
    summaries=[]
    for ji,jobs in cohorts.items():
        for operand in ('a','b'):
            for grain in grains:
                clients=[(ci,_visits(g,operand,grain)) for ci,g in jobs]
                clients=[(ci,v) for ci,v in clients if v is not None]
                if not clients:continue
                total=Counter();distinct=Counter()
                for _,v in clients:total.update(v);distinct.update(v.keys())
                result=[]
                for ci,v in clients:
                    visits=sum(v.values());peer=sum(count*(total[line]-count) for line,count in v.items())
                    result.append(dict(core=ci,row_interval_visits=visits,unique_intervals=len(v),
                        same_core_revisits=visits-len(v),weighted_peer_visits=peer,
                        mean_other_core_visits_per_visit=peer/max(1,visits),
                        max_total_visits_to_touched_interval=max(total[line] for line in v),
                        max_distinct_clients_to_touched_interval=max(distinct[line] for line in v)))
                # Exact counting identity, independent of timing semantics.
                peer_sum=sum(x['weighted_peer_visits'] for x in result)
                expected=sum(count*count for count in total.values())-sum(count*count for _,v in clients for count in v.values())
                assert peer_sum==expected
                summaries.append(dict(nominal_job_wave=ji,operand=operand,interval_bytes=grain,
                    clients=len(clients),total_row_interval_visits=sum(total.values()),unique_intervals=len(total),
                    max_visits_per_interval=max(total.values()),per_core=result,
                    conservation_identity_verified=True))
    return dict(summary=summaries,source_layout='ND-only;NZ operands omitted',
        semantics='First descriptor of each nominal per-core job;row/address geometry,not actual simultaneous arrivals or cache requests.',
        attached_to_timing_formula=False,physical_grain_asserted=False)
