"""Separate ordinary-ND A source phase service; fixed offline coefficients.

Uses the existing A-row feature; B, rowwise copies and native services stay
unchanged. Phase densities are effective geometry proxies, not bank proofs.
"""
from math import gcd,isfinite
from . import matmul_source_phase_v1579 as Prior,matmul_streaming_v918 as Stream
from . import matmul_CPU_catlass_periodic_v1595 as Fast
FIELD='dma.catlass.a_krows'

def ordinary(row):
    return row['implementation']=='catlass' and not Prior.Prior.rowwise(row) and row['shape'][0]%16==0 and row['shape'][2]%16==0

def phase_factors(row):
    if not ordinary(row):return (0.,0.)
    extra=row.get('source_a_offset_extra_bytes',0)
    if type(extra)!=int or extra not in (0,128,256,384):raise ValueError('unreviewed ordinary-ND A phase')
    pitch=2*row['shape'][2]%256;period=256//gcd(pitch,256)
    bits=[int((extra+j*pitch)%256>=128) for j in range(period)]
    return sum(bits)/period,sum(bits[j]!=bits[j-1] for j in range(period))/period

def resolve(row,model):
    F,c=Prior.resolve(row,model)
    if ordinary(row):
        if FIELD in c['fixed_parameter_values'] or FIELD in F.FIXED_INTERVALS:raise ValueError('hard hardware rate override forbidden')
        values=[model[k] for k in ('ND_a_krow_base_rate','ND_higher_half_phase_rate','ND_phase_transition_rate')]
        if any(not isfinite(v) or v<0 for v in values):raise ValueError('invalid fixed ordinary-ND service')
        bad,turn=phase_factors(row);rate=values[0]+values[1]*bad+values[2]*turn
        c=dict(c,parameters={**c['parameters'],FIELD:rate})
    return F,c

def predict(row,model,engine='auto'):
    if engine not in ('auto','periodic','reference'):raise ValueError('unknown CPU arithmetic engine')
    F,c=resolve(row,model)
    if engine=='reference':result=Stream.predict(F,row,c)
    elif row['implementation']=='catlass':result=Fast.predict(F,row,c)
    else:
        from .matmul_CPU_dispatch_v1006 import periodic_useful
        if engine=='periodic' or periodic_useful(row):
            from . import matmul_CPU_periodic_v1188 as P
            result=P.predict(F,row,c)
        else:result=Stream.predict(F,row,c)
    bad,turn=phase_factors(row)
    result.update(model_version=model['version'],ordinary_ND_A_source=ordinary(row),static_ND_A_higher_half_fraction=bad,
        static_ND_A_phase_transition_density=turn,source_phase_proxy_NOT_proven_bank_mechanism=True,
        hardware_execution_performed=False,timing_input_required=False,accuracy_qualified=False,stable_ranking_qualified=False)
    return result
