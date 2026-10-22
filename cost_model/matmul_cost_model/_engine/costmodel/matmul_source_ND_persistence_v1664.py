"""Ordinary-ND A phase persistence and128B crossings, fixed offline services.

The persistent fraction equals mean(b[j]*b[j-1])=bad-turn/2, so alternating
phases need not be penalized. These are geometry proxies, not bank claims.
"""
from math import isfinite
from . import matmul_source_phase_v1579 as Prior,matmul_source_ND_phase_v1655 as Geometry
from . import matmul_streaming_v918 as Stream,matmul_CPU_catlass_periodic_v1595 as Fast
ordinary=Geometry.ordinary
phase_factors=Geometry.phase_factors
FIELD=Geometry.FIELD
CROSS='dma.catlass.a_row_crossing_klines'

def persistence(row):
    bad,turn=phase_factors(row);value=bad-turn/2
    if value < -1e-12:raise ValueError('invalid cyclic phase density')
    return max(0.,value)

def resolve(row,model):
    F,c=Prior.resolve(row,model)
    if ordinary(row):
        if any(n in c['fixed_parameter_values'] or n in F.FIXED_INTERVALS for n in (FIELD,CROSS)):
            raise ValueError('hard hardware rate override forbidden')
        values=[model[n] for n in ('ND_a_krow_base_rate','ND_persistent_high_rate','ND_cross128_extra_rate')]
        if any(not isfinite(v) or v<0 for v in values):raise ValueError('invalid nonnegative ND services')
        c=dict(c,parameters={**c['parameters'],FIELD:values[0]+values[1]*persistence(row),CROSS:c['parameters'][CROSS]+values[2]})
    return F,c

def predict(row,model,engine='auto'):
    if engine not in ('auto','periodic','reference'):raise ValueError('unknown CPU engine')
    F,c=resolve(row,model)
    if engine=='reference':r=Stream.predict(F,row,c)
    elif row['implementation']=='catlass':r=Fast.predict(F,row,c)
    else:
        from .matmul_CPU_dispatch_v1006 import periodic_useful
        if engine=='periodic' or periodic_useful(row):
            from . import matmul_CPU_periodic_v1188 as P
            r=P.predict(F,row,c)
        else:r=Stream.predict(F,row,c)
    r.update(model_version=model['version'],ordinary_ND_A_source=ordinary(row),static_ND_persistent_high_fraction=persistence(row),
        geometry_service_NOT_proven_bank_mechanism=True,hardware_execution_performed=False,timing_input_required=False,accuracy_qualified=False,stable_ranking_qualified=False)
    return r
