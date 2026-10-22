"""DEV hypothesis: existing source crossing service conditioned on actual API.

CATLASS CopyGmToL1<RowMajor> emits one ND2NZ per row for srcDValue>=65536.
The extra effective service is a fixed offline scalar times the existing
static128B crossing proxy. It is not an assertion about hardware cache lines.
"""
from math import isfinite
from . import matmul_semantic_services_v1403 as Semantic
from . import matmul_streaming_v918 as Stream

FIELD='dma.catlass.a_row_crossing_klines'

def rowwise(row):
    return row['implementation']=='catlass' and row['shape'][2]>=65536

def resolve(row,model):
    alpha=model['additional_rowwise_A_crossing_us_per_kline']
    if not isfinite(alpha) or alpha<0:raise ValueError('invalid fixed rowwise source service')
    F,c=Semantic.resolve(row,model['baseline_model'])
    if rowwise(row):
        if FIELD in c['fixed_parameter_values'] or FIELD in F.FIXED_INTERVALS:
            raise ValueError('hard fixed hardware component override forbidden')
        c=dict(c,parameters=dict(c['parameters']))
        c['parameters'][FIELD]+=alpha
    return F,c

def predict(row,model,engine='auto'):
    if engine not in ('auto','reference','periodic'):raise ValueError('unknown CPU arithmetic engine')
    F,c=resolve(row,model)
    from .matmul_CPU_dispatch_v1006 import periodic_useful
    if engine=='periodic' or (engine=='auto' and periodic_useful(row)):
        from . import matmul_CPU_periodic_v1188 as Periodic
        result=Periodic.predict(F,row,c)
    else:result=Stream.predict(F,row,c)
    result.update(model_version=model['version'],rowwise_A_API=bool(rowwise(row)),
      fixed_offline_source_service_only=True,hardware_execution_performed=False,timing_input_required=False,
      accuracy_qualified=False,stable_ranking_qualified=False)
    return result
