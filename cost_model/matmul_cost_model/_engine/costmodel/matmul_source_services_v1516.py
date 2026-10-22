"""DEV re-calibration of existing CAT effective services and rowwise API cost.

CAT unitFlag3 workspace writes and its source API differ from native ops-nn.
Only existing effective CAT geometry services may move. Shared calibrated
hardware, native fixed FP32 services and conversion constants remain fixed.
"""
from math import isfinite
from . import matmul_semantic_services_v1403 as Semantic
from . import matmul_streaming_v918 as Stream
from .matmul_source_services_v1497 import rowwise,FIELD

def allowed(component):
    return {n for n in component['parameters'] if n.startswith(('dma.catlass.','writeback.fp32.')) and n not in component['fixed_parameter_values']}

def resolve(row,model):
    F,c=Semantic.resolve(row,model['baseline_model']);alpha=model['additional_rowwise_A_crossing_us_per_kline']
    if not isfinite(alpha) or alpha<0:raise ValueError('invalid fixed rowwise source service')
    if row['implementation']=='catlass':
        values=model.get('catlass_effective_service_overrides',{})
        if set(values)-allowed(c):raise ValueError('override outside effective CAT geometry services')
        if set(values)&set(F.FIXED_INTERVALS):raise ValueError('hard fixed hardware service override forbidden')
        if any(not isfinite(v) or v<0 for v in values.values()):raise ValueError('invalid CAT effective service')
        c=dict(c,parameters={**c['parameters'],**values})
        if rowwise(row):c['parameters'][FIELD]+=alpha
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
      hardware_execution_performed=False,timing_input_required=False,accuracy_qualified=False,stable_ranking_qualified=False,
      shared_hardware_components_unchanged=True,existing_effective_CAT_services_recalibrated=True)
    return result
