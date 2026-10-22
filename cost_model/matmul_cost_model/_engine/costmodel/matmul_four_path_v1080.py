"""Shared source DAG with implementation-conditioned offline effective rates.

Component constants remain identical across implementations. Only effective
full-kernel service constants that were already free may differ. No shape
timing table, measured target features or implementation fallbacks are used.
"""
from . import matmul_four_path_v942 as Common
from . import matmul_four_path_v1064 as Destination
from . import matmul_four_path_v1103 as EventHead
from . import matmul_four_path_v1121 as AlignedSource
from . import matmul_four_path_v1120 as AlignedOnly
from . import matmul_four_path_v1142 as RawHead
from . import matmul_streaming_v918 as Stream

def resolve(row,model):
    keys={'catlass','base','scsk','deterministic'}
    if set(model['implementation_models'])!=keys or set(model['implementation_formula_versions'])!=keys:raise ValueError('incomplete implementation bundle')
    kind=row['implementation'];version=model['implementation_formula_versions'][kind]
    if version not in (942,1064,1103,1120,1121,1142):raise ValueError('unsupported source formula')
    component=model['implementation_models'][kind]
    first=model['implementation_models']['base']
    for name in ('component_fixed_mte2','component_fixed_native_fp32','fixed_parameter_values'):
        if component[name]!=first[name]:raise ValueError('shared physical component constants disagree')
    if component['schedule']['conversion_constants']!=first['schedule']['conversion_constants']:raise ValueError('shared conversion constants disagree')
    used_heads=[v['schedule']['event_head_constants'] for v in model['implementation_models'].values() if 'event_head_constants' in v['schedule']]
    if used_heads and any(v!=used_heads[0] for v in used_heads):raise ValueError('shared event head constants disagree')
    return {942:Common,1064:Destination,1103:EventHead,1120:AlignedOnly,1121:AlignedSource,1142:RawHead}[version],component

def predict(row,model):
    formula,component=resolve(row,model)
    result=Stream.predict(formula,row,component)
    result.update(model_version=model['version'],implementation_conditioned_effective_services=True,common_source_DAG_framework=True,accuracy_qualified=False,stable_ranking_qualified=False)
    return result
