"""Shared exact event DAG, static compiled-layout effective control services.

Overrides are offline constants keyed only by implementation and NZ A/B flags.
Hardware/component constants remain unchanged. This is a research candidate;
no accuracy or ranking qualification is implied by successful prediction.
"""
from math import isfinite
from . import matmul_four_path_v1080 as Base
from . import matmul_streaming_v918 as Stream

def branch(row):
    if row['implementation']=='catlass':return 'catlass:NN'
    return row['implementation']+':'+str(int(bool(row['plan']['nd2nz_a'])))+str(int(bool(row['plan']['nd2nz_b'])))

def eligible(kind):
    scalar={'catlass':'cat_tasks','base':'base_calls','scsk':'sc_calls','deterministic':'det_calls'}[kind]
    names={'entry.'+kind,'launch.'+kind,'scalar.'+scalar,'scalar.inner_kloops.'+kind,
      'scalar.outer_kloops.'+kind,'release.'+kind,'control.copy.'+kind,'control.load.'+kind}
    if kind!='catlass':
        names.add('init.converted.'+kind)
        names.update('dma.'+kind+'.'+n for n in ('peer_read_mb','peer_copy_calls','peer_b_strided_mb',
          'same_address_b_short_density','interval_b_peer_excess_short_density','b_dst_span_krows',
          'b_dst_extra_column_blocks_span_krows','preload_b_dst_span_krows','a_dst_density',
          'a_fill_stride_mb','a_source_excess_klines','b_strided_mb','b_cache_disable_mb','b_cache_disable_span_mb'))
    return names

def resolve(row,model):
    formula,component=Base.resolve(row,model['baseline_bundle'])
    key=branch(row)
    if key not in model['semantic_service_overrides']:raise ValueError('unregistered compiled semantic branch')
    values=model['semantic_service_overrides'][key];allowed=eligible(row['implementation'])
    if set(values)-allowed:raise ValueError('override outside effective control/layout services')
    if set(values)&set(component['fixed_parameter_values']):raise ValueError('offline hardware constant override forbidden')
    if any(not isfinite(v) or v<0 for v in values.values()):raise ValueError('invalid effective service')
    return formula,dict(component,parameters={**component['parameters'],**values})

def predict(row,model,engine='auto'):
    formula,component=resolve(row,model)
    if engine not in ('auto','periodic','reference'):raise ValueError('unknown CPU arithmetic engine')
    from .matmul_CPU_dispatch_v1006 import periodic_useful
    periodic=engine=='periodic' or (engine=='auto' and periodic_useful(row))
    if periodic:
        from . import matmul_CPU_periodic_v1188 as Periodic
        result=Periodic.predict(formula,row,component)
    else:result=Stream.predict(formula,row,component)
    result.update(model_version=model['version'],semantic_branch=branch(row),
      common_exact_event_graph=True,shared_hardware_constants_unchanged=True,
      timing_input_required=False,hardware_execution_performed=False,
      accuracy_qualified=False,stable_ranking_qualified=False)
    return result
