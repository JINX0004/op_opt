"""Serial common head plus dependent body, fixed offline corrections."""
from math import exp,tanh
from matmul_cost_model._engine.static_core_path_features_v1773 import features,NAMES,correction
from matmul_cost_model._engine.static_serial_head_v1785 import serial_head,head_features,HEAD_NAMES
from matmul_cost_model._engine.costmodel import matmul_static_path_trace_v1755 as Trace
def linear_factor(x,c):
    if len(x)!=len(c['weights']):raise ValueError('phase feature dimensions differ')
    value=c['bias']+sum(w*(v-m)/s for w,v,m,s in zip(c['weights'],x,c['center'],c['scale']))
    return exp(c['log_cap']*tanh(value/c['log_cap']))
def predict(row,model,engine='auto'):
    if engine not in ('auto','reference'):raise ValueError('unknown CPU engine')
    if tuple(model['feature_names'])!=NAMES or tuple(model['head_feature_names'])!=HEAD_NAMES:raise ValueError('phase contract mismatch')
    base=Trace.predict(row,model['baseline_model']);head=serial_head(row,model['baseline_model']);body=base['predicted_us']-head
    if body<0:raise RuntimeError('negative dependent body')
    x=features(row,base['predicted_us'],model['baseline_model'],path=base);cfg=model['models'][row['implementation']]
    factor=correction(x[:len(cfg['prior_body']['center'])],cfg['prior_body'])*linear_factor(x,cfg['body'])
    hh=head*linear_factor(head_features(row),cfg['head']);bb=body*factor
    return dict(base,pipeline_us=base['predicted_us'],predicted_us=hh+bb,corrected_common_head_us=hh,corrected_body_us=bb,
        same_input_same_cores_same_conversion_head_tiling_invariant=True,model_version=model['version'],
        accuracy_qualified=False,stable_ranking_qualified=False,hardware_execution_performed=False,timing_input_required=False)
