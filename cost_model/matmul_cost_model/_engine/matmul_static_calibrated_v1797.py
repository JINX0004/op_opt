"""One source DAG, fixed per-implementation phase/candidate corrections."""
from math import exp,tanh,isfinite
from matmul_cost_model._engine.static_core_path_features_v1773 import features,NAMES,correction
from matmul_cost_model._engine.static_serial_head_v1785 import serial_head,head_features,HEAD_NAMES
from matmul_cost_model._engine.matmul_hierarchical_residual_v1651 import common_features,COMMON_NAMES
from matmul_cost_model._engine.matmul_serial_phase_residual_v1787 import linear_factor
from matmul_cost_model._engine.costmodel import matmul_static_path_trace_v1755 as Trace

def evaluate_static_features(kind,model,pipeline_us,x,u,v,head):
    """Internal offline arithmetic on reconstructed features, never labels."""
    branches=model['implementation_formulas'][kind]
    if abs(sum(b['weight'] for b in branches)-1)>1e-12:raise ValueError('invalid fixed weights')
    values=[]
    for branch in branches:
        w=branch['weight']
        if not isfinite(w) or w<0:raise ValueError('invalid fixed branch weight')
        cfg=branch['parameters'];family=branch['family']
        if family=='full_graph_residual':
            xx=[x[i] for i in branch['feature_indices']]
            value=pipeline_us*correction(xx,cfg['candidate'])*(correction(u,cfg['common']) if cfg['common'] else 1.)
        elif family=='serial_head_body':
            body=pipeline_us-head
            if body<0:raise ValueError('invalid static serial decomposition')
            value=head*linear_factor(v,cfg['head'])+body*correction(x[:len(cfg['prior_body']['center'])],cfg['prior_body'])*linear_factor(x,cfg['body'])
        else:raise ValueError('unregistered fixed formula family')
        values.append(w*value)
    return sum(values)

def predict(row,model,engine='auto'):
    if engine not in ('auto','reference'):raise ValueError('unknown CPU engine')
    if tuple(model['feature_names'])!=NAMES or tuple(model['head_feature_names'])!=HEAD_NAMES or tuple(model['common_feature_names'])!=COMMON_NAMES:raise ValueError('static feature contract differs')
    base=Trace.predict(row,model['baseline_model']);p=base['predicted_us'];x=features(row,p,model['baseline_model'],path=base)
    head=serial_head(row,model['baseline_model']);value=evaluate_static_features(row['implementation'],model,p,x,common_features(row),head_features(row),head)
    return dict(base,pipeline_us=p,predicted_us=value,model_version=model['version'],fixed_per_implementation_weights=True,
        no_shape_time_lookup=True,timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False,stable_ranking_qualified=False)
