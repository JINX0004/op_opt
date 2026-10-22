"""Fixed common-input and bounded candidate-relative residual, CPU-only."""
from math import log2
from matmul_cost_model._engine.static_halfphase_features_v1586 import features,correction,NAMES
from matmul_cost_model._engine.costmodel import matmul_source_phase_v1579 as Source,matmul_CPU_catlass_periodic_v1595 as Fast
COMMON_NAMES=('M','N','K','K_requires_conversion','N_requires_conversion','N128_aligned','N_padding_ratio')

def common_features(row):
    m,n,k=row['shape']
    raw=(m,n,k,int(k>65535),int(n%16!=0),int(n%128==0),((n+15)//16*16)/n)
    return [log2(1+v) for v in raw]

def predict(row,model,engine='auto'):
    if tuple(model['feature_names'])!=NAMES or tuple(model['common_feature_names'])!=COMMON_NAMES:
        raise ValueError('hierarchical static feature contract mismatch')
    if row['implementation']=='catlass' and engine!='reference':
        if engine!='auto':raise ValueError('unknown CPU engine')
        F,c=Source.resolve(row,model['baseline_model']);base=Fast.predict(F,row,c)
    else:base=Source.predict(row,model['baseline_model'],engine=engine)
    cfg=model['models'][row['implementation']]
    common=correction(common_features(row),cfg['common']) if cfg['common'] is not None else 1.
    candidate=correction(features(row,base['predicted_us'],model['baseline_model']),cfg['candidate'])
    return dict(base,pipeline_us=base['predicted_us'],predicted_us=base['predicted_us']*common*candidate,
        common_input_factor=common,candidate_factor=candidate,residual_factor=common*candidate,
        model_version=model['version'],common_input_factor_cancels_in_same_input_ranking=True,
        hardware_execution_performed=False,timing_input_required=False,accuracy_qualified=False,stable_ranking_qualified=False)
