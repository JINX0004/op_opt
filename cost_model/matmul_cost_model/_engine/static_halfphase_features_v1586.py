"""Static source phase and layout features on the repeat-checked pipeline."""
from math import log2,gcd
from matmul_cost_model._engine.static_layout_features_v1536 import features as prior_features,NAMES as PRIOR_NAMES,correction
from matmul_cost_model._engine.costmodel import matmul_source_phase_v1579 as Source
NAMES=PRIOR_NAMES+('source_bad128_phase_fraction','source_phase_transition_density','source_quarter128_fraction','source_A_extra_phase512')
def features(row,pipeline_us,model):
    prior=dict(model['baseline_model']);prior['additional_rowwise_A_crossing_us_per_kline']=model['rowwise128_us_per_kline']
    prior['catlass_effective_service_overrides']={**prior['catlass_effective_service_overrides'],Source.FIELD:model['rowwise_setup_us_per_kcall']}
    values=prior_features(row,pipeline_us,prior);bad,turn=Source.phase_factors(row);quarter=0.;extra=0
    if Source.Prior.rowwise(row):
        extra=row.get('source_a_offset_extra_bytes',0);pitch=2*row['shape'][2];period=512//gcd(pitch,512)
        quarter=sum(128<=(extra+i*pitch)%512<256 for i in range(period))/period
    return values+[log2(1+max(0,v)) for v in (bad,turn,quarter,extra/512)]
