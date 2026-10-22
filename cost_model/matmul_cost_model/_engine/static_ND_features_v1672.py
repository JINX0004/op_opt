"""Source-reviewed ordinary ND geometry; no target timings or case keys."""
from math import log2
from matmul_cost_model._engine.static_halfphase_features_v1586 import features as prior_features,NAMES as PRIOR_NAMES,correction
from matmul_cost_model._engine.costmodel.matmul_source_ND_persistence_v1664 import ordinary,phase_factors,persistence
EXTRA_NAMES=('ordinary_ND_A','ordinary_ND_high_phase','ordinary_ND_persistent_high',
             'ordinary_ND_phase_transitions','ordinary_ND_A_extra_phase256','ordinary_ND_source_pitch_phase128')
NAMES=PRIOR_NAMES+EXTRA_NAMES

def extra_features(row):
    if not ordinary(row):return [0.]*len(EXTRA_NAMES)
    bad,turn=phase_factors(row)
    raw=(1.,bad,persistence(row),turn,row.get('source_a_offset_extra_bytes',0)%256/256,
         (2*row['shape'][2])%128/128)
    return [log2(1+v) for v in raw]

def features(row,pipeline_us,model):return prior_features(row,pipeline_us,model)+extra_features(row)
