"""Static tail/layout and model-dominant-path fractions; never PMU."""
from math import log2
from matmul_cost_model._engine.static_ND_features_v1672 import features as prior_features,NAMES as PRIOR_NAMES,correction
from matmul_cost_model._engine.static_native_tail_features_v1678 import extra_features as tail_features,EXTRA_NAMES as TAIL_NAMES
from matmul_cost_model._engine.costmodel import matmul_static_path_trace_v1755 as Trace
PATH_NAMES=tuple('static_path_'+k+'_fraction' for k in Trace.GROUPS)
NAMES=PRIOR_NAMES+TAIL_NAMES+PATH_NAMES

def features(row,pipeline_us,model,path=None):
    if path is None:path=Trace.predict(row,model)
    if abs(path['predicted_us']-pipeline_us)>1e-7:raise ValueError('static path and pipeline mismatch')
    return prior_features(row,pipeline_us,model)+tail_features(row)+[log2(1+path['fractions'][k]) for k in Trace.GROUPS]
