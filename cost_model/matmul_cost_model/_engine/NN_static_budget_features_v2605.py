"""Input-derived work divided by COMPUTED model time, never measured time.

q/T_static permits an effective constant startup/work correction to have a
different relative effect on a short kernel and a long kernel. The baseline
must come from the fixed static service predictor, not a benchmark label.
"""
from math import isfinite
from matmul_cost_model._engine.NN_service_static_features_v2597 import features as geometry_features,NAMES as GEOMETRY_NAMES
from matmul_cost_model._engine.static_work_features_v1362 import NAMES as WORK_NAMES
COUNTERS=('launched','active','max_jobs','max_mmad','max_source_A','max_source_B',
          'max_source_MB','total_source_MB','max_padded_GF','launched_idle')
NAMES=GEOMETRY_NAMES+('inverse_static_us',)+tuple(n+'_per_static_us' for n in COUNTERS)+('whole_K_resident_A','whole_K_resident_B')

def features(row,static_prediction_us):
    t=float(static_prediction_us)
    if not isfinite(t) or t<=0:raise ValueError('positive computed static baseline required')
    x=geometry_features(row)
    raw=lambda n:2.**x[WORK_NAMES.index(n)]-1.
    k=row['shape'][2];bk=raw('baseK');sa=raw('stepA');sb=raw('stepB')
    return x+[1/t]+[raw(n)/t for n in COUNTERS]+[float(k<=bk*sa+1e-8),float(k<=bk*sb+1e-8)]
