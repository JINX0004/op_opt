"""Fixed smooth directions; offline fitted output only, no lookup table."""
from math import tanh
from matmul_cost_model._engine.NN_source_windows_v2711 import features,NAMES

def names(hidden):return NAMES+tuple('static_tanh_'+str(i) for i in range(hidden))

def transform(row,base,spec):
    assert tuple(spec['base_feature_names'])==NAMES
    x=features(row,base);n=spec['normalizers'][row['implementation']]
    z=[(v-c)/s for v,c,s in zip(x,n['center'],n['scale'])]
    return x+[tanh(b+sum(a*v for a,v in zip(w,z))) for w,b in zip(spec['directions'],spec['bias'])]

