"""Static paired-event head increment, not a pure hardware conversion latency.

The two controls execute source Init+conversion+sync and the same Init without
conversion. Fixed offline models estimate their event times from static work.
Their positive difference includes initialization/conversion overlap and
synchronization. No target timing, PMU, shape timing table or NPU is consumed.
"""
from math import log2,isfinite
from functools import lru_cache
from . import matmul_conversion_v854 as C

FAMILIES=('onlyA','onlyB','both')
INIT_NAMES=tuple(f+'.'+n for f in FAMILIES for n in ('entry','extra_cores','log2_cores'))

def init_features(row):
    p=row['plan'];a,b=bool(p['nd2nz_a']),bool(p['nd2nz_b']);c=row['cores']
    if not (a or b):return {}
    family='both' if a and b else 'onlyA' if a else 'onlyB'
    return {family+'.entry':1.,family+'.extra_cores':float(c-1),family+'.log2_cores':log2(c)}

def predict_init(row,weights):
    return sum(weights[n]*v for n,v in init_features(row).items())

@lru_cache(maxsize=4096)
def _static(m,n,k,cores,a,b,init_tuple,head_tuple):
    row=dict(implementation='base',shape=[m,n,k],cores=cores,plan=dict(dtype='float16',trans_a=False,trans_b=False,nd2nz_a=a,nd2nz_b=b))
    init=predict_init(row,dict(zip(INIT_NAMES,init_tuple)))
    head=max(init,C.predict_conversion(row,dict(zip(C.PARAMETER_NAMES,head_tuple))))
    return head,init,max(0.,head-init)

def predict(row,constants):
    init=constants['init_weights'];head=constants['head_weights']
    if set(init)!=set(INIT_NAMES) or set(head)!=set(C.PARAMETER_NAMES):raise ValueError('complete fixed event-head constants required')
    if any(not isfinite(v) or v<0 for v in (*init.values(),*head.values())):raise ValueError('invalid event-head services')
    p=row['plan']
    if p.get('dtype','float16')!='float16' or p.get('trans_a',False) or p.get('trans_b',False):raise ValueError('FP16 NN event head only')
    h,i,d=_static(*row['shape'],row['cores'],bool(p['nd2nz_a']),bool(p['nd2nz_b']),tuple(init[n] for n in INIT_NAMES),tuple(head[n] for n in C.PARAMETER_NAMES))
    return dict(head_event_us=h,init_event_us=i,increment_us=d,timing_input_required=False,not_pure_conversion_latency=True)
