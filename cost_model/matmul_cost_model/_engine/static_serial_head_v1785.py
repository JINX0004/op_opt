"""Exact common root of frozen942/1120 event DAG, not isolated latency."""
from math import log2,isfinite
from matmul_cost_model._engine.costmodel import matmul_source_phase_v1579 as Source
from matmul_cost_model._engine.matmul_hierarchical_residual_v1651 import common_features,COMMON_NAMES
HEAD_NAMES=COMMON_NAMES+('launched_cores','A_NZ','B_NZ')
def serial_head(row,model):
    F,c=Source.resolve(row,model);F.validate_prediction_contract(row,c)
    if F.__name__.rsplit('_v',1)[-1] not in ('942','1120'):raise ValueError('unreviewed common root topology')
    kind=row['implementation'];a=kind!='catlass' and bool(row['plan']['nd2nz_a']);b=kind!='catlass' and bool(row['plan']['nd2nz_b'])
    theta=c['parameters'];init=theta['entry.'+kind]+row['cores']*theta['launch.'+kind]+(theta['init.converted.'+kind] if a or b else 0.)
    conv=0. if kind=='catlass' else F.conversion_us(*row['shape'],row['cores'],a,b,tuple(c['schedule']['conversion_constants'][n] for n in F.C.PARAMETER_NAMES))
    result=max(init,conv)
    if not isfinite(result) or result<0:raise ValueError('invalid static common head')
    return result
def head_features(row):
    p=row.get('plan',{});return common_features(row)+[log2(1+row['cores']),log2(1+int(bool(p.get('nd2nz_a',False)))),log2(1+int(bool(p.get('nd2nz_b',False))))]
