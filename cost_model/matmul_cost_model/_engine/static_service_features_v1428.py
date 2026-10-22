"""Static work plus frozen-predictor summaries. No measured target features."""
from math import log2,exp,tanh
from matmul_cost_model._engine.static_work_features_v1362 import features as work_features,NAMES as WORK_NAMES
from matmul_cost_model._engine.costmodel import matmul_semantic_services_v1403 as Semantic
NAMES=WORK_NAMES+('pred_pipeline_us','pred_conversion_fraction','pred_head_fraction',
  'pred_cube_work_fraction','pred_scalar_API_work_fraction','pred_dense_byte_service_fraction')
def features(row,pipeline_us,model):
    x=work_features(row);raw={n:2**v-1 for n,v in zip(WORK_NAMES,x)}
    F,c=Semantic.resolve(row,model);theta=c['parameters'];kind=row['implementation']
    if kind=='catlass':conv=0.;converted=False
    else:
        a,b=bool(row['plan']['nd2nz_a']),bool(row['plan']['nd2nz_b']);converted=a or b
        conv=F.conversion_us(*row['shape'],row['cores'],a,b,tuple(c['schedule']['conversion_constants'][n] for n in F.C.PARAMETER_NAMES))
    init=theta['entry.'+kind]+theta['launch.'+kind]*row['cores']+(theta['init.converted.'+kind] if converted else 0.)
    # These are static service proxies, not PMU activity or exact overlap bounds.
    cube=theta['cube.cube_kcalls']*raw['max_mmad']/1000+theta['cube.cube_kblocks']*raw['max_padded_GF']*1e9/(4096*1000)
    api=theta['scalar.'+{'catlass':'cat_tasks','base':'base_calls','scsk':'sc_calls','deterministic':'det_calls'}[kind]]*raw['max_jobs']
    dense=theta['mte2.nd.interval.dense_mb']*raw['max_source_MB']
    values=[pipeline_us,conv/pipeline_us,max(init,conv)/pipeline_us,cube/pipeline_us,api/pipeline_us,dense/pipeline_us]
    return x+[log2(1+max(0,v)) for v in values]
def correction(x,model):
    z=[(a-b)/s for a,b,s in zip(x,model['center'],model['scale'])]
    h=[tanh(sum(a*w for a,w in zip(z,col))+b) for col,b in zip(model['W_columns'],model['hidden_bias'])]
    raw=model['output_bias']+sum(a*w for a,w in zip(h,model['output_weights']))
    cap=model['residual_log_cap'];return exp(cap*tanh(raw/cap))
