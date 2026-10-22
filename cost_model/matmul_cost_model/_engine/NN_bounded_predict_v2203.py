"""CPU-only NN candidate: frozen1844 plus <=5% static log-linear correction."""
from math import exp,isfinite
from matmul_cost_model._engine.matmul_static_CPU_v1877 import predict as baseline_predict
from matmul_cost_model._engine.matmul_cpu_predict_cli_v1606 import validate_row
from matmul_cost_model._engine.NN_relative_source_features_v2196 import features,NAMES

def apply(kind,base_us,x,model):
 if tuple(model['feature_names'])!=NAMES:raise ValueError('NN feature contract differs')
 if not isfinite(base_us) or base_us<=0:raise ValueError('positive static baseline required')
 c=model['implementation_corrections'][kind]
 value=c['weights'][0]+sum(w*(a-m)/s for w,a,m,s in zip(c['weights'][1:],x,c['center'],c['scale']))
 cap=model['correction_cap'];delta=min(cap,max(-cap,value))
 return dict(predicted_us=float(base_us)*exp(delta),frozen1844_us=float(base_us),static_log_correction=delta,correction_clipped=delta!=value,hardware_execution_performed=False,target_timing_required=False,no_shape_time_lookup=True,accuracy_qualified=False)

def predict(row,model):
 validate_row(row);m,n,k=row['shape']
 if m not in range(16,129,16) or n not in (16,24,32,64,96,128) or not 64<=k<=131072:raise ValueError('outside reviewed NN FP16 research domain')
 base=baseline_predict(row,model['frozen_base_model']);path=dict(base,predicted_us=base['pipeline_us'])
 x=features(row,base['pipeline_us'],model['frozen_base_model']['baseline_model'],path)
 return apply(row['implementation'],base['predicted_us'],x,model)
