"""Research six-constant startup/steady correction for native A-ND/B-NZ.

Tnew = Told/Pold * Pnew. The old static correction is kept independent of
the changed service rates: no dominant-feature jump when rates change.
No target timings, PMU, IDs or measured-case table are inference inputs.
"""
from math import isfinite
from copy import deepcopy
from matmul_cost_model._engine import NN_bounded_onepass_predict_v2221 as Old
from matmul_cost_model._engine.matmul_static_unified_v2342 import validate_nn

FIELDS={
 'base':('entry.base','dma.base.a_dst_density'),
 'scsk':('entry.scsk','scalar.sc_calls'),
 'deterministic':('entry.deterministic','dma.deterministic.a_dst_density')}

def physical(model):return model['frozen_base_model']['baseline_model']
def semantics(p):return p['baseline_model']['baseline_model']['semantic_service_overrides']

def initial_parameters(model):
 s=semantics(physical(model))
 return {kind:{field:float(s[kind+':01'][field]) for field in fields} for kind,fields in FIELDS.items()}

def old_value(row,model):
 frozen=model['frozen_base_model'];p=physical(model)
 path=(Old.grouped_pipeline if Old.use_group_recurrence(row) else Old.reference_pipeline)(row,p);base=path['predicted_us']
 x=Old.old_features(row,base,p,path=path)
 t1844=Old.evaluate_static_features(row['implementation'],frozen,base,x,Old.common_features(row),Old.head_features(row),Old.serial_head(row,p))
 value=Old.apply(row['implementation'],t1844,x+Old.extras(row),model)
 return base,value['predicted_us']

class Predictor:
 def __init__(self,old_model,parameters):
  if set(parameters)!=set(FIELDS):raise ValueError('three native fixed-rate groups required')
  for kind,fields in FIELDS.items():
   if set(parameters[kind])!=set(fields) or any(not isfinite(v) or v<0 for v in parameters[kind].values()):raise ValueError('six finite nonnegative offline rates required')
  self.old=deepcopy(old_model);self.parameters=deepcopy(parameters);self.initial=initial_parameters(self.old)
  self.updated_physical=deepcopy(physical(self.old));s=semantics(self.updated_physical)
  for kind,values in parameters.items():s[kind+':01'].update(values)

 def predict(self,row):
  validate_nn(row);p0,t0=old_value(row,self.old);kind=row['implementation'];plan=row.get('plan',{})
  changed=(kind in FIELDS and not plan.get('nd2nz_a') and bool(plan.get('nd2nz_b')) and self.parameters[kind]!=self.initial[kind])
  if changed:
   p1=(Old.grouped_pipeline if Old.use_group_recurrence(row) else Old.reference_pipeline)(row,self.updated_physical)['predicted_us']
   value=t0*(p1/p0)
  else:p1=p0;value=t0
  if not all(isfinite(v) and v>0 for v in [p0,p1,t0,value]):raise ValueError('positive finite reviewed static model required')
  return dict(predicted_us=value,old_frozen_prediction_us=t0,old_pipeline_us=p0,new_pipeline_us=p1,
   static_source_ratio=p1/p0,semantic_01_rates_changed=changed,fixed_parameters_only=True,
   target_measurements_required=False,hardware_execution_performed=False,accuracy_qualified=False,
   steady_rate_fields=FIELDS.get(kind,()),old_correction_independent_of_updated_services=True)
