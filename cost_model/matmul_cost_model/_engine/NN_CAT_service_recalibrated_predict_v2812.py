"""Static source01 path replaces old residuals; other paths frozen2713."""
from math import exp,isfinite
from copy import deepcopy
from matmul_cost_model._engine.NN_source_window_basis_v2712 import transform,names
from matmul_cost_model._engine.NN_semantic01_rebased_predict_v2798 import Predictor as Parent
from matmul_cost_model._engine.NN_thirteen_service_predict_v2578 import Predictor as Service
from matmul_cost_model._engine.NN_bounded_onepass_predict_v2221 import grouped_pipeline,reference_pipeline,use_group_recurrence
from matmul_cost_model._engine.matmul_static_unified_v2342 import validate_nn

def enabled(row):
    return row['implementation']=='catlass'

def apply_physical_head(row,physical_us,model):
    if not enabled(row):raise ValueError('CAT branch only')
    if not isfinite(physical_us) or physical_us<=0:raise ValueError('positive computed physical time required')
    spec=model['static_tanh_basis'];head=model['head'];assert tuple(head['feature_names'])==names(spec['hidden'])
    x=transform(row,physical_us,spec);normal=head['implementation_corrections'][row['implementation']]
    w=normal['weights'];delta=w[0]+sum(a*(v-c)/s for a,v,c,s in zip(w[1:],x,normal['center'],normal['scale']))
    cap=head['correction_cap'];delta=max(-cap,min(cap,delta))
    return physical_us*exp(delta),delta

class Predictor:
    def __init__(self,model):
        self.model=deepcopy(model);self.parent=Parent(model['parent2797_model'])
        service=model['physical_service_model'];self.service=Service(service['old_frozen_model'],service['updated_parameters'],service['graph_version'])
    def predict(self,row):
        validate_nn(row)
        if not enabled(row):return self.parent.predict(row)
        static={k:row[k] for k in ('implementation','shape','cores','dtype','trans_a','trans_b','plan','tile','l1_tile','l0_tile','execution_profile','transport','source_a_offset_extra_bytes') if k in row}
        base=self.service.predict(static)
        value,delta=apply_physical_head(static,base['predicted_us'],self.model)
        return dict(predicted_us=value,pipeline_us=base['new_pipeline_us'],static_service13_us=base['predicted_us'],bounded_head_log_correction=delta,
            CAT_service_recalibrated=True,old_residual_heads_used=True,model_version=self.model['version'],
            target_measurements_required=False,hardware_execution_performed=False,independent_accuracy_qualified=False,training_only=True)
