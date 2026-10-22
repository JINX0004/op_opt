"""One fixed source10 BASE/SC correction; all other semantic paths unchanged."""
from math import exp,isfinite
from copy import deepcopy
from matmul_cost_model._engine.NN_CAT_partial_C0_predict_v2856 import Predictor as Parent
from matmul_cost_model._engine.NN_source_window_basis_v2712 import transform,names
from matmul_cost_model._engine.matmul_static_unified_v2342 import validate_nn

def enabled(r):
    p=r.get('plan',{})
    return r['implementation'] in ('base','scsk') and bool(p.get('nd2nz_a')) and not p.get('nd2nz_b')

def apply_head(r,b,model):
    assert enabled(r) and isfinite(b) and b>0
    spec=model['static_tanh_basis'];head=model['head'];assert tuple(head['feature_names'])==names(spec['hidden'])
    x=transform(r,b,spec);c=head['implementation_corrections'][r['implementation']]
    delta=c['weights'][0]+sum(w*(v-a)/s for w,v,a,s in zip(c['weights'][1:],x,c['center'],c['scale']))
    cap=head['correction_cap'];delta=max(-cap,min(cap,delta))
    return b*exp(delta),delta

class Predictor:
    def __init__(self,model):
        assert model['static_gate']=='native10 BASE/SC, A converted/B ND'
        self.model=deepcopy(model);self.parent=Parent(model['parent2858_model']);self.service=self.parent.all.service
    def predict(self,r):
        validate_nn(r)
        if not enabled(r):return self.parent.predict(r)
        static={k:r[k] for k in ('implementation','shape','cores','dtype','trans_a','trans_b','plan','tile','l1_tile','l0_tile','execution_profile','transport','source_a_offset_extra_bytes') if k in r}
        b=self.service.predict(static);value,delta=apply_head(static,b['predicted_us'],self.model)
        return dict(predicted_us=value,computed_static_service13_us=b['predicted_us'],bounded_head_log_correction=delta,
            source10_recalibrated=True,model_version=self.model['version'],target_measurements_required=False,
            hardware_execution_performed=False,independent_accuracy_qualified=False,training_only=True)
