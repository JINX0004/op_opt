"""Apply the CAT repair only to partial-C0 B columns; preserve other paths.

FP16/BF16 research geometry uses C0=16 elements. This is a static layout
branch, not a case-ID selector. Existing FP16 public contract is unchanged.
"""
from copy import deepcopy
from matmul_cost_model._engine.NN_CAT_service_recalibrated_predict_v2812 import Predictor as CompleteCAT
from matmul_cost_model._engine.matmul_static_unified_v2342 import validate_nn

def partial_CAT(row):return row['implementation']=='catlass' and row['shape'][1]%16!=0

class Predictor:
    def __init__(self,model):
        if model['CAT_head_gate']!='partial source-B C0 (N%16!=0), NN FP16':raise ValueError('static partial-C0 gate required')
        self.model=deepcopy(model);self.all=CompleteCAT(model)
    def predict(self,row):
        validate_nn(row)
        repaired=partial_CAT(row)
        value=self.all.predict(row) if repaired or row['implementation']!='catlass' else self.all.parent.predict(row)
        return dict(value,model_version=self.model['version'],branch_model_version=value['model_version'],
            CAT_partial_C0_head_applied=repaired,static_layout_gate=True,target_measurements_required=False,
            hardware_execution_performed=False,independent_accuracy_qualified=False)
