"""Static warm400/target-core profile contract; fixed offline coefficients only."""
from .matmul_four_path_v301 import *
from .matmul_four_path_v301 import compile_graph as _compile_graph, predict as _predict
EXECUTION_PROFILE='warm400_gate_targetC'
def _check(row,model=None):
    for data in (row,row.get('plan',{}),row.get('transport',{})):
        if data.get('execution_profile',EXECUTION_PROFILE)!=EXECUTION_PROFILE:
            raise ValueError('model requires fixed warm400/gate_targetC execution profile')
    if model is not None and model.get('execution_profile')!=EXECUTION_PROFILE:
        raise ValueError('profile-specific fixed offline model required')
validate_prediction_contract=_check
def compile_graph(row,*args,**kwargs):
    _check(row)
    return _compile_graph(row,*args,**kwargs)
def predict(row,model):
    _check(row,model)
    result=_predict(row,model)
    result['execution_profile']=EXECUTION_PROFILE
    return result
