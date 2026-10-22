"""Run from any directory: python examples/use_api.py. No installation needed."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from matmul_cost_model import MatmulCostModel,make_config,generate_candidates

model=MatmulCostModel()
config=make_config('ops_nn_deterministic_split_k',[16,16,102400],'fp16',24,
                   {'base_m':128,'base_n':128,'base_k':128,'step_k':2})
print(json.dumps(model.predict(config),ensure_ascii=False,indent=2))
candidates=generate_candidates([64,128,512],'bf16',implementations=('base',),cores=(1,2,4,8))
print(json.dumps(model.compare(candidates,top=3)['groups'],ensure_ascii=False,indent=2))
