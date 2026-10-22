"""Four-path740 core graph with separately frozen source-derived ND2NZ phase.

One common max-plus framework; implementation-dependent source schedules remain
explicit. Obsolete coarse conversion services must be zero, new phase scale1.
The new converter constants come only from offline component calibration.
"""
from math import isfinite
from functools import lru_cache
from . import matmul_four_path_v740 as B
from . import matmul_conversion_v803 as C
from .matmul_four_path_v320 import validate_prediction_contract
EXTRA_NAMES=('conversion.frozen_pipeline_us',)
PARAMETER_NAMES=B.PARAMETER_NAMES+EXTRA_NAMES
INDEX={n:i for i,n in enumerate(PARAMETER_NAMES)}
FIXED_INTERVALS={**B.FIXED_INTERVALS,**{n:0. for n in B.PARAMETER_NAMES if n.startswith('conversion.')},EXTRA_NAMES[0]:1.}
compile_plan=B.compile_plan

@lru_cache(maxsize=4096)
def _conversion_us(m,n,k,cores,a,b,constants):
 row=dict(implementation='base',shape=[m,n,k],cores=cores,plan=dict(dtype='float16',trans_a=False,trans_b=False,nd2nz_a=a,nd2nz_b=b))
 return C.predict_conversion(row,dict(zip(C.PARAMETER_NAMES,constants)))

class EventGraph(B.EventGraph):
 def __init__(self):super().__init__();self.conversion_us=0.;self.phase_attached=False
 def plain_node(self,predecessors=(),features=None,fraction=1.):
  self.predecessors.append(sorted(set(predecessors)-{0}));self.features.append([(INDEX[k],float(v)*fraction) for k,v in (features or {}).items() if v]);return len(self.features)-1
 def node(self,predecessors=(),features=None,fraction=1.):
  if features and any(k.startswith('entry.') for k in features):
   if self.phase_attached:raise RuntimeError('conversion attached twice')
   self.phase_attached=True;features={k:v for k,v in features.items() if not k.startswith('conversion.')}
   if self.conversion_us:features[EXTRA_NAMES[0]]=self.conversion_us
  return super().node(predecessors,features,fraction)

def compile_graph(row,graph=None,*,conversion_constants,**schedule):
 validate_prediction_contract(row);graph=graph or EventGraph()
 if not isinstance(graph,EventGraph):raise ValueError('807 event graph required')
 if set(conversion_constants)!=set(C.PARAMETER_NAMES):raise ValueError('complete frozen conversion constants required')
 graph.conversion_us=0. if row['implementation']=='catlass' else _conversion_us(*row['shape'],row['cores'],bool(row['plan']['nd2nz_a']),bool(row['plan']['nd2nz_b']),tuple(conversion_constants[n] for n in C.PARAMETER_NAMES))
 result=B.compile_graph(row,graph,**schedule)
 if not graph.phase_attached:raise RuntimeError('conversion phase missing')
 return result

def predict(row,model):
 validate_prediction_contract(row,model);theta=model['parameters']
 if set(theta)!=set(PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid offline services')
 if any(abs(theta[n]-v)>1e-12*max(1,v) for n,v in FIXED_INTERVALS.items()):raise ValueError('fixed component constant changed')
 graph,sink,p=compile_graph(row,**model['schedule']);value=graph.evaluate([theta[n] for n in PARAMETER_NAMES],sink)
 return dict(predicted_us=value,model_version=model.get('version','four_path_v807'),active_cores=p['active_cores'],launched_cores=p['cores'],split_factor=p['split_factor'],generated_events=len(graph.features),conversion_phase_us=graph.conversion_us,timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False,stable_ranking_qualified=False)
