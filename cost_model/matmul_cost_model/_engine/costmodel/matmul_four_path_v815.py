"""807 control: kernel initialization and converter start concurrently.

Source AIC Init precedes WaitCV while AIV converts. Both must complete before
core GEMM; converter elapsed includes AIV setup and hardware synchronization.
Effective entry parameters are still hypotheses, not uniquely identified Init.
"""
from . import matmul_four_path_v740 as B
from . import matmul_four_path_v807 as R
from .matmul_four_path_v320 import validate_prediction_contract
from math import isfinite
PARAMETER_NAMES=R.PARAMETER_NAMES
INDEX=R.INDEX
FIXED_INTERVALS=R.FIXED_INTERVALS
compile_plan=R.compile_plan

class EventGraph(B.EventGraph):
 def __init__(self):super().__init__();self.head=0;self.phase_attached=False;self.conversion_us=0.
 def plain_node(self,predecessors=(),features=None,fraction=1.):
  ps=sorted(set(predecessors)-{0})
  if not ps and self.head:ps=[self.head]
  self.predecessors.append(ps);self.features.append([(INDEX[k],float(v)*fraction) for k,v in (features or {}).items() if v]);return len(self.features)-1
 def node(self,predecessors=(),features=None,fraction=1.):
  if features and any(k.startswith('entry.') for k in features):
   if self.phase_attached:raise RuntimeError('final phase attached twice')
   self.phase_attached=True
   marker=next(k for k in features if k.startswith('entry.'))
   features={k:v for k,v in features.items() if not k.startswith(('conversion.','entry.','launch.'))}
   features[marker]=0. # Retain740 final-body hook, charge startup only at head.
  return super().node(predecessors,features,fraction)

def compile_graph(row,graph=None,*,conversion_constants,**schedule):
 validate_prediction_contract(row);graph=graph or EventGraph()
 if not isinstance(graph,EventGraph):raise ValueError('815 event graph required')
 if set(conversion_constants)!=set(R.C.PARAMETER_NAMES):raise ValueError('complete frozen conversion constants required')
 kind=row['implementation'];graph.conversion_us=0. if kind=='catlass' else R._conversion_us(*row['shape'],row['cores'],bool(row['plan']['nd2nz_a']),bool(row['plan']['nd2nz_b']),tuple(conversion_constants[n] for n in R.C.PARAMETER_NAMES))
 init=graph.plain_node(features={'entry.'+kind:1,'launch.'+kind:row['cores']})
 conversion=graph.plain_node(features={'conversion.frozen_pipeline_us':graph.conversion_us})
 graph.head=graph.plain_node([init,conversion])
 result=B.compile_graph(row,graph,**schedule)
 if not graph.phase_attached:raise RuntimeError('final phase missing')
 return result

def predict(row,model):
 validate_prediction_contract(row,model);theta=model['parameters']
 if set(theta)!=set(PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid offline services')
 if any(abs(theta[n]-v)>1e-12*max(1,v) for n,v in FIXED_INTERVALS.items()):raise ValueError('fixed component constant changed')
 graph,sink,p=compile_graph(row,**model['schedule']);value=graph.evaluate([theta[n] for n in PARAMETER_NAMES],sink)
 return dict(predicted_us=value,model_version=model.get('version','four_path_v815'),active_cores=p['active_cores'],launched_cores=p['cores'],split_factor=p['split_factor'],generated_events=len(graph.features),conversion_phase_us=graph.conversion_us,init_conversion_overlap_hypothesis=True,timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False,stable_ranking_qualified=False)
