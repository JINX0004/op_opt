"""Evaluate emitted source events immediately; preserve formula plain_node hooks."""
from math import isfinite

class _Predecessors:
 def __init__(self,graph):self.graph=graph
 def __len__(self):return len(self.graph._finish)
 def append(self,parents):
  g=self.graph
  if g._pending is not None:raise RuntimeError('unpaired predecessor emission')
  if any(type(p)!=int or not 0<=p<len(g._finish) for p in parents):raise RuntimeError('source graph is not topological')
  g._pending=parents

class _Features:
 def __init__(self,graph):self.graph=graph
 def __len__(self):return len(self.graph._finish)
 def append(self,features):
  g=self.graph
  if g._pending is None:raise RuntimeError('missing predecessor emission')
  g._finish.append(max((g._finish[p] for p in g._pending),default=0.)+sum(g._theta[k]*v for k,v in features));g._pending=None

def graph_type(formula):
 class StreamingGraph(formula.EventGraph):
  def __init__(self,theta):
   super().__init__();self._theta=theta;self._finish=[0.];self._pending=None;self.predecessors=_Predecessors(self);self.features=_Features(self)
  def evaluate(self,theta,sink):
   if tuple(theta)!=self._theta or self._pending is not None:raise ValueError('streamed weights changed or incomplete node')
   return self._finish[sink]
 return StreamingGraph

def predict(formula,row,model):
 formula.validate_prediction_contract(row,model);theta=model['parameters']
 if set(theta)!=set(formula.PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid offline services')
 fp32=model.get('component_fixed_native_fp32',{})
 if set(fp32)!={*formula.FIX_COMPLETION_NAMES,*('writeback.native_fp32.'+n for n in formula.FP32_FIELDS)}:raise ValueError('missing fixed FP32 calibration')
 if set(model.get('component_fixed_mte2',{}))!=set(formula.D.PARAMETER_NAMES):raise ValueError('missing fixed MTE2 calibration')
 fixed={**formula.FIXED_INTERVALS,**model['component_fixed_mte2'],**fp32}
 if any(abs(theta[n]-v)>1e-12*max(1,v) for n,v in fixed.items()):raise ValueError('fixed component service changed')
 values=tuple(theta[n] for n in formula.PARAMETER_NAMES);graph=graph_type(formula)(values)
 graph,sink,p=formula.compile_graph(row,graph=graph,**model['schedule']);value=graph.evaluate(values,sink)
 return dict(predicted_us=value,model_version=model['version'],active_cores=p['active_cores'],launched_cores=p['cores'],split_factor=p['split_factor'],generated_events=len(graph.features),conversion_phase_us=graph.conversion_us,timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False,stable_ranking_qualified=False,source_plain_node_hooks_preserved=True)
