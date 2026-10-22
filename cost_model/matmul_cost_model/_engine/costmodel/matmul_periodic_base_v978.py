"""Audited CPU max-plus recurrence shortcut; fixed formula weights, no hardware data.

Only BASE inner loops are eligible. A recorded one-period max-plus transfer is
checked at three points on the proposed affine trajectory before skipping.
The original source files and public/reference predictor remain untouched.
"""
from pathlib import Path
from types import ModuleType,SimpleNamespace,FunctionType
from collections import deque
from math import lcm,ulp
import sys,json
ROOT=Path(__file__).resolve().parent
from . import matmul_four_path_v942 as F,matmul_streaming_v918 as Stream

class TraceFeatures(Stream._Features):
 def append(self,features):
  g=self.graph;parents=tuple(g._pending);cost=sum(g._theta[k]*v for k,v in features);node=len(g._finish)
  g._finish.append(max((g._finish[p] for p in parents),default=0.)+cost);g._pending=None
  g._trace[node]=(parents,cost);g._trace_order.append(node)
  if len(g._trace_order)>6000:del g._trace[g._trace_order.popleft()]

BaseStream=Stream.graph_type(F)
class FastGraph(BaseStream):
 def __init__(self,theta):
  super().__init__(theta);self.features=TraceFeatures(self);self._trace={};self._trace_order=deque();self._history={};self.skipped_nodes=0;self.skipped_inner_steps=0;self.error_bound_us=0.;self.certificates=[]
 def _frontier(self,clocks,free,l0free,source,acc):
  return tuple([clocks[p][field] for p in ('scalar','dma','load','cube') for field in ('issue','complete')]+list(free['a'])+list(free['b'])+list(l0free)+[acc,source['a'],source['b']])
 def try_skip_inner(self,ko_next,bk,k,clocks,free,l0free,source,seq,inner,acc,g):
  if ko_next==bk:self._history={}
  if g['kind']!='base':return ko_next,inner
  span_a,span_b=bk*g['sa'],bk*g['sb'];group=lcm(span_a,span_b,256)
  if ko_next%group:return ko_next,inner
  ids=self._frontier(clocks,free,l0free,source,acc)
  if any(i<=self.head for i in ids):return ko_next,inner
  values=tuple(self._finish[i] for i in ids);phase=(inner%len(l0free),seq['a']%len(free['a']),seq['b']%len(free['b']))
  history=self._history.setdefault(phase,deque(maxlen=2));record=dict(ko=ko_next,ids=ids,values=values,cut=len(self._finish))
  if len(history)<2:history.append(record);return ko_next,inner
  first,previous=history;history.append(record);period=ko_next-previous['ko']
  if period<=0 or previous['ko']-first['ko']!=period or period%group:return ko_next,inner
  periods=(k//group*group-ko_next)//period
  if periods<4:return ko_next,inner
  delta=tuple(b-a for a,b in zip(previous['values'],values));old_delta=tuple(b-a for a,b in zip(first['values'],previous['values']));drift=max(abs(a-b) for a,b in zip(delta,old_delta))
  if min(delta)<-1e-10 or drift>1e-9:return ko_next,inner
  # Alias roles must have the same structure in consecutive periods.
  aliases=lambda vv:tuple(vv.index(v) for v in vv)
  if aliases(previous['ids'])!=aliases(ids):return ko_next,inner
  try:trace=[self._trace[i] for i in range(previous['cut'],record['cut'])]
  except KeyError:return ko_next,inner
  input_map={n:i for i,n in enumerate(previous['ids'])};head=self.head;head_value=self._finish[head]
  def replay(x):
   computed={n:x[i] for n,i in input_map.items()};computed[head]=head_value
   for node,(parents,cost) in enumerate(trace,previous['cut']):
    if any(p not in computed for p in parents):return None
    computed[node]=max((computed[p] for p in parents),default=0.)+cost
   if any(n not in computed for n in ids):return None
   return tuple(computed[n] for n in ids)
  # The macro is max-plus affine in its input frontier. Convex collinearity
  # at0,1,Q bounds its trajectory on the entire interval, not just samples.
  worst=drift
  for q in (0,1,periods-1):
   probe=tuple(v+q*d for v,d in zip(values,delta));result=replay(probe)
   if result is None:return ko_next,inner
   expected=tuple(v+(q+1)*d for v,d in zip(values,delta));worst=max(worst,max(abs(a-b) for a,b in zip(result,expected)))
  future=max(v+periods*d for v,d in zip(values,delta));bound=periods*periods*worst+4*ulp(max(1.,future))*len(trace)*periods
  if bound>1e-9*max(1.,future):return ko_next,inner
  replacements={}
  for n,v,d in zip(ids,values,delta):
   value=v+periods*d
   if n in replacements and abs(replacements[n]-value)>worst+1e-12:return ko_next,inner
   replacements[n]=value
  for n,v in replacements.items():self._finish[n]=v
  advance=periods*period;seq['a']+=advance//span_a;seq['b']+=advance//span_b
  self.skipped_nodes+=periods*len(trace);self.skipped_inner_steps+=advance//bk;self.error_bound_us+=bound
  self.certificates.append(dict(period_inner_steps=period//bk,periods_skipped=periods,probe_residual_us=worst,error_bound_us=bound));self._history={}
  return ko_next+advance,inner+advance//bk

def formula():
 schedule=ModuleType('matmul_cost_model._engine.costmodel.private_fast_schedule');schedule.__package__='matmul_cost_model._engine.costmodel';text=(ROOT/'matmul_schedule_v942.py').read_text()
 assert text.count('            for ko in range(0,k,bk):')==1;text=text.replace('            for ko in range(0,k,bk):','            ko=0\n            while ko<k:')
 line='                acc=cend if small_barrier else cissue;l0free[inner%l0]=cend;inner+=1';assert text.count(line)==1
 text=text.replace(line,line+'\n                ko,inner=graph.try_skip_inner(ko+bk,bk,k,clocks,free,l0free,source,seq,inner,acc,g)')
 exec(compile(text,'private_fast_schedule','exec'),schedule.__dict__)
 globals_map=dict(F.__dict__,Schedule=schedule);compile_graph=FunctionType(F.compile_graph.__code__,globals_map,F.compile_graph.__name__,F.compile_graph.__defaults__,F.compile_graph.__closure__);compile_graph.__kwdefaults__=F.compile_graph.__kwdefaults__
 return SimpleNamespace(**dict(F.__dict__,compile_graph=compile_graph))

MODULE=formula()
def predict(row,model):
 if row['implementation']!='base' or model['schedule'].get('scalar_queue_depth',0)!=0 or model['schedule'].get('wait_location')!='pipeline':return Stream.predict(F,row,model)
 holder={}
 def factory(_formula):
  class LocalGraph(FastGraph):
   def __init__(self,theta):super().__init__(theta);holder['graph']=self
  return LocalGraph
 runner=FunctionType(Stream.predict.__code__,dict(Stream.__dict__,graph_type=factory),Stream.predict.__name__,Stream.predict.__defaults__,Stream.predict.__closure__)
 result=runner(MODULE,row,model);g=holder['graph'];result.update(generated_events=result['generated_events']+g.skipped_nodes,emitted_events=len(g._finish),skipped_inner_steps=g.skipped_inner_steps,formula_execution_error_bound_us=g.error_bound_us,CPU_recurrence_certificates=g.certificates,CPU_periodic_arithmetic_only=True)
 return result
