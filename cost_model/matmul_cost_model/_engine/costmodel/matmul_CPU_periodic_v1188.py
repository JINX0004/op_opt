from pathlib import Path
"""CPU-only recurrence executor,validated1161;fixed formula is unchanged."""
from types import ModuleType,SimpleNamespace,FunctionType
from collections import deque
from functools import lru_cache
from matmul_cost_model._engine.costmodel import matmul_periodic_v978 as P,matmul_periodic_base_v978 as B,matmul_streaming_v918 as Stream

@lru_cache(maxsize=8)
def module_for(F):
 schedule=ModuleType('matmul_cost_model._engine.costmodel.private_generic_schedule1160');schedule.__package__='matmul_cost_model._engine.costmodel';text=Path(F.Schedule.__file__).read_text(encoding='utf-8')
 replacements={
  '        for ji,job in enumerate(jobs):':'        ji=0\n        while ji<len(jobs):\n            job=jobs[ji]',
  '            for ko in range(0,k,bk):':'            ko=0\n            while ko<k:',
  '                acc=cend if small_barrier else cissue;l0free[inner%l0]=cend;inner+=1':'                acc=cend if small_barrier else cissue;l0free[inner%l0]=cend;inner+=1\n                ko,inner=graph.try_skip_inner(ko+bk,bk,k,clocks,free,l0free,source,seq,inner,acc,g)',
  "        sink=graph.join([*[c['complete'] for c in clocks.values()],aiv['complete']]);cache[signature]=sink;sinks.append(sink)":"            ji,inner=graph.try_skip_jobs(ji+1,jobs,clocks,free,l0free,cfree,aiv,aiv_seen,seq,inner)\n\n        sink=graph.join([*[c['complete'] for c in clocks.values()],aiv['complete']]);cache[signature]=sink;sinks.append(sink)"}
 for a,b in replacements.items():assert text.count(a)==1;text=text.replace(a,b)
 exec(compile(text,'generic_schedule1160','exec'),schedule.__dict__)
 f=FunctionType(F.compile_graph.__code__,dict(F.__dict__,Schedule=schedule),F.compile_graph.__name__,F.compile_graph.__defaults__,F.compile_graph.__closure__);f.__kwdefaults__=F.compile_graph.__kwdefaults__
 return SimpleNamespace(**dict(F.__dict__,compile_graph=f))

def predict(F,row,model):
 if row['implementation']=='catlass' or model['schedule'].get('scalar_queue_depth',0)!=0 or model['schedule'].get('wait_location')!='pipeline':return Stream.predict(F,row,model)
 holder={}
 def factory(formula):
  class LocalGraph(Stream.graph_type(formula)):
   def __init__(self,theta):
    super().__init__(theta);holder['graph']=self;self.features=B.TraceFeatures(self);self._trace={};self._trace_order=deque();self._history={};self.skipped_nodes=0;self.skipped_inner_steps=0;self.error_bound_us=0.;self.certificates=[]
   _frontier=B.FastGraph._frontier
   try_skip_inner=B.FastGraph.try_skip_inner
   try_skip_jobs=P.FastGraph.try_skip_jobs
  return LocalGraph
 runner=FunctionType(Stream.predict.__code__,dict(Stream.__dict__,graph_type=factory),Stream.predict.__name__,Stream.predict.__defaults__,Stream.predict.__closure__)
 result=runner(module_for(F),row,model);g=holder['graph'];result.update(generated_events=result['generated_events']+g.skipped_nodes,emitted_events=len(g._finish),skipped_inner_steps=g.skipped_inner_steps,formula_execution_error_bound_us=g.error_bound_us,CPU_recurrence_certificates=g.certificates,CPU_periodic_arithmetic_only=True,CPU_prototype1160=True)
 return result
