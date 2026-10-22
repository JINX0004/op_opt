"""Audited CPU shortcut for repeated native jobs, with static-pattern checks.

No hardware labels, no fitted additions, and no changes to frozen formulas.
All unsupported schedules fall back to the original streaming evaluator.
"""
from types import ModuleType,SimpleNamespace,FunctionType
from collections import deque
from math import ulp
from . import matmul_periodic_base_v978 as B
F=B.F;Stream=B.Stream;ROOT=B.ROOT

class FastGraph(B.FastGraph):
 def try_skip_jobs(self,next_job,jobs,clocks,free,l0free,cfree,aiv,aiv_seen,seq,inner):
  kind=jobs[0]['geometry']['kind']
  if kind not in ('scsk','deterministic'):return next_job,inner
  if next_job==1:
   self._job_history={};tokens=[]
   for ji,job in enumerate(jobs):
    g=job['geometry'];last=ji+1==len(jobs) or any(jobs[ji+1]['geometry'][x]!=g[x] for x in ('mo','no'))
    tokens.append((ji==0,last,tuple(sorted((x,v%256 if x in ('ko','mo','no') else v) for x,v in g.items()))))
   self._job_tokens=tokens
  if next_job>=len(jobs)-1:return next_job,inner
  ids=tuple([clocks[p][field] for p in ('scalar','dma','load','cube','fix') for field in ('issue','complete')]+list(free['a'])+list(free['b'])+list(l0free)+list(cfree)+[aiv['issue'],aiv['complete']])
  values=tuple(self._finish[i] for i in ids)
  phase=(next_job%len(cfree),inner%len(l0free),seq['a']%len(free['a']),seq['b']%len(free['b']),aiv_seen,self._job_tokens[next_job])
  history=self._job_history.setdefault(phase,deque(maxlen=2));record=dict(ji=next_job,ids=ids,values=values,cut=len(self._finish),inner=inner,seq=seq.copy())
  if len(history)<2:history.append(record);return next_job,inner
  first,previous=history;history.append(record);period=next_job-previous['ji']
  if period<=0 or previous['ji']-first['ji']!=period:return next_job,inner
  # Check every skipped job's complete static geometry and source cast boundary.
  pattern=self._job_tokens[previous['ji']:next_job];periods=0
  for start in range(next_job,len(jobs)-period,period):
   if self._job_tokens[start:start+period]!=pattern:break
   periods+=1
  if periods<4:return next_job,inner
  delta=tuple(b-a for a,b in zip(previous['values'],values));old_delta=tuple(b-a for a,b in zip(first['values'],previous['values']))
  worst=max(abs(a-b) for a,b in zip(delta,old_delta))
  if min(delta)<-1e-10 or worst>1e-9:return next_job,inner
  aliases=lambda vv:tuple(vv.index(v) for v in vv)
  if aliases(previous['ids'])!=aliases(ids):return next_job,inner
  inner_delta=inner-previous['inner'];seq_delta={x:seq[x]-previous['seq'][x] for x in ('a','b')}
  if inner_delta!=previous['inner']-first['inner'] or any(seq_delta[x]!=previous['seq'][x]-first['seq'][x] for x in seq):return next_job,inner
  try:trace=[self._trace[i] for i in range(previous['cut'],record['cut'])]
  except KeyError:return next_job,inner
  input_map={n:i for i,n in enumerate(previous['ids'])};head=self.head;head_value=self._finish[head]
  def replay(x):
   computed={n:x[i] for n,i in input_map.items()};computed[0]=0.;computed[head]=head_value
   for node,(parents,cost) in enumerate(trace,previous['cut']):
    if any(p not in computed for p in parents):return None
    computed[node]=max((computed[p] for p in parents),default=0.)+cost
   if any(n not in computed for n in ids):return None
   return tuple(computed[n] for n in ids)
  for q in (0,1,periods-1):
   result=replay(tuple(v+q*d for v,d in zip(values,delta)))
   if result is None:return next_job,inner
   expected=tuple(v+(q+1)*d for v,d in zip(values,delta));worst=max(worst,max(abs(a-b) for a,b in zip(result,expected)))
  future=max(v+periods*d for v,d in zip(values,delta));bound=periods*periods*worst+4*ulp(max(1.,future))*len(trace)*periods
  if bound>1e-9*max(1.,future):return next_job,inner
  replacements={}
  for n,v,d in zip(ids,values,delta):
   value=v+periods*d
   if n in (0,head):
    if d!=0:return next_job,inner
    continue
   if n in replacements and abs(replacements[n]-value)>worst+1e-12:return next_job,inner
   replacements[n]=value
  for n,v in replacements.items():self._finish[n]=v
  for x in seq:seq[x]+=periods*seq_delta[x]
  self.skipped_nodes+=periods*len(trace);self.skipped_inner_steps+=periods*inner_delta;self.error_bound_us+=bound
  self.certificates.append(dict(kind=kind,period_jobs=period,periods_skipped=periods,probe_residual_us=worst,error_bound_us=bound));self._job_history={}
  return next_job+periods*period,inner+periods*inner_delta

def formula():
 schedule=ModuleType('matmul_cost_model._engine.costmodel.private_fast_schedule968');schedule.__package__='matmul_cost_model._engine.costmodel';text=(ROOT/'matmul_schedule_v942.py').read_text()
 line='        for ji,job in enumerate(jobs):';assert text.count(line)==1;text=text.replace(line,'        ji=0\n        while ji<len(jobs):\n            job=jobs[ji]')
 line='            for ko in range(0,k,bk):';assert text.count(line)==1;text=text.replace(line,'            ko=0\n            while ko<k:')
 line='                acc=cend if small_barrier else cissue;l0free[inner%l0]=cend;inner+=1';assert text.count(line)==1
 text=text.replace(line,line+'\n                ko,inner=graph.try_skip_inner(ko+bk,bk,k,clocks,free,l0free,source,seq,inner,acc,g)')
 line='        sink=graph.join([*[c[\'complete\'] for c in clocks.values()],aiv[\'complete\']]);cache[signature]=sink;sinks.append(sink)';assert text.count(line)==1
 text=text.replace(line,'            ji,inner=graph.try_skip_jobs(ji+1,jobs,clocks,free,l0free,cfree,aiv,aiv_seen,seq,inner)\n\n'+line)
 exec(compile(text,'private_fast_schedule968','exec'),schedule.__dict__)
 globals_map=dict(F.__dict__,Schedule=schedule);compile_graph=FunctionType(F.compile_graph.__code__,globals_map,F.compile_graph.__name__,F.compile_graph.__defaults__,F.compile_graph.__closure__);compile_graph.__kwdefaults__=F.compile_graph.__kwdefaults__
 return SimpleNamespace(**dict(F.__dict__,compile_graph=compile_graph))

MODULE=formula()
def predict(row,model):
 if row['implementation']=='catlass' or model['schedule'].get('scalar_queue_depth',0)!=0 or model['schedule'].get('wait_location')!='pipeline':return Stream.predict(F,row,model)
 holder={}
 def factory(_formula):
  class LocalGraph(FastGraph):
   def __init__(self,theta):super().__init__(theta);holder['graph']=self
  return LocalGraph
 runner=FunctionType(Stream.predict.__code__,dict(Stream.__dict__,graph_type=factory),Stream.predict.__name__,Stream.predict.__defaults__,Stream.predict.__closure__)
 result=runner(MODULE,row,model);g=holder['graph'];result.update(generated_events=result['generated_events']+g.skipped_nodes,emitted_events=len(g._finish),skipped_inner_steps=g.skipped_inner_steps,formula_execution_error_bound_us=g.error_bound_us,CPU_recurrence_certificates=g.certificates,CPU_periodic_arithmetic_only=True)
 return result
