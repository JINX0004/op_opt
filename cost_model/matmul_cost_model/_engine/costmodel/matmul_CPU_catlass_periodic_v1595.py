"""CPU-only CAT packet recurrence with explicit prefetch frontier certificates.

No performance observations or model fitting. Replays the fixed event macro
at interval endpoints and an interior point before skipping any repetitions.
The final full packet and partial tail remain explicit.
"""
from types import ModuleType,SimpleNamespace,FunctionType
from collections import deque
from functools import lru_cache
from . import matmul_streaming_v918 as Stream,matmul_periodic_base_v978 as B

@lru_cache(maxsize=8)
def module_for(F):
    schedule=ModuleType('matmul_cost_model._engine.costmodel.private_CAT_schedule1595');schedule.__package__='matmul_cost_model._engine.costmodel'
    with open(F.Schedule.__file__,encoding='utf-8') as f:text=f.read()
    replacements={
        '            for ko in range(0,k,bk):':'            ko=0\n            while ko<k:',
        '                acc=cend if small_barrier else cissue;l0free[inner%l0]=cend;inner+=1':
        '                acc=cend if small_barrier else cissue;l0free[inner%l0]=cend;inner+=1\n                ko,inner=graph.try_skip_CAT(ko+bk,bk,k,clocks,free,l0free,source,seq,inner,acc,g,copies)'}
    for a,b in replacements.items():
        if text.count(a)!=1:raise ValueError('unreviewed source schedule for CAT recurrence')
        text=text.replace(a,b)
    exec(compile(text,'private_CAT_schedule1595','exec'),schedule.__dict__)
    fun=FunctionType(F.compile_graph.__code__,dict(F.__dict__,Schedule=schedule),F.compile_graph.__name__,F.compile_graph.__defaults__,F.compile_graph.__closure__)
    fun.__kwdefaults__=F.compile_graph.__kwdefaults__
    return SimpleNamespace(**dict(F.__dict__,compile_graph=fun))

def predict(F,row,model):
    schedule=model['schedule']
    if row['implementation']!='catlass' or schedule.get('scalar_queue_depth',0) or schedule.get('wait_location')!='pipeline' or schedule.get('expand_microcommands'):
        return Stream.predict(F,row,model)
    plan=F.compile_plan(row)
    if not any(j['geometry']['k']//(j['geometry']['bk']*j['geometry']['sa'])>=16 for jobs in plan['work'] for j in jobs):
        result=Stream.predict(F,row,model);result.update(CPU_CAT_recurrence_fallback='short_packet_chain');return result
    holder={}
    def factory(formula):
        class LocalGraph(Stream.graph_type(formula)):
            def __init__(self,theta):
                super().__init__(theta);holder['graph']=self;self.features=B.TraceFeatures(self)
                self._trace={};self._trace_order=deque();self._history={};self._pending_copy_ids=()
                self.skipped_nodes=0;self.skipped_inner_steps=0;self.error_bound_us=0.;self.certificates=[]
            def _frontier(self,clocks,free,l0free,source,acc):
                return B.FastGraph._frontier(self,clocks,free,l0free,source,acc)+self._pending_copy_ids
            def try_skip_CAT(self,ko_next,bk,k,clocks,free,l0free,source,seq,inner,acc,g,copies):
                if ko_next==bk:self._history={}
                span=bk*g['sa']
                if g['kind']!='catlass' or span!=bk*g['sb'] or span%256 or ko_next%span:return ko_next,inner
                if any((op,ko_next) not in copies for op in ('a','b')):return ko_next,inner
                # Preserve the last full packet: its next prefetch may be a
                # partial tail with a different service vector.
                limit=(k//span-1)*span
                if limit<=ko_next:return ko_next,inner
                self._pending_copy_ids=tuple(copies[(op,ko_next)][0] for op in ('a','b'))
                next_ko,next_inner=B.FastGraph.try_skip_inner(self,ko_next,bk,limit,clocks,free,l0free,source,seq,inner,acc,dict(g,kind='base'))
                if next_ko!=ko_next:
                    for op in ('a','b'):copies[(op,next_ko)]=copies[(op,ko_next)]
                return next_ko,next_inner
        return LocalGraph
    runner=FunctionType(Stream.predict.__code__,dict(Stream.__dict__,graph_type=factory),Stream.predict.__name__,Stream.predict.__defaults__,Stream.predict.__closure__)
    result=runner(module_for(F),row,model);g=holder['graph']
    result.update(generated_events=result['generated_events']+g.skipped_nodes,emitted_events=len(g._finish),
        skipped_inner_steps=g.skipped_inner_steps,formula_execution_error_bound_us=g.error_bound_us,
        CPU_recurrence_certificates=g.certificates,CPU_CAT_prefetch_frontier_included=True,
        CPU_periodic_arithmetic_only=True,accuracy_formula_and_fixed_parameters_UNCHANGED=True)
    return result
