"""942 framework with measured-control aligned native A residual cleanup.

Constants are offline-only; no measurements are read during inference.
512B source windows are effective geometry proxies, not hardware cache claims.
"""
from math import isfinite
from . import matmul_four_path_v815 as B
from . import matmul_four_path_v740 as L
from . import matmul_four_path_v614 as Cast
from . import matmul_schedule_v1121 as Schedule
from . import mte2_descriptor_geometry_v942 as D
from .matmul_four_path_v320 import validate_prediction_contract
from . import matmul_conversion_v854 as C
from functools import lru_cache

@lru_cache(maxsize=4096)
def conversion_us(m,n,k,cores,a,b,constants):
 row=dict(implementation='base',shape=[m,n,k],cores=cores,plan=dict(dtype='float16',trans_a=False,trans_b=False,nd2nz_a=a,nd2nz_b=b))
 return C.predict_conversion(row,dict(zip(C.PARAMETER_NAMES,constants)))

FP32_FIELDS=('calls','logical_mb','partial128_k','partial128_mb','fullwin_partial256_mb','trailing_partial256_mb','shortwin_partial256_mb','row_phase256_mb','packed_start64','packed_start128','row_gaps')
FP32_FIELDS=FP32_FIELDS+tuple('atomic_'+n for n in FP32_FIELDS)+('peer_logical_mb',)
FIX_COMPLETION_NAMES=tuple('writeback.native_fp32.extra_completion.'+n for n in FP32_FIELDS)
NATIVE_FP32_INTERVAL_NAMES=tuple('writeback.native_fp32.'+n for n in FP32_FIELDS)
NATIVE_FP32_PEER_NAMES=('writeback.native_fp32.peer128_k','writeback.native_fp32.peer512_k')
INIT_LAYOUT_NAMES=tuple('init.converted.'+k for k in ('base','scsk','deterministic'))
PARAMETER_NAMES=B.PARAMETER_NAMES+D.PARAMETER_NAMES+NATIVE_FP32_INTERVAL_NAMES+NATIVE_FP32_PEER_NAMES+FIX_COMPLETION_NAMES+INIT_LAYOUT_NAMES
INDEX={n:i for i,n in enumerate(PARAMETER_NAMES)}
RESIDUAL_SUFFIXES={'peer_read_mb','peer_short_density','peer_copy_calls','peer_b_strided_mb','same_address_b_short_density','interval_b_peer_excess_short_density','b_dst_span_krows','a_dst_density','b_dst_extra_column_blocks_span_krows','preload_b_dst_span_krows','a_fill_stride_mb','a_source_excess_klines','a_row_crossing_klines','b_excess_klines','b_row_crossing_klines','b_cache_disable_mb','b_cache_disable_span_mb'}
RESIDUAL_SUFFIXES.update({'b_even_c0_strided_short_density', 'b_even_c0_unique_short_lines', 'b_strided_mb'})
FIXED_INTERVALS=dict(B.FIXED_INTERVALS)
for n in B.PARAMETER_NAMES:
 if n.startswith(('dma.base.','dma.scsk.','dma.deterministic.')) and n.split('.',2)[2] not in RESIDUAL_SUFFIXES:FIXED_INTERVALS[n]=0.
for kind in ('base','scsk','deterministic'):
 for operand in ('a','b'):FIXED_INTERVALS['dma.'+kind+'.'+operand+'_row_crossing_klines']=0.
compile_plan=B.compile_plan

def _cast_contexts(row):
    if row['implementation']!='scsk' or row['shape'][1]%128:return []
    p=Schedule.attach_output_geometry(row,compile_plan(row))
    if p['has_conversion']:return []
    seen=set();contexts=[]
    for jobs in p['work']:
        for job in jobs:job['geometry']['source_clients']=p['active_cores']
        signature=tuple(tuple(sorted((x,(v%256 if x in ('ko','mo','no') else v)) for x,v in j['geometry'].items())) for j in jobs)
        if signature in seen:continue
        seen.add(signature)
        for ji,job in enumerate(jobs):
            g=job['geometry']
            if ji+1==len(jobs) or any(jobs[ji+1]['geometry'][x]!=g[x] for x in ('mo','no')):
                contexts.append(Cast.cast_features(g,row['shape'][1]))
    return contexts

class EventGraph(B.EventGraph):
 def plain_node(self,predecessors=(),features=None,fraction=1.):
  if features and not self.head and self.initialization_converted and 'entry.'+self.initialization_kind in features:features={**features,'init.converted.'+self.initialization_kind:1.}
  ps=sorted(set(predecessors)-{0})
  if not ps and self.head:ps=[self.head]
  self.predecessors.append(ps);self.features.append([(INDEX[k],float(v)*fraction) for k,v in (features or {}).items() if v]);return len(self.features)-1
 def issue(self,clock,ready,features,fraction=1.):
  if 'mte2.nd.interval.dense_calls' in features:
   start=self.join([clock['issue'],*ready]);branches=[]
   for group in D.II_GROUPS:
    branches.append(self.node([start],{'mte2.nd.interval.'+n:features.get('mte2.nd.interval.'+n,0) for n in group}))
   residual={k:v for k,v in features.items() if not k.startswith('mte2.nd.')}
   issued=self.node(branches,residual);ends=[]
   for group in D.EXTRA_GROUPS:ends.append(self.node([issued],{'mte2.nd.extra_completion.'+n:features.get('mte2.nd.extra_completion.'+n,0) for n in group}))
   end=self.join(ends);clock['issue']=issued;clock['complete']=self.join([clock['complete'],end]);return end,issued
  if 'writeback.native_fp32.calls' in features:
   start=self.join([clock['issue'],*ready]);issued=self.node([start],features)
   completion={'writeback.native_fp32.extra_completion.'+n:features.get('writeback.native_fp32.'+n,0) for n in FP32_FIELDS}
   end=self.node([issued],completion);clock['issue']=issued;clock['complete']=self.join([clock['complete'],end]);return end,issued
  if any(k.startswith('mte2.') for k in features):
   interval={k:v for k,v in features.items() if '.extra_completion.' not in k}
   completion={k:v for k,v in features.items() if '.extra_completion.' in k}
   start=self.join([clock['issue'],*ready])
   if any(k.startswith('mte2.raw.interval.') for k in interval):
    head={k:v for k,v in interval.items() if k in ('mte2.raw.interval.calls','mte2.raw.interval.head_fill')}
    byte={k:v for k,v in interval.items() if k=='mte2.raw.interval.logical_mb'}
    residual={k:v for k,v in interval.items() if not k.startswith('mte2.raw.interval.')}
    dispatch=self.node([start],head);transfer=self.node([start],byte)
    issued=self.node([dispatch,transfer],residual)
   else:issued=self.node([start],interval)
   end=self.node([issued],completion)
   clock['issue']=issued;clock['complete']=self.join([clock['complete'],end]);return end,issued
  return super().issue(clock,ready,features,fraction)

def compile_graph(row,graph=None,*,conversion_constants,**schedule):
 validate_prediction_contract(row);graph=graph or EventGraph()
 if not isinstance(graph,EventGraph):raise ValueError('942 event graph required')
 if set(conversion_constants)!=set(C.PARAMETER_NAMES):raise ValueError('complete frozen conversion constants required')
 kind=row['implementation'];graph.initialization_kind=kind;graph.initialization_converted=kind!='catlass' and bool(row['plan']['nd2nz_a'] or row['plan']['nd2nz_b']);graph.conversion_us=0. if kind=='catlass' else conversion_us(*row['shape'],row['cores'],bool(row['plan']['nd2nz_a']),bool(row['plan']['nd2nz_b']),tuple(conversion_constants[n] for n in C.PARAMETER_NAMES))
 init=graph.plain_node(features={'entry.'+kind:1,'launch.'+kind:row['cores']});conv=graph.plain_node(features={'conversion.frozen_pipeline_us':graph.conversion_us});graph.head=graph.plain_node([init,conv])
 graph.l1_budgets=L.service_budgets(row);graph.native_load3d=kind!='catlass';graph.load3d_ii=schedule.pop('load3d_ii',.7)
 if graph.load3d_ii!=.7:raise ValueError('fixed Load3D interval implementation required')
 contexts=_cast_contexts(row);graph.cast_contexts=iter(contexts)
 result=Schedule.compile_graph(row,graph,**schedule)
 if graph.cast_count!=len(contexts) or not graph.phase_attached or graph.shared_branches!=len(graph.l1_budgets):raise RuntimeError('source phase/shared/cast attachment mismatch')
 return result

def predict(row,model):
 validate_prediction_contract(row,model);theta=model['parameters']
 if set(theta)!=set(PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid fixed offline services')
 fp32=model.get('component_fixed_native_fp32',{})
 if set(fp32)!={*FIX_COMPLETION_NAMES,*('writeback.native_fp32.'+n for n in FP32_FIELDS)}:raise ValueError('all FP32 II andcompletion services must have fixedoffline calibration')
 fixed={**FIXED_INTERVALS,**model.get('component_fixed_mte2',{}),**fp32}
 if set(model.get('component_fixed_mte2',{}))!=set(D.PARAMETER_NAMES):raise ValueError('all common MTE2 services must have fixed offline calibration')
 if any(abs(theta[n]-v)>1e-12*max(1,v) for n,v in fixed.items()):raise ValueError('fixed component service changed')
 graph,sink,p=compile_graph(row,**model['schedule']);value=graph.evaluate([theta[n] for n in PARAMETER_NAMES],sink)
 return dict(predicted_us=value,model_version=model.get('version','four_path_v942'),active_cores=p['active_cores'],launched_cores=p['cores'],split_factor=p['split_factor'],generated_events=len(graph.features),conversion_phase_us=graph.conversion_us,mte2_shared_descriptor_interval_and_completion=True,timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False,stable_ranking_qualified=False)
