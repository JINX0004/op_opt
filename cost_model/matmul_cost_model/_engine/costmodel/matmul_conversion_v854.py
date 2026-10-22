"""854 source-derived FP16 NN ND2NZ graph with UB group occupancy.

Bank-map source: https://www.hiascend.com/doc_center/source/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00020.html
Only UB. Read/write overlap and ISA lowering remain unverified.

No timing, PMU or case identity is consumed. Service constants are offline
inputs, in microseconds per named static unit. This is a schedule hypothesis;
source buffer ownership is explicit, effective service costs need qualification.
"""
from math import ceil,log2,isfinite
from collections import Counter
from .ops_kernel_features import _conversion

ENGINE_FEATURES={
 'mte2':('calls','mb','kbursts','strided_kbursts','partial32_kbursts'),
 'mte3':('calls','mb','kbursts','strided_kbursts','strided_mb'),
 'vector':('calls','copy_kbank_cycles','transdata_kbank_cycles','duplicate_kbank_cycles'),
 'scalar':('kloops','kaddress_assignments','kapi_calls'),
}
PARAMETER_NAMES=tuple(e+'.'+k for e,fs in ENGINE_FEATURES.items() for k in fs)+tuple(e+'.entry' for e in ENGINE_FEATURES)+('phase.entry','phase.extra_cores','phase.log2_cores')+tuple(e+'.idle_entry' for e in ENGINE_FEATURES)

def operation_features(op):
 api=op['api_work'];result={e:{} for e in ENGINE_FEATURES}
 for e in ('mte2','mte3'):
  rows=api[e]
  result[e]=dict(calls=len(rows),mb=sum(r['bursts']*r['burst_bytes'] for r in rows)/1e6,
    kbursts=sum(r['bursts'] for r in rows)/1000,
    strided_kbursts=sum(r['bursts'] for r in rows if r.get('source_stride_bytes',0) or r.get('destination_stride_bytes',0))/1000)
 result['mte2']['partial32_kbursts']=sum(r['bursts'] for r in api['mte2'] if r['api']=='DataCopyPad' and r['burst_bytes']%32)/1000
 result['mte3']['strided_mb']=sum(r['bursts']*r['burst_bytes'] for r in api['mte3'] if r.get('source_stride_bytes',0) or r.get('destination_stride_bytes',0))/1e6
 v=api['vector'];g=api['geometry'];copy_work=trans_work=duplicate_work=0.
 def occupancy(stride_bytes,blocks):
  if blocks<=0:return 0
  return max(Counter((j*stride_bytes//32)%16 for j in range(blocks)).values())
 for r in v:
  if r['elements']<=0:continue
  if r['api']=='Copy':
   blocks=ceil(r['mask_elements']/16)
   conflict=max(occupancy(32*r['src_block_stride'],blocks),occupancy(32*r['dst_block_stride'],blocks))
   copy_work+=r['repeat_times']*conflict
  elif r['api']=='TransDataTo5HD':
   # PadDMain first reads16 row pointers spaced hBlockNum*width*2B;
   # subsequent transposes read contiguous blocks and write row pointers
   # spaced hBlockNum*32B. Repeat offsets translate all groups equally.
   first=r is v[0]
   read_stride=g['h_block_num']*g['original_columns']*2 if first else 32
   write_stride=32 if first else g['h_block_num']*32
   trans_work+=r['repeat_times']*max(occupancy(read_stride,16),occupancy(write_stride,16))
  elif r['api']=='Duplicate_count':
   duplicate_work+=ceil(r['elements']/128)
  elif r['api']=='Duplicate':
   duplicate_work+=r['repeat_times']*occupancy(32*r.get('dst_block_stride',1),ceil(r['mask_elements']/16))
 result['vector']=dict(calls=sum(r['elements']>0 for r in v),copy_kbank_cycles=copy_work/1000,
   transdata_kbank_cycles=trans_work/1000,duplicate_kbank_cycles=duplicate_work/1000)
 c=api['counts'];result['scalar']=dict(kloops=c['scalar_loop_iterations']/1000,kaddress_assignments=c['scalar_address_assignments']/1000,
   kapi_calls=(c['mte2_calls']+c['mte3_calls']+c['vector_api_calls']+c['set_flag_calls']+c['wait_flag_calls']+c['pipe_barrier_calls'])/1000)
 return result

def setup_features(group):
 c=group['setup']['counts'];return {'vector':dict(calls=c['nonzero_vector_api_calls'],duplicate_kbank_cycles=ceil(c['duplicate_elements']/128)/1000),
 'scalar':dict(kloops=c['scalar_loop_iterations']/1000,kaddress_assignments=c['scalar_address_assignments']/1000,
 kapi_calls=(c['set_flag_calls']+c['wait_flag_calls']+c['pipe_barrier_calls']+c['vector_api_calls'])/1000)}

def compile_conversion(row):
 if row['implementation']=='catlass':return dict(cores=row['cores'],per_aiv=[],source_reviewed=True)
 m,n,k=row['shape'];p=row['plan'];cfg=dict(m=m,n=n,k=k,block_dim=row['cores'],nd2nz_a=bool(p['nd2nz_a']),nd2nz_b=bool(p['nd2nz_b']))
 if p.get('dtype','float16')!='float16' or p.get('trans_a',False) or p.get('trans_b',False):raise ValueError('FP16 NN conversion only')
 source=_conversion(cfg,192*1024);threads=[]
 for r in source['per_core']:
  nodes=[];last={e:-1 for e in ENGINE_FEATURES};totals={e:{} for e in ENGINE_FEATURES};drain=-1
  def node(engine,features,deps=()):
   deps=tuple(sorted({d for d in (*deps,last.get(engine,-1)) if d>=0}));idx=len(nodes)
   nodes.append(dict(engine=engine,features=features,dependencies=deps));last[engine]=idx
   for key,value in features.items():totals[engine][key]=totals[engine].get(key,0)+value
   return idx
  for group in r['conversion_groups']:
   sf=setup_features(group);s=node('scalar',sf['scalar'],(drain,));v=node('vector',sf['vector'],(s,drain))
   depth=group['buffer_depth'];input_free=[drain]*depth;output_free=[drain]*depth;tails=[]
   for ti,op in enumerate(group['operations']):
    f=operation_features(op);slot=ti%depth
    # Source Scalar instruction service can issue ahead; hardware event waits
    # constrain receiving engines, not an invented Scalar stall at every flag.
    control={key:value/3 for key,value in f['scalar'].items()}
    s0=node('scalar',control,(s,));d=node('mte2',f['mte2'],(s0,input_free[slot],drain))
    s1=node('scalar',control,(s0,));vv=node('vector',f['vector'],(s1,d,output_free[slot],v))
    s2=node('scalar',control,(s1,));out=node('mte3',f['mte3'],(s2,vv))
    input_free[slot]=vv if group['path']=='vnchw_in_out_db' else out
    output_free[slot]=out;tails.extend((s2,out));s=s2;v=vv
   # Helper returns after draining its credits; A and B reuse the same UB.
   deps=tuple(sorted({d for d in (drain,*tails,*last.values()) if d>=0}));drain=len(nodes)
   nodes.append(dict(engine='join',features={},dependencies=deps))
  threads.append(dict(aiv_id=r['core_id'],nodes=nodes,totals=totals,sink=drain))
 return dict(cores=row['cores'],per_aiv=threads,source_reviewed=True,qualification='API work and buffer dependencies reviewed; service timing is a hypothesis')

def predict_compiled(plan,weights,*,detail=False):
 if any(not isfinite(float(v)) or v<0 for v in weights.values()):raise ValueError('finite nonnegative service constants required')
 results=[]
 for t in plan['per_aiv']:
  end=[]
  for n in t['nodes']:
   e=n['engine'];cost=sum(weights.get(e+'.'+k,0)*v for k,v in n['features'].items())
   end.append(max((end[d] for d in n['dependencies']),default=0)+cost)
  # Engine entry activity is reported separately: prologue instructions may
  # overlap. Phase entry below absorbs their elapsed critical path once.
  results.append(dict(aiv_id=t['aiv_id'],pipeline_us=end[t['sink']] if t['sink']>=0 else 0,
   activity_us={e:weights.get(e+('.entry' if any(v>0 for v in f.values()) else '.idle_entry'),0)+sum(weights.get(e+'.'+k,0)*v for k,v in f.items()) for e,f in t['totals'].items()}))
 has_work=any(t['sink']>=0 for t in plan['per_aiv']);c=plan['cores']
 elapsed=(max((r['pipeline_us'] for r in results),default=0)+weights.get('phase.entry',0)+weights.get('phase.extra_cores',0)*(c-1)+weights.get('phase.log2_cores',0)*log2(c)) if has_work else 0
 return dict(conversion_us=elapsed,per_aiv=results) if detail else elapsed

def predict_conversion(row,weights,*,detail=False):return predict_compiled(compile_conversion(row),weights,detail=detail)
