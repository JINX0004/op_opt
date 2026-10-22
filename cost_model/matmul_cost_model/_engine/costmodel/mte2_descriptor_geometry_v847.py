"""847 static MTE2 descriptor work, shared across implementations and A/B roles.

FP16 NN current direct-wrapper domain. 512B source windows are an effective
grouping hypothesis, not an asserted cache line, page or bank mapping.
"""
from collections import Counter
from functools import lru_cache
from .matmul_four_path_v590 import descriptor_features as original_features,r16,cd
ND_FIELDS=('calls','logical_mb','odd_short_krows','odd_short_window_calls','odd_short_window_krows')
RAW_FIELDS=('calls','logical_mb','head_fill','drain_fill')
PARAMETER_NAMES=tuple('mte2.nd.interval.'+n for n in ND_FIELDS)+tuple('mte2.nd.extra_completion.'+n for n in ND_FIELDS)+tuple('mte2.raw.interval.'+n for n in RAW_FIELDS)+tuple('mte2.raw.extra_completion.'+n for n in RAW_FIELDS)

@lru_cache(maxsize=8192)
def nd_features(rows,columns,pitch,source_phase_bytes=0):
 if any(type(v)!=int or v<=0 for v in (rows,columns,pitch)) or pitch<columns or source_phase_bytes<0:raise ValueError('positive static ND row geometry required')
 width=columns*2;visits=Counter()
 for r in range(rows):
  start=source_phase_bytes+r*2*pitch
  visits.update(range(start//512,(start+width-1)//512+1))
 extra=max(0,max(visits.values())-1);density=max(0.,1-width/128)*((columns//16)%2)
 return dict(calls=1.,logical_mb=rows*width/1e6,odd_short_krows=rows/1000*density,odd_short_window_calls=extra*density,odd_short_window_krows=rows/1000*extra*density)

def descriptor_features(g,operand,offset,extent):
 residual,calls=original_features(g,operand,offset,extent)
 nz=g['nza'] if operand=='a' else g['nzb'];kind=g['kind']
 # CATLASS has a distinct row-wise source API at large K, retained for now.
 if kind=='catlass':return residual,calls
 if nz:
  rows,width=(g['m'],extent) if operand=='a' else (extent,g['n'])
  payload=2*r16(rows)*r16(width);per_call=payload/calls
  if per_call<512:raise ValueError('raw packet below reviewed512B minimum')
  common=dict(calls=calls,logical_mb=payload/1e6,head_fill=calls*(1-512/per_call),drain_fill=calls*min(1,per_call/8192));family='raw'
 else:
  rows,columns,pitch=(g['m'],extent,g['global_k']) if operand=='a' else (extent,g['n'],g['global_n'])
  source=(512+2*(g['mo']*g['global_k']+g['ko']+offset)) if operand=='a' else (512+2*((g['ko']+offset)*g['global_n']+g['no']))
  common=nd_features(rows,columns,pitch,source%512);family='nd'
  if calls!=1:raise ValueError('native ND descriptor expected one API call')
 # The legacy features remain named for provenance and residual controls.
 # A calibrated model must freeze intrinsic duplicates to zero explicitly.
 out=dict(residual)
 for phase in ('interval','extra_completion'):
  for name,value in common.items():
   if value:out['mte2.'+family+'.'+phase+'.'+name]=value
 return out,calls
