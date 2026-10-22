"""Shared static ND branches plusunchanged raw-NZ service geometry."""
from . import mte2_descriptor_geometry_v847 as Old
from .nd_descriptor_work_v941 import work,II_NAMES,EXTRA_NAMES
NEW_NAMES=tuple('mte2.nd.interval.'+n for n in II_NAMES)+tuple('mte2.nd.extra_completion.'+n for n in EXTRA_NAMES)
PARAMETER_NAMES=Old.PARAMETER_NAMES+NEW_NAMES
II_GROUPS=(('dense_calls','dense_mb','dense_above32k_mb','dense_above128k_mb'),('packed_odd_head',),('packed_odd_krows',),('reader_head','reader_first','reader_repeat'),('reader_krows','reader_first_krows','reader_repeat_krows'))
II_GROUPS=tuple(group+('row_cross128_krows',) for group in II_GROUPS)
EXTRA_GROUPS=(('dense_calls','dense_mb','dense_fill8k'),('packed_odd_calls','packed_odd_fill'),('reader_calls','reader_repeats_krows'))
def descriptor_features(g,operand,offset,extent):
 features,calls=Old.descriptor_features(g,operand,offset,extent)
 if g['kind']=='catlass' or (g['nza'] if operand=='a' else g['nzb']):return features,calls
 rows,columns,pitch=(g['m'],extent,g['global_k']) if operand=='a' else (extent,g['n'],g['global_n'])
 address=512+2*(g['mo']*g['global_k']+g['ko']+offset) if operand=='a' else 512+2*((g['ko']+offset)*g['global_n']+g['no'])
 w=work(rows,columns,pitch,address%512);out={k:v for k,v in features.items() if not k.startswith('mte2.nd.')}
 cross=w['row_cross128_krows'];prefix='dma.'+g['kind']+'.'
 # Replace only the measured128B crossing part, retaining partial-row andlayout residuals.
 excess=prefix+('a_source_excess_klines' if operand=='a' else 'b_excess_klines');out[excess]=max(0.,out.get(excess,0.)-cross)
 out[prefix+operand+'_row_crossing_klines']=0.
 for phase in ('interval','extra_completion'):
  for branch in w[phase]:
   for n,v in branch.items():
    if v:out['mte2.nd.'+phase+'.'+n]=v
 return out,calls
